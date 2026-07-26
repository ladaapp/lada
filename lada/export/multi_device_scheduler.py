# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import gc
import logging
import multiprocessing
import os
import queue
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from lada.utils.device_utils import build_worker_device_slots, safe_device_name
from lada.utils.perf_utils import get_resource_snapshot, log_json
from lada.utils.video_utils import bind_nvenc_encoder_options_to_device, get_video_meta_data


logger = logging.getLogger(__name__)


ExportEventType = Literal[
    "worker_started",
    "task_started",
    "task_progress",
    "task_finished",
    "task_failed",
    "worker_finished",
    "scheduler_finished",
    "log",
]


@dataclass(frozen=True)
class ExportTask:
    task_id: str
    input_path: str
    output_path: str


@dataclass(frozen=True)
class ExportWorkerSettings:
    base_temp_dir: str
    run_id: str
    mosaic_restoration_model_name: str = ""
    mosaic_restoration_model_path: str = ""
    mosaic_restoration_config_path: str | None = None
    mosaic_detection_model_path: str = ""
    fp16: bool = False
    detect_face_mosaics: bool = False
    max_clip_length: int = 180
    encoder: str = ""
    encoder_options: str = ""
    mp4_fast_start: bool = False
    progress_update_step_size: int = 100
    cpu_threads_per_worker: int | None = None
    log_directory: str | None = None
    perf_sample_interval_s: float | None = None


@dataclass(frozen=True)
class ExportEvent:
    event_type: ExportEventType
    worker_id: int | None = None
    task_id: str | None = None
    input_path: str | None = None
    output_path: str | None = None
    device: str | None = None
    progress: float | None = None
    frames_done: int | None = None
    frames_total: int | None = None
    temporary_directory: str | None = None
    message: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class FailedExportTask:
    task_id: str
    input_path: str
    output_path: str
    device: str | None
    error: str


@dataclass
class ExportSummary:
    total_count: int
    successful_count: int
    failed_count: int
    failed_tasks: list[FailedExportTask]
    duration_seconds: float
    cancelled: bool = False
    events: list[ExportEvent] = field(default_factory=list)


ProcessFileFunc = Callable[
    [ExportTask, str, str, ExportWorkerSettings, Callable[[float, int, int], None], Any],
    bool | None,
]


def generate_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def build_task_temp_dir(base_temp_dir: str, run_id: str, device: str, task_id: str) -> str:
    return os.path.join(
        base_temp_dir,
        "lada-export",
        run_id,
        safe_device_name(device),
        task_id,
    )


def build_worker_log_file_path(log_directory: str, run_id: str, worker_id: int, device: str) -> str:
    return str(
        Path(log_directory)
        / "workers"
        / f"lada-worker-{run_id}-w{worker_id:02d}-{safe_device_name(device)}.log"
    )


def create_export_tasks(input_files: list[str], output_files: list[str]) -> list[ExportTask]:
    if len(input_files) != len(output_files):
        raise ValueError("input_files and output_files must have the same length")
    return [
        ExportTask(f"{idx + 1:06d}", input_path, output_path)
        for idx, (input_path, output_path) in enumerate(zip(input_files, output_files))
    ]


def _emit_event(event_queue, event: ExportEvent):
    event_queue.put(event)


def build_worker_settings_for_device(settings: ExportWorkerSettings, device: str) -> ExportWorkerSettings:
    encoder_options = bind_nvenc_encoder_options_to_device(
        settings.encoder,
        settings.encoder_options,
        device,
    )
    if encoder_options == settings.encoder_options:
        return settings
    return replace(settings, encoder_options=encoder_options)


def calculate_cpu_threads_per_worker(worker_count: int, cpu_count: int | None = None) -> int:
    cpu_count = cpu_count or os.cpu_count() or 1
    return max(1, min(4, cpu_count // max(worker_count, 1)))


def _parse_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%s, using %.2f", name, value, default)
        return default


def _parse_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%s, using %d", name, value, default)
        return default


def _cuda_device_total_memory_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        if ":" in device:
            index = int(device.split(":", 1)[1])
        else:
            index = torch.cuda.current_device()
        return torch.cuda.get_device_properties(index).total_memory / (1024 * 1024)
    except Exception as e:
        logger.debug("Unable to query CUDA memory for %s: %s", device, e)
        return None


def _estimate_worker_cuda_vram_mb(task: ExportTask, settings: ExportWorkerSettings) -> float | None:
    try:
        metadata = get_video_meta_data(task.input_path)
    except Exception as e:
        logger.warning("Unable to inspect %s for GPU worker sizing: %s", task.input_path, e)
        return None

    model_mb_default = 6144 if "basicvsr" in settings.mosaic_restoration_model_name.lower() else 4096
    model_mb = _parse_env_int("LADA_GPU_WORKER_MODEL_MB", model_mb_default)
    encoder_mb = _parse_env_int("LADA_GPU_WORKER_ENCODER_MB", 512)
    frame_working_mb = metadata.video_width * metadata.video_height * 3 * 12 / (1024 * 1024)
    clip_pipeline_mb = settings.max_clip_length * 256 * 256 * 4 * 3 / (1024 * 1024)
    safety_mb = _parse_env_int("LADA_GPU_WORKER_SAFETY_MB", 512)
    return model_mb + encoder_mb + frame_working_mb + clip_pipeline_mb + safety_mb


def _adapt_jobs_per_device_for_cuda_memory(
    devices: list[str],
    tasks: list[ExportTask],
    settings: ExportWorkerSettings,
    jobs_per_device: int,
    gpu_worker_policy: str | None = None,
) -> int:
    policy = (gpu_worker_policy or os.environ.get("LADA_BATCH_GPU_WORKER_POLICY", "auto")).strip().lower()
    if policy in ("fixed", "config", "off", "disable", "disabled"):
        logger.info(
            "GPU worker sizing is fixed by configuration: using jobs_per_device=%d",
            jobs_per_device,
        )
        return jobs_per_device
    if policy in ("one-per-gpu", "single", "1"):
        if jobs_per_device != 1:
            logger.info(
                "GPU worker sizing policy one-per-gpu changed jobs_per_device from %d to 1",
                jobs_per_device,
            )
        return 1
    if policy not in ("", "auto"):
        logger.warning("Invalid GPU worker policy %s, using auto", policy)

    cuda_devices = [device for device in devices if device.startswith("cuda")]
    if jobs_per_device <= 1 or not cuda_devices or len(cuda_devices) != len(devices) or not tasks:
        return jobs_per_device

    memory_fraction = _parse_env_float("LADA_GPU_WORKER_MEMORY_FRACTION", 0.94)
    memory_fraction = min(max(memory_fraction, 0.1), 1.0)
    task_estimates = [
        estimate
        for task in tasks
        if (estimate := _estimate_worker_cuda_vram_mb(task, settings)) is not None
    ]
    if not task_estimates:
        return jobs_per_device
    estimated_worker_mb = max(task_estimates)

    device_totals = [
        total_mb
        for device in cuda_devices
        if (total_mb := _cuda_device_total_memory_mb(device)) is not None
    ]
    if not device_totals:
        return jobs_per_device
    smallest_device_mb = min(device_totals)
    requested_mb = estimated_worker_mb * jobs_per_device
    allowed_mb = smallest_device_mb * memory_fraction
    if requested_mb <= allowed_mb:
        logger.info(
            "Auto GPU worker sizing kept jobs_per_device=%d "
            "(estimated %.0fMB per worker, requested %.0fMB, allowed %.0fMB on smallest CUDA device, fraction %.2f).",
            jobs_per_device,
            estimated_worker_mb,
            requested_mb,
            allowed_mb,
            memory_fraction,
        )
        return jobs_per_device

    adapted_jobs = max(1, int(allowed_mb // max(estimated_worker_mb, 1)))
    adapted_jobs = min(adapted_jobs, jobs_per_device)
    logger.info(
        "Auto GPU worker sizing changed jobs_per_device from %d to %d "
        "(estimated %.0fMB per worker, requested %.0fMB, allowed %.0fMB on smallest CUDA device, fraction %.2f). "
        "Set LADA_BATCH_GPU_WORKER_POLICY=fixed to keep the configured value.",
        jobs_per_device,
        adapted_jobs,
        estimated_worker_mb,
        requested_mb,
        allowed_mb,
        memory_fraction,
    )
    return adapted_jobs


def configure_worker_runtime(cpu_threads_per_worker: int | None):
    if cpu_threads_per_worker is None:
        return
    thread_count = max(1, int(cpu_threads_per_worker))
    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[env_name] = str(thread_count)

    try:
        import torch

        torch.set_num_threads(thread_count)
        torch.set_num_interop_threads(max(1, min(2, thread_count)))
    except Exception:
        pass

    try:
        import cv2

        cv2.setNumThreads(thread_count)
    except Exception:
        pass


def run_export_worker(
    worker_id: int,
    device: str,
    task_queue,
    event_queue,
    cancel_event,
    settings: ExportWorkerSettings,
    process_file_func: ProcessFileFunc | None = None,
):
    worker_started_at = time.monotonic()
    worker_log_file_path = None
    tasks_started = 0
    tasks_succeeded = 0
    tasks_failed = 0
    total_frames_done = 0
    total_frames_total = 0

    if settings.log_directory:
        try:
            from lada import set_log_file

            worker_log_file_path = build_worker_log_file_path(
                settings.log_directory,
                settings.run_id,
                worker_id,
                device,
            )
            set_log_file(worker_log_file_path, propagate_directory=False)
            logger.info("Worker %s on %s logging to %s", worker_id, device, worker_log_file_path)
            _emit_event(
                event_queue,
                ExportEvent(
                    "log",
                    worker_id=worker_id,
                    device=device,
                    message=f"Worker {worker_id} on {device} logging to {worker_log_file_path}",
                ),
            )
        except Exception as e:
            _emit_event(
                event_queue,
                ExportEvent(
                    "log",
                    worker_id=worker_id,
                    device=device,
                    error=f"Failed to set worker log directory to {settings.log_directory}: {e}",
                ),
            )
    configure_worker_runtime(settings.cpu_threads_per_worker)
    worker_settings = build_worker_settings_for_device(settings, device)
    log_json(
        logger,
        "WORKER_START_JSON",
        {
            "event": "worker_start",
            "run_id": settings.run_id,
            "worker_id": worker_id,
            "device": device,
            "pid": os.getpid(),
            "log_file": worker_log_file_path,
            "cpu_threads_per_worker": worker_settings.cpu_threads_per_worker,
            "encoder": worker_settings.encoder,
            "encoder_options": worker_settings.encoder_options,
        },
    )
    _emit_event(
        event_queue,
        ExportEvent(
            "worker_started",
            worker_id=worker_id,
            device=device,
            message=(
                f"Worker {worker_id} started on {device}"
                + (
                    f" with {worker_settings.cpu_threads_per_worker} CPU threads"
                    if worker_settings.cpu_threads_per_worker is not None
                    else ""
                )
            ),
        ),
    )
    if worker_settings.encoder_options != settings.encoder_options:
        _emit_event(
            event_queue,
            ExportEvent(
                "log",
                worker_id=worker_id,
                device=device,
                message=(
                    f"Bound NVENC encoder {worker_settings.encoder} to GPU "
                    f"{worker_settings.encoder_options.rsplit(' ', 1)[-1]} for {device}"
                ),
            ),
        )

    mosaic_detection_model = None
    mosaic_restoration_model = None
    preferred_pad_mode = None

    try:
        if process_file_func is None:
            import torch

            from lada.export.single_file import process_video_file
            from lada.restorationpipeline import load_models

            torch_device = torch.device(device)
            if torch_device.type == "cuda":
                torch.cuda.set_device(torch_device)
            mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode = load_models(
                torch_device,
                worker_settings.mosaic_restoration_model_name,
                worker_settings.mosaic_restoration_model_path,
                worker_settings.mosaic_restoration_config_path,
                worker_settings.mosaic_detection_model_path,
                worker_settings.fp16,
                worker_settings.detect_face_mosaics,
            )

        while not cancel_event.is_set():
            try:
                task = task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if task is None or cancel_event.is_set():
                break

            temp_dir_path = build_task_temp_dir(
                settings.base_temp_dir,
                settings.run_id,
                device,
                task.task_id,
            )
            os.makedirs(temp_dir_path, exist_ok=True)
            tasks_started += 1
            task_started_at = time.monotonic()
            task_frames_done = 0
            task_frames_total = 0
            log_json(
                logger,
                "TASK_START_JSON",
                {
                    "event": "task_start",
                    "run_id": settings.run_id,
                    "worker_id": worker_id,
                    "task_id": task.task_id,
                    "device": device,
                    "input_path": task.input_path,
                    "output_path": task.output_path,
                    "temporary_directory": temp_dir_path,
                },
            )

            _emit_event(
                event_queue,
                ExportEvent(
                    "task_started",
                    worker_id=worker_id,
                    task_id=task.task_id,
                    input_path=task.input_path,
                    output_path=task.output_path,
                    device=device,
                    temporary_directory=temp_dir_path,
                    progress=0.0,
                    message=f"Started {task.input_path} on {device}",
                ),
            )

            def progress_callback(progress: float, frames_done: int, frames_total: int):
                nonlocal task_frames_done, task_frames_total
                task_frames_done = frames_done
                task_frames_total = frames_total
                _emit_event(
                    event_queue,
                    ExportEvent(
                        "task_progress",
                        worker_id=worker_id,
                        task_id=task.task_id,
                        input_path=task.input_path,
                        output_path=task.output_path,
                        device=device,
                        progress=progress,
                        frames_done=frames_done,
                        frames_total=frames_total,
                        temporary_directory=temp_dir_path,
                    ),
                )
                if cancel_event.is_set():
                    raise RuntimeError("Export cancelled")

            try:
                if process_file_func is not None:
                    result = process_file_func(
                        task,
                        temp_dir_path,
                        device,
                        worker_settings,
                        progress_callback,
                        cancel_event,
                    )
                    success = True if result is None else bool(result)
                else:
                    success = process_video_file(
                        input_path=task.input_path,
                        output_path=task.output_path,
                        temp_dir_path=temp_dir_path,
                        device=device,
                        mosaic_restoration_model=mosaic_restoration_model,
                        mosaic_detection_model=mosaic_detection_model,
                        mosaic_restoration_model_name=worker_settings.mosaic_restoration_model_name,
                        preferred_pad_mode=preferred_pad_mode,
                        max_clip_length=worker_settings.max_clip_length,
                        encoder=worker_settings.encoder,
                        encoder_options=worker_settings.encoder_options,
                        mp4_fast_start=worker_settings.mp4_fast_start,
                        progress_callback=progress_callback,
                        progress_update_step_size=worker_settings.progress_update_step_size,
                        raise_on_error=True,
                        print_status=False,
                        perf_metadata={
                            "run_id": settings.run_id,
                            "worker_id": worker_id,
                            "task_id": task.task_id,
                            "worker_log_file": worker_log_file_path,
                        },
                        perf_sample_interval_s=worker_settings.perf_sample_interval_s,
                    )

                task_elapsed_s = time.monotonic() - task_started_at
                total_frames_done += task_frames_done
                total_frames_total += task_frames_total
                if success:
                    tasks_succeeded += 1
                    log_json(
                        logger,
                        "TASK_SUMMARY_JSON",
                        {
                            "event": "task_summary",
                            "run_id": settings.run_id,
                            "worker_id": worker_id,
                            "task_id": task.task_id,
                            "device": device,
                            "input_path": task.input_path,
                            "output_path": task.output_path,
                            "success": True,
                            "elapsed_s": task_elapsed_s,
                            "frames_done": task_frames_done,
                            "frames_total": task_frames_total,
                            "fps_average": task_frames_done / task_elapsed_s if task_elapsed_s > 0 else None,
                            **get_resource_snapshot(device),
                        },
                    )
                    _emit_event(
                        event_queue,
                        ExportEvent(
                            "task_finished",
                            worker_id=worker_id,
                            task_id=task.task_id,
                            input_path=task.input_path,
                            output_path=task.output_path,
                            device=device,
                            progress=1.0,
                            temporary_directory=temp_dir_path,
                            message=f"Finished {task.input_path}",
                        ),
                    )
                else:
                    tasks_failed += 1
                    log_json(
                        logger,
                        "TASK_SUMMARY_JSON",
                        {
                            "event": "task_summary",
                            "run_id": settings.run_id,
                            "worker_id": worker_id,
                            "task_id": task.task_id,
                            "device": device,
                            "input_path": task.input_path,
                            "output_path": task.output_path,
                            "success": False,
                            "elapsed_s": task_elapsed_s,
                            "frames_done": task_frames_done,
                            "frames_total": task_frames_total,
                            "error": "Video export failed",
                            **get_resource_snapshot(device),
                        },
                    )
                    _emit_event(
                        event_queue,
                        ExportEvent(
                            "task_failed",
                            worker_id=worker_id,
                            task_id=task.task_id,
                            input_path=task.input_path,
                            output_path=task.output_path,
                            device=device,
                            temporary_directory=temp_dir_path,
                            error="Video export failed",
                        ),
                    )
            except Exception as e:
                task_elapsed_s = time.monotonic() - task_started_at
                total_frames_done += task_frames_done
                total_frames_total += task_frames_total
                tasks_failed += 1
                if cancel_event.is_set():
                    log_json(
                        logger,
                        "TASK_SUMMARY_JSON",
                        {
                            "event": "task_summary",
                            "run_id": settings.run_id,
                            "worker_id": worker_id,
                            "task_id": task.task_id,
                            "device": device,
                            "input_path": task.input_path,
                            "output_path": task.output_path,
                            "success": False,
                            "cancelled": True,
                            "elapsed_s": task_elapsed_s,
                            "frames_done": task_frames_done,
                            "frames_total": task_frames_total,
                            **get_resource_snapshot(device),
                        },
                    )
                    _emit_event(
                        event_queue,
                        ExportEvent(
                            "log",
                            worker_id=worker_id,
                            task_id=task.task_id,
                            input_path=task.input_path,
                            output_path=task.output_path,
                            device=device,
                            temporary_directory=temp_dir_path,
                            message="Task stopped after cancellation request",
                        ),
                    )
                    break
                log_json(
                    logger,
                    "TASK_SUMMARY_JSON",
                    {
                        "event": "task_summary",
                        "run_id": settings.run_id,
                        "worker_id": worker_id,
                        "task_id": task.task_id,
                        "device": device,
                        "input_path": task.input_path,
                        "output_path": task.output_path,
                        "success": False,
                        "elapsed_s": task_elapsed_s,
                        "frames_done": task_frames_done,
                        "frames_total": task_frames_total,
                        "error": str(e),
                        **get_resource_snapshot(device),
                    },
                )
                _emit_event(
                    event_queue,
                    ExportEvent(
                        "task_failed",
                        worker_id=worker_id,
                        task_id=task.task_id,
                        input_path=task.input_path,
                        output_path=task.output_path,
                        device=device,
                        temporary_directory=temp_dir_path,
                        error=str(e),
                    ),
                )
    except Exception as e:
        _emit_event(
            event_queue,
            ExportEvent(
                "log",
                worker_id=worker_id,
                device=device,
                error=str(e),
                message=f"Worker {worker_id} failed on {device}",
            ),
        )
    finally:
        del mosaic_detection_model
        del mosaic_restoration_model
        gc.collect()
        try:
            import torch

            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif device.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()
            elif device == "mps" and getattr(torch, "mps", None) is not None:
                torch.mps.empty_cache()
        except Exception:
            pass
        worker_elapsed_s = time.monotonic() - worker_started_at
        log_json(
            logger,
            "WORKER_SUMMARY_JSON",
            {
                "event": "worker_summary",
                "run_id": settings.run_id,
                "worker_id": worker_id,
                "device": device,
                "pid": os.getpid(),
                "log_file": worker_log_file_path,
                "elapsed_s": worker_elapsed_s,
                "tasks_started": tasks_started,
                "tasks_succeeded": tasks_succeeded,
                "tasks_failed": tasks_failed,
                "frames_done": total_frames_done,
                "frames_total": total_frames_total,
                "fps_average": total_frames_done / worker_elapsed_s if worker_elapsed_s > 0 else None,
                **get_resource_snapshot(device),
            },
        )
        _emit_event(
            event_queue,
            ExportEvent(
                "worker_finished",
                worker_id=worker_id,
                device=device,
                message=f"Worker {worker_id} finished on {device}",
            ),
        )


class MultiDeviceExportScheduler:
    def __init__(
        self,
        input_files: list[str],
        output_files: list[str],
        devices: list[str],
        settings: ExportWorkerSettings,
        parallel: int | None = None,
        jobs_per_device: int = 1,
        allow_parallel_cpu: bool = False,
        worker_target: Callable[..., None] = run_export_worker,
        process_file_func: ProcessFileFunc | None = None,
        multiprocessing_context=None,
        worker_shutdown_timeout_s: float = 5.0,
        gpu_worker_policy: str | None = None,
    ):
        self.tasks = create_export_tasks(input_files, output_files)
        self.devices = devices
        adapted_jobs_per_device = _adapt_jobs_per_device_for_cuda_memory(
            devices,
            self.tasks,
            settings,
            jobs_per_device,
            gpu_worker_policy,
        )
        self.worker_devices = build_worker_device_slots(
            devices,
            jobs_per_device=adapted_jobs_per_device,
            parallel=parallel,
            allow_parallel_cpu=allow_parallel_cpu,
        )
        if settings.cpu_threads_per_worker is None:
            settings = replace(
                settings,
                cpu_threads_per_worker=calculate_cpu_threads_per_worker(len(self.worker_devices)),
            )
        self.settings = settings
        self.worker_target = worker_target
        self.process_file_func = process_file_func
        self.multiprocessing_context = multiprocessing_context
        self.worker_shutdown_timeout_s = worker_shutdown_timeout_s
        self._cancel_event = None
        self._processes = []

    @property
    def worker_count(self) -> int:
        return len(self.worker_devices)

    def cancel(self):
        if self._cancel_event is not None:
            self._cancel_event.set()

    def _terminate_lingering_processes(self):
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(timeout=1.0)

    def run(self, event_callback: Callable[[ExportEvent], None] | None = None) -> ExportSummary:
        ctx = self.multiprocessing_context or multiprocessing.get_context("spawn")
        task_queue = ctx.Queue()
        event_queue = ctx.Queue()
        self._cancel_event = ctx.Event()
        self._processes = []

        for task in self.tasks:
            task_queue.put(task)
        for _ in self.worker_devices:
            task_queue.put(None)

        start_time = time.monotonic()
        events: list[ExportEvent] = []
        succeeded_task_ids: set[str] = set()
        failed_tasks_by_id: dict[str, FailedExportTask] = {}
        finished_worker_ids: set[int] = set()
        cancel_started_at: float | None = None

        def handle_event(event: ExportEvent):
            events.append(event)
            if event_callback is not None:
                event_callback(event)
            if event.event_type == "task_finished" and event.task_id:
                succeeded_task_ids.add(event.task_id)
            elif event.event_type == "task_failed" and event.task_id:
                failed_tasks_by_id[event.task_id] = FailedExportTask(
                    task_id=event.task_id,
                    input_path=event.input_path or "",
                    output_path=event.output_path or "",
                    device=event.device,
                    error=event.error or "Video export failed",
                )
            elif event.event_type == "worker_finished" and event.worker_id is not None:
                finished_worker_ids.add(event.worker_id)

        for worker_id, device in enumerate(self.worker_devices):
            process = ctx.Process(
                target=self.worker_target,
                args=(
                    worker_id,
                    device,
                    task_queue,
                    event_queue,
                    self._cancel_event,
                    self.settings,
                    self.process_file_func,
                ),
                name=f"lada-export-worker-{worker_id}-{safe_device_name(device)}",
            )
            process.start()
            self._processes.append(process)

        try:
            while len(finished_worker_ids) < len(self._processes):
                try:
                    event = event_queue.get(timeout=0.1)
                    handle_event(event)
                except queue.Empty:
                    pass

                for worker_id, process in enumerate(self._processes):
                    if worker_id in finished_worker_ids:
                        continue
                    if process.exitcode is not None:
                        if process.exitcode != 0:
                            handle_event(
                                ExportEvent(
                                    "log",
                                    worker_id=worker_id,
                                    device=self.worker_devices[worker_id],
                                    error=f"Worker exited with code {process.exitcode}",
                                )
                            )
                        finished_worker_ids.add(worker_id)

                if self._cancel_event.is_set():
                    if cancel_started_at is None:
                        cancel_started_at = time.monotonic()
                    elif time.monotonic() - cancel_started_at > self.worker_shutdown_timeout_s:
                        for process in self._processes:
                            if process.is_alive():
                                process.terminate()
        except KeyboardInterrupt:
            self.cancel()
            self._terminate_lingering_processes()
            raise

        for process in self._processes:
            process.join()

        # Drain any events emitted just before a process exited.
        while True:
            try:
                handle_event(event_queue.get_nowait())
            except queue.Empty:
                break

        cancelled = self._cancel_event.is_set()
        if not cancelled:
            processed_task_ids = succeeded_task_ids.union(failed_tasks_by_id.keys())
            for task in self.tasks:
                if task.task_id not in processed_task_ids:
                    failed_tasks_by_id[task.task_id] = FailedExportTask(
                        task_id=task.task_id,
                        input_path=task.input_path,
                        output_path=task.output_path,
                        device=None,
                        error="Task was not processed",
                    )

        duration_seconds = time.monotonic() - start_time
        summary = ExportSummary(
            total_count=len(self.tasks),
            successful_count=len(succeeded_task_ids),
            failed_count=len(failed_tasks_by_id),
            failed_tasks=list(failed_tasks_by_id.values()),
            duration_seconds=duration_seconds,
            cancelled=cancelled,
            events=events,
        )
        scheduler_event = ExportEvent(
            "scheduler_finished",
            progress=1.0,
            message=(
                f"Finished export: {summary.successful_count} succeeded, "
                f"{summary.failed_count} failed in {summary.duration_seconds:.1f}s"
            ),
        )
        handle_event(scheduler_event)
        summary.events = events
        return summary
