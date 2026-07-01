# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import ctypes
import json
import logging
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


DEFAULT_PERF_SAMPLE_INTERVAL_S = 30.0


@dataclass
class StageTiming:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0


def get_perf_sample_interval_s(default: float = DEFAULT_PERF_SAMPLE_INTERVAL_S) -> float:
    try:
        return max(0.0, float(os.environ.get("LADA_PERF_SAMPLE_INTERVAL_S", default)))
    except ValueError:
        return default


def _get_windows_rss_mb() -> float | None:
    try:
        size_t = ctypes.c_size_t

        class ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", size_t),
                ("WorkingSetSize", size_t),
                ("QuotaPeakPagedPoolUsage", size_t),
                ("QuotaPagedPoolUsage", size_t),
                ("QuotaPeakNonPagedPoolUsage", size_t),
                ("QuotaNonPagedPoolUsage", size_t),
                ("PagefileUsage", size_t),
                ("PeakPagefileUsage", size_t),
                ("PrivateUsage", size_t),
            ]

        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(ProcessMemoryCountersEx)
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        process = kernel32.GetCurrentProcess()
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCountersEx),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        if psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        return None
    return None


def get_process_rss_mb() -> float | None:
    if sys.platform == "win32":
        return _get_windows_rss_mb()

    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except Exception:
        pass

    try:
        import resource

        max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return max_rss / (1024 * 1024)
        return max_rss / 1024
    except Exception:
        return None


def get_torch_device_memory(device: str | None) -> dict[str, float | int | str]:
    if not device:
        return {}
    try:
        import torch

        torch_device = torch.device(device)
        if torch_device.type == "cuda" and torch.cuda.is_available():
            index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
            return {
                "gpu_backend": "cuda",
                "gpu_index": index,
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated(index) / (1024 * 1024),
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved(index) / (1024 * 1024),
                "gpu_max_memory_allocated_mb": torch.cuda.max_memory_allocated(index) / (1024 * 1024),
                "gpu_max_memory_reserved_mb": torch.cuda.max_memory_reserved(index) / (1024 * 1024),
            }
        if torch_device.type == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            index = torch_device.index if torch_device.index is not None else 0
            memory_allocated = getattr(torch.xpu, "memory_allocated", None)
            memory_reserved = getattr(torch.xpu, "memory_reserved", None)
            stats: dict[str, float | int | str] = {
                "gpu_backend": "xpu",
                "gpu_index": index,
            }
            if memory_allocated is not None:
                stats["gpu_memory_allocated_mb"] = memory_allocated(index) / (1024 * 1024)
            if memory_reserved is not None:
                stats["gpu_memory_reserved_mb"] = memory_reserved(index) / (1024 * 1024)
            return stats
    except Exception:
        return {}
    return {}


def log_json(logger: logging.Logger, marker: str, payload: dict[str, Any], level: int = logging.INFO):
    logger.log(level, "%s %s", marker, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


class PerformanceSampler:
    def __init__(
        self,
        name: str,
        logger: logging.Logger,
        metadata: dict[str, Any] | None = None,
        device: str | None = None,
        interval_s: float | None = None,
    ):
        self.name = name
        self.logger = logger
        self.metadata = metadata or {}
        self.device = device
        self.interval_s = get_perf_sample_interval_s() if interval_s is None else max(0.0, interval_s)
        self.started_at_s = time.time()
        self._start_wall = time.monotonic()
        self._start_process_time = time.process_time()
        self._last_wall = self._start_wall
        self._last_process_time = self._start_process_time
        self._last_frames_done = 0

    def maybe_log(
        self,
        frames_done: int,
        frames_total: int,
        progress: float | None = None,
        force: bool = False,
    ):
        if self.interval_s <= 0 and not force:
            return

        now = time.monotonic()
        interval_s = now - self._last_wall
        if not force and interval_s < self.interval_s:
            return

        elapsed_s = now - self._start_wall
        process_time = time.process_time()
        process_time_delta_s = process_time - self._last_process_time
        frames_delta = max(0, frames_done - self._last_frames_done)
        payload: dict[str, Any] = {
            "event": "performance_sample",
            "timer": self.name,
            "metadata": self.metadata,
            "pid": os.getpid(),
            "elapsed_s": elapsed_s,
            "interval_s": interval_s,
            "frames_done": frames_done,
            "frames_total": frames_total,
            "frames_delta": frames_delta,
            "fps_interval": frames_delta / interval_s if interval_s > 0 else None,
            "fps_average": frames_done / elapsed_s if elapsed_s > 0 else None,
            "progress": progress,
            "cpu_process_percent": (process_time_delta_s / interval_s) * 100 if interval_s > 0 else None,
            "process_time_delta_s": process_time_delta_s,
            "rss_mb": get_process_rss_mb(),
        }
        payload.update(get_torch_device_memory(self.device))
        log_json(self.logger, "PERF_SAMPLE_JSON", payload)

        self._last_wall = now
        self._last_process_time = process_time
        self._last_frames_done = frames_done


class StageTimer:
    def __init__(self, name: str, metadata: dict[str, object] | None = None):
        self.name = name
        self.metadata = metadata or {}
        self.started_at_s = time.time()
        self._stats: dict[str, StageTiming] = {}
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, time.perf_counter() - start)

    def record(self, stage: str, duration_s: float):
        with self._lock:
            timing = self._stats.setdefault(stage, StageTiming())
            timing.count += 1
            timing.total_s += duration_s
            timing.max_s = max(timing.max_s, duration_s)

    def log_summary(self, logger: logging.Logger, level: int = logging.INFO):
        with self._lock:
            stats = dict(self._stats)
        if not stats:
            return

        lines = []
        for stage, timing in sorted(stats.items(), key=lambda item: item[1].total_s, reverse=True):
            avg_ms = (timing.total_s / timing.count) * 1000
            lines.append(
                f"    {stage}: count={timing.count}, "
                f"total={timing.total_s:.3f}s, avg={avg_ms:.2f}ms, max={timing.max_s:.3f}s"
            )
        logger.log(level, "%s stage timings:\n%s", self.name, "\n".join(lines))
        log_json(logger, "PERF_SUMMARY_JSON", self.to_summary_dict(stats), level=level)

    def to_summary_dict(self, stats: dict[str, StageTiming] | None = None) -> dict[str, object]:
        if stats is None:
            with self._lock:
                stats = dict(self._stats)
        stages = []
        for stage, timing in sorted(stats.items(), key=lambda item: item[1].total_s, reverse=True):
            avg_ms = (timing.total_s / timing.count) * 1000
            stages.append(
                {
                    "stage": stage,
                    "count": timing.count,
                    "total_s": timing.total_s,
                    "avg_ms": avg_ms,
                    "max_s": timing.max_s,
                }
            )
        return {
            "event": "performance_summary",
            "timer": self.name,
            "metadata": self.metadata,
            "pid": os.getpid(),
            "elapsed_s": time.time() - self.started_at_s,
            "stages": stages,
        }
