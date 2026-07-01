# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import os
import pathlib
import logging
from collections.abc import Callable
from typing import Any

from lada.restorationpipeline.frame_restorer import FrameRestorer
from lada.utils import audio_utils
from lada.utils.perf_utils import StageTimer
from lada.utils.threading_utils import STOP_MARKER, ErrorMarker
from lada.utils.video_utils import get_video_meta_data, VideoWriter


logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, int, int], None]
ProgressBarFactory = Callable[[Any], Any]


def process_video_file(
    input_path: str,
    output_path: str,
    temp_dir_path: str,
    device,
    mosaic_restoration_model,
    mosaic_detection_model,
    mosaic_restoration_model_name,
    preferred_pad_mode,
    max_clip_length,
    encoder: str,
    encoder_options: str,
    mp4_fast_start,
    progress_bar_factory: ProgressBarFactory | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_update_step_size: int = 100,
    raise_on_error: bool = False,
    print_status: bool = True,
) -> bool:
    video_metadata = get_video_meta_data(input_path)
    frames_total = max(video_metadata.frames_count, 1)
    frame_restorer = None
    progressbar = None
    success = True
    error_message = None
    frames_done = 0
    stage_timer = StageTimer(
        f"Export[{os.path.basename(input_path)}]",
        metadata={
            "input_path": input_path,
            "output_path": output_path,
            "device": str(device),
            "encoder": encoder,
            "encoder_options": encoder_options,
            "mp4_fast_start": mp4_fast_start,
            "frames_total": video_metadata.frames_count,
            "video_width": video_metadata.video_width,
            "video_height": video_metadata.video_height,
            "video_fps": video_metadata.video_fps_exact,
        },
    )

    video_tmp_file_output_path = os.path.join(
        temp_dir_path,
        f"{os.path.basename(os.path.splitext(output_path)[0])}.tmp{os.path.splitext(output_path)[1]}",
    )
    pathlib.Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    pathlib.Path(temp_dir_path).mkdir(exist_ok=True, parents=True)

    if progress_bar_factory is not None:
        progressbar = progress_bar_factory(video_metadata)

    try:
        frame_restorer = FrameRestorer(
            device,
            input_path,
            max_clip_length,
            mosaic_restoration_model_name,
            mosaic_detection_model,
            mosaic_restoration_model,
            preferred_pad_mode,
        )
        with stage_timer.measure("frame_restorer_start"):
            frame_restorer.start()
        if progressbar is not None:
            progressbar.init()
        with VideoWriter(
            video_tmp_file_output_path,
            video_metadata.video_width,
            video_metadata.video_height,
            video_metadata.video_fps_exact,
            encoder=encoder,
            encoder_options=encoder_options,
            time_base=video_metadata.time_base,
            mp4_fast_start=mp4_fast_start,
        ) as video_writer:
            frame_iterator = iter(frame_restorer)
            while True:
                try:
                    with stage_timer.measure("frame_restorer_next"):
                        elem = next(frame_iterator)
                except StopIteration:
                    break
                if elem is STOP_MARKER or isinstance(elem, ErrorMarker):
                    success = False
                    error_message = "frame restorer stopped prematurely"
                    if progressbar is not None:
                        progressbar.error = True
                    if print_status:
                        print("Error on export: frame restorer stopped prematurely")
                    break
                restored_frame, restored_frame_pts = elem
                with stage_timer.measure("video_writer_write"):
                    video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)
                frames_done += 1
                if progressbar is not None:
                    progressbar.update()
                if (
                    progress_callback is not None
                    and progress_update_step_size > 0
                    and frames_done % progress_update_step_size == 0
                ):
                    progress_callback(
                        min(frames_done / frames_total, 1.0),
                        frames_done,
                        video_metadata.frames_count,
                    )
    except (Exception, KeyboardInterrupt) as e:
        success = False
        error_message = str(e)
        if isinstance(e, KeyboardInterrupt):
            raise e
        else:
            if print_status:
                print("Error on export", e)
    finally:
        if frame_restorer is not None:
            frame_restorer.stop()
        if progressbar is not None:
            progressbar.close(ensure_completed_bar=success)

    if success:
        if progress_callback is not None:
            progress_callback(1.0, frames_done, video_metadata.frames_count)
        if print_status:
            print(_("Processing audio"))
        with stage_timer.measure("audio_video_mux"):
            audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, output_path)
    else:
        if os.path.exists(video_tmp_file_output_path):
            os.remove(video_tmp_file_output_path)

    stage_timer.log_summary(logger)

    if not success and raise_on_error:
        raise RuntimeError(error_message or "Video export failed")

    return success
