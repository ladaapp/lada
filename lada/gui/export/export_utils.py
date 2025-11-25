# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from dataclasses import dataclass
from fractions import Fraction

from gi.repository import Gtk, Adw, Gio

from lada.gui.export.export_item_data import ExportItemDataProgress, ExportItemState
from lada.utils import VideoMetadata
from lada.utils import video_utils

MIN_VISIBLE_PROGRESS_FRACTION = 0.01

def _format_duration(duration_s):
    if not duration_s or duration_s == -1:
        return "0:00"
    seconds = int(duration_s)
    minutes = int(seconds / 60)
    hours = int(minutes / 60)
    seconds = seconds % 60
    minutes = minutes % 60
    hours, minutes, seconds = int(hours), int(minutes), int(seconds)
    time = f"{minutes}:{seconds:02d}" if hours == 0 else f"{hours}:{minutes:02d}:{seconds:02d}"
    return time

def get_video_metadata_string(file: Gio.File):
    meta_data = video_utils.get_video_meta_data(file.get_path())
    return _("Duration: {duration}, Resolution: {resolution}, Frame rate: {fps} FPS").format(
        duration=_format_duration(meta_data.duration),
        resolution=f"{meta_data.video_width}x{meta_data.video_height}",
        fps=f"{meta_data.video_fps_exact:.2f}")

def open_error_dialog(parent: Gtk.Widget, filename:str, details:str|None):
    extra_child = None
    if details:
        textview = Gtk.TextView()
        PADDING = 6
        textview.set_left_margin(PADDING)
        textview.set_right_margin(PADDING)
        textview.set_top_margin(PADDING)
        textview.set_bottom_margin(PADDING)
        textbuffer = textview.get_buffer()
        textbuffer.set_text(details.strip())

        scrolledwindow = Gtk.ScrolledWindow()
        scrolledwindow.props.hexpand = True
        scrolledwindow.props.vexpand = True
        scrolledwindow.set_child(textview)
        extra_child = scrolledwindow

    dialog = Adw.AlertDialog(
        heading=_("Restoration failed"),
        body=_("Error while processing <span><tt>{filename}</tt></span>").format(filename=filename),
        close_response="okay",
        extra_child=extra_child,
        body_use_markup=True
    )

    def on_response_selected(_dialog, task):
        _dialog.choose_finish(task)

    dialog.add_response("okay", _("Okay"))

    dialog.choose(parent, None, on_response_selected)

def get_progressbar_text(state: ExportItemState, progress: ExportItemDataProgress):
    if state == ExportItemState.FAILED:
        text = _("Failed")
    elif state == ExportItemState.QUEUED:
        text = ""
    elif state == ExportItemState.FINISHED:
        time_done = _format_duration(progress._time_done_s)
        text = _("Finished  |  Processed: {time_done} ({frames_done} frames)").format(time_done=time_done, frames_done=progress.frames_done)
    elif state == ExportItemState.PROCESSING:
        done_fraction = max(MIN_VISIBLE_PROGRESS_FRACTION, progress.fraction)
        time_done = _format_duration(progress._time_done_s)
        done_percent = f"{(done_fraction * 100):3.0f}"
        if progress.enough_datapoints:
            time_remaining = _format_duration(progress.time_remaining_s)
            speed_fps = f"{progress.speed_fps:.1f}"
            text = _("Processing… {done_percent}%  |  Processed: {time_done} ({frames_done} frames)  |  Remaining: {time_remaining} ({frames_remaining} frames)  |  Speed: {speed_fps} FPS").format(
                done_percent=done_percent,
                time_done=time_done,
                time_remaining=time_remaining,
                frames_done=progress.frames_done,
                frames_remaining=progress.frames_remaining,
                speed_fps=speed_fps)
        else:
            text = _("Processing… {done_percent}%  |  Processed: {time_done} ({frames_done} frames)  |  Remaining: Estimating… |  Speed: Estimating…").format(
                done_percent=done_percent,
                time_done=time_done,
                frames_done=progress.frames_done)
    elif state == ExportItemState.PAUSED:
        time_done = _format_duration(progress._time_done_s)
        text = _("Paused  |  Processed: {time_done} ({frames_done} frames)").format(time_done=time_done, frames_done=progress.frames_done)
    else:
        raise ValueError(f"Unknown ExportItemState: {state}")
    return text

class ProgressCalculator:
    def __init__(self, video_metadata: VideoMetadata):
        self.frame_processing_durations_buffer = []
        self.time_done_s = 0
        self.frames_done = 0
        self.video_metadata = video_metadata
        self.frame_processing_durations_buffer_min_len = min(video_metadata.frames_count - 1, int(video_metadata.video_fps * 15))
        self.frame_processing_durations_buffer_max_len = min(video_metadata.frames_count - 1, int(video_metadata.video_fps * 120))

    def _get_mean_processing_duration(self):
        return sum(self.frame_processing_durations_buffer) / len(self.frame_processing_durations_buffer)

    def update(self, duration):
        if len(self.frame_processing_durations_buffer) >= self.frame_processing_durations_buffer_max_len:
            self.frame_processing_durations_buffer.pop(0)
        self.frame_processing_durations_buffer.append(duration)
        self.time_done_s += duration
        self.frames_done += 1

    def get_progress(self) -> ExportItemDataProgress:
        progress = ExportItemDataProgress()
        progress.time_done_s = self.time_done_s
        progress.frames_done = self.frames_done
        progress.fraction = progress.frames_done / self.video_metadata.frames_count
        progress.frames_remaining = self.video_metadata.frames_count - progress.frames_done
        progress.enough_datapoints =  len(self.frame_processing_durations_buffer) > self.frame_processing_durations_buffer_min_len
        if progress.enough_datapoints:
            mean_duration = self._get_mean_processing_duration()
            progress.time_remaining_s = progress.frames_remaining * mean_duration
            progress.speed_fps = 1. / mean_duration
        return progress

@dataclass
class ResumeInformation:
    frame_pts: int
    time_base: Fraction
    frame_num : int

    def get_resume_timestamp_ns(self):
        SECOND = 1_000_000_000
        return int((self.frame_pts * self.time_base) * SECOND)