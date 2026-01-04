# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import mimetypes
import os
import pathlib
import subprocess
import sys
import time

import torch
from tqdm import tqdm
from wcwidth import wcswidth

from lada import ModelFiles
from lada.utils import VideoMetadata, video_utils

COL_SEP = "  "

def wcrjust(text, length, padding=' '):
    return text + padding * max(0, (length - wcswidth(text)))

def _filter_video_files(directory_path: str):
    video_files = []
    for name in os.listdir(directory_path):
        path = os.path.join(directory_path, name)
        if not os.path.isfile(path):
            continue
        if sys.version_info >= (3, 13):
            mime_type, _ = mimetypes.guess_file_type(path)
        else:
            mime_type, _ = mimetypes.guess_type(path)
        if not mime_type:
            continue
        if not mime_type.lower().startswith("video/"):
            continue
        video_files.append(path)
    return video_files

def _get_output_file_path(input_file_path: str, output_directory: str, output_file_pattern: str):
    output_file_name = output_file_pattern.replace("{orig_file_name}", pathlib.Path(input_file_path).stem)
    return os.path.join(output_directory, output_file_name)

def setup_input_and_output_paths(input_arg, output_arg, output_file_pattern):
    single_file_input = os.path.isfile(input_arg)

    if single_file_input:
        input_files = [os.path.abspath(input_arg)]
    else:
        input_files = _filter_video_files(input_arg)

    if len(input_files) == 0:
        print(_("No video files found"))
        sys.exit(1)

    if single_file_input:
        if not output_arg:
            input_file_path = input_files[0]
            output_dir_path = str(pathlib.Path(input_file_path).parent)
            output_files = [_get_output_file_path(input_file_path, output_dir_path, output_file_pattern)]
        elif os.path.isdir(output_arg):
            input_file_path = input_files[0]
            output_files = [_get_output_file_path(input_file_path, output_arg, output_file_pattern)]
        else:
            output_files = [output_arg]
    else:
        if output_arg:
            if not os.path.exists(output_arg):
                os.makedirs(output_arg)
            output_dir_path = output_arg
        else:
            output_dir_path = str(pathlib.Path(input_files[0]).parent)
        output_files = [_get_output_file_path(input_file_path, output_dir_path, output_file_pattern) for input_file_path in input_files]

    assert len(input_files) == len(output_files)

    return input_files, output_files

def _dump_table(table):
    row_count = len(table)
    col_count = len(table[0])
    col_widths = [0] * col_count
    for row in table:
        for col_i, col in enumerate(row):
            col_widths[col_i] = max(wcswidth(col), col_widths[col_i])
    s = ""
    for row_i, row in enumerate(table):
        for col_i, col in enumerate(row):
            s += f"{COL_SEP}{wcrjust(col, col_widths[col_i])}"
        if row_i < (row_count - 1):
            s += "\n"
        if row_i == 0:
            for col_i, col in enumerate(row):
                s += f"{COL_SEP}{col_widths[col_i] * "-"}"
            s += "\n"
    print(s)

def dump_encoders():
    from lada.utils.video_utils import get_video_encoder_codecs
    encoders = get_video_encoder_codecs()
    print(_("Available video encoders:"))
    table = [[_("Name"), _("Description"), _("Hardware-accelerated"), _("Hardware devices")]]
    for e in encoders:
        hardware = _("Yes") if e.hardware_encoder else ""
        devices = str(e.hardware_devices) if len(e.hardware_devices) > 0 else ""
        table.append([e.name, e.long_name, hardware, devices])
    _dump_table(table)

def dump_torch_devices():
    print(_("Available devices:"))
    cuda_device_count = torch.cuda.device_count()
    devices = ["cpu"] + [f"cuda:{i}" for i in range(cuda_device_count)]
    descriptions = ["CPU"] + [torch.cuda.get_device_properties(i).name for i in range(cuda_device_count)]
    table = [[_("Device"), _("Description")]]
    for device, description in zip(devices, descriptions):
        table.append([device, description])
    _dump_table(table)

def dump_available_detection_models():
    modelfiles = ModelFiles.get_detection_models()
    print(_("Available detection models:"))
    if len(modelfiles) == 0:
        print(f"{COL_SEP}{_("None!")}")
    else:
        table = [[_("Name"), _("Description"), _("Path")]]
        for modelfile in modelfiles:
            table.append([modelfile.name, modelfile.description if modelfile.description else "", modelfile.path])
        _dump_table(table)

def dump_available_restoration_models():
    modelfiles = ModelFiles.get_restoration_models()
    print(_("Available restoration models:"))
    if len(modelfiles) == 0:
        print(f"{COL_SEP}{_("None!")}")
    else:
        table = [[_("Name"), _("Description"), _("Path")]]
        for modelfile in modelfiles:
            table.append([modelfile.name, modelfile.description if modelfile.description else "", modelfile.path])
        _dump_table(table)

def dump_available_encoding_presets():
    print(_("Available encoding presets:"))
    encoding_presets = video_utils.get_encoding_presets()
    if len(encoding_presets) == 0:
        s += f"\n{COL_SEP}{_("None!")}"
    else:
        table = [[_("Name"), _("Description")]]
        for preset in encoding_presets:
            table.append([preset.name, preset.description])
        _dump_table(table)

def dump_encoder_options(encoder: str):
    result = subprocess.run(["ffmpeg", "-loglevel", "quiet", "-h", f"encoder={encoder}"], capture_output=True, text=True)
    text = result.stdout.strip().replace("Exiting with exit code 0", "").strip()
    print(text)

class TranslatableHelpFormatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, *args, **kwargs):
        super(TranslatableHelpFormatter, self).__init__(*args, **kwargs)

    def add_usage(self, usage, actions, groups, prefix=None):
        prefix = _("Usage: ")
        args = usage, actions, groups, prefix
        self._add_item(self._format_usage, args)

class Progressbar:
    def __init__(self, video_metadata: VideoMetadata):
        self.frame_processing_durations_buffer = []
        self.video_metadata = video_metadata
        self.frame_processing_durations_buffer_min_len = min(video_metadata.frames_count - 1, int(video_metadata.video_fps * 15))
        self.frame_processing_durations_buffer_max_len = min(video_metadata.frames_count - 1, int(video_metadata.video_fps * 120))
        self.error = False

        # Use {unit} instead of {postfix} as tqdm will add an additional comma without a way to overwrite this behavior (https://github.com/tqdm/tqdm/issues/712)
        BAR_FORMAT = _("Processing video: {done_percent}%|{bar}|Processed: {time_done} ({frames_done}f){bar_suffix}")
        BAR_FORMAT_TQDM = BAR_FORMAT.format(done_percent="{percentage:3.0f}", bar="{bar}", time_done="{elapsed}", frames_done="{n_fmt}", bar_suffix="{desc}")
        initial_estimating_bar_suffix = _(" | Remaining: ? | Speed: ?")
        self.tqdm = tqdm(dynamic_ncols=True, total=video_metadata.frames_count, bar_format=BAR_FORMAT_TQDM, desc=initial_estimating_bar_suffix)
        self.duration_start = None

    def init(self):
        self.duration_start = time.time()

    def close(self, ensure_completed_bar=False):
        if ensure_completed_bar:
            # On some media files the frame count, which is used to set up total of the progressbar, is not correct.
            # To prevent not showing not 100% completed bar update total to actual number of frames and refresh before closing
            if not self.error and self.tqdm.total != self.tqdm.n:
                self.tqdm.total = self.tqdm.n
                self._update_time_remaining_and_speed(completed=True)
                self.tqdm.refresh()
        self.tqdm.close()

    def update(self):
        duration_end = time.time()
        duration = duration_end - self.duration_start
        self.duration_start = duration_end

        if len(self.frame_processing_durations_buffer) >= self.frame_processing_durations_buffer_max_len:
            self.frame_processing_durations_buffer.pop(0)
        self.frame_processing_durations_buffer.append(duration)

        self._update_time_remaining_and_speed()

        self.tqdm.update()

    def _get_mean_processing_duration(self):
        return sum(self.frame_processing_durations_buffer) / len(self.frame_processing_durations_buffer)

    def _format_duration(self, duration_s):
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

    def _update_time_remaining_and_speed(self, completed=False) -> float | None:
        frames_remaining = self.tqdm.format_dict['total'] - self.tqdm.format_dict['n'] if not completed else 0
        enough_datapoints =  len(self.frame_processing_durations_buffer) > self.frame_processing_durations_buffer_min_len
        if enough_datapoints:
            mean_duration = self._get_mean_processing_duration()
            time_remaining_s = frames_remaining * mean_duration
            time_remaining = self._format_duration(time_remaining_s)
            speed_fps = f"{1. / mean_duration:.1f}"
            self.tqdm.desc = _(" | Remaining: {time_remaining} ({frames_remaining}f) | Speed: {speed_fps}fps").format(time_remaining=time_remaining, frames_remaining=frames_remaining, speed_fps=speed_fps)