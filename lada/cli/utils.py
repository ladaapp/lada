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

from lada import MODEL_WEIGHTS_DIR, ModelFiles, ModelFile
from lada.utils import VideoMetadata, video_utils
from lada.restorationpipeline.frame_restorer import FrameRestorer

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

def dump_encoders():
    from lada.utils.video_utils import get_video_encoder_codecs
    encoders = get_video_encoder_codecs()
    is_hardware_accelerated = _("Yes")
    name_header = _("Name")
    description_header = _("Description")
    hardware_header = _("Hardware-accelerated")
    devices_header = _("Hardware devices")
    name_column_width = max([len(e.name) for e in encoders] + [len(name_header)])
    description_column_width = max([len(e.long_name) for e in encoders] + [len(description_header)])
    hardware_column_width = max([len(is_hardware_accelerated), len(hardware_header)])
    devices_column_width = max([len(e.hardware_devices) for e in encoders] + [len(devices_header)])
    s = _("Available video encoders:")
    s += f"\n\t{name_header.ljust(name_column_width)}\t{description_header.ljust(description_column_width)}\t{hardware_header.ljust(hardware_column_width)}\t{devices_header}"
    s += f"\n\t{name_column_width*"-"}\t{description_column_width*"-"}\t{hardware_column_width*"-"}\t{devices_column_width*"-"}"
    for e in encoders:
        hardware = is_hardware_accelerated if e.hardware_encoder else ""
        s += f"\n\t{e.name.ljust(name_column_width)}\t{e.long_name.ljust(description_column_width)}\t{hardware.ljust(hardware_column_width)}\t{e.hardware_devices if len(e.hardware_devices) > 0 else ''}"
    print(s)

def dump_torch_devices():
    cuda_device_count = torch.cuda.device_count()
    devices = ["cpu"] + [f"cuda:{i}" for i in range(cuda_device_count)]
    descriptions = ["CPU"] + [torch.cuda.get_device_properties(i).name for i in range(cuda_device_count)]
    device_header = _("Device")
    description_header = _("Description")
    device_header_width = max([len(item) for item in devices + [device_header]])
    description_header_width = max([len(item) for item in descriptions + [description_header]])
    s = _("Available devices:")
    s += f"\n\t{device_header.ljust(device_header_width)}\t{description_header}"
    s += f"\n\t{device_header_width*"-"}\t{description_header_width*"-"}"
    for device, description in zip(devices, descriptions):
        s += f"\n\t{device.ljust(device_header_width)}\t{description}"
    print(s)

def _dump_available_models(modelfiles: list[ModelFile]):
    s = _("Model weights directory:")
    s += "\n\t" + os.path.abspath(MODEL_WEIGHTS_DIR)
    s += "\n" + _("Available restoration models:")
    if len(modelfiles) == 0:
        s += f"\n\t{_("None!")}"
    else:
        model_name_header = _("Name")
        model_description_header = _("Description")
        model_path_header = _("Path")
        model_name_column_width = max([len(item.name) for item in modelfiles] + [len(model_name_header)])
        model_description_column_width = max([len(item.description) if item.description else 0 for item in modelfiles] + [len(model_name_header)])
        model_path_column_width = max([len(item.path) for item in modelfiles] + [len(model_path_header)])
        s += f"\n\t{model_name_header.ljust(model_name_column_width)}\t{model_description_header.ljust(model_description_column_width)}\t{model_path_header}"
        s += f"\n\t{model_name_column_width * "-"}\t{model_description_column_width * "-"}\t{model_path_column_width * "-"}"
        for modelfile in modelfiles:
            s += f"\n\t{modelfile.name.ljust(model_name_column_width)}\t{(modelfile.description if modelfile.description else "").ljust(model_description_column_width)}\t{modelfile.path}"
    print(s)

def dump_available_detection_models():
    _dump_available_models(ModelFiles.get_detection_models())

def dump_available_restoration_models():
    _dump_available_models(ModelFiles.get_restoration_models())

def dump_available_encoding_presets(show_encoder_details=False):
    s = _("Available encoding presets:")
    encoding_presets = video_utils.get_encoding_presets()
    if len(encoding_presets) == 0:
        s += f"\n\t{_("None!")}"
    else:
        preset_name_header = _("Name")
        preset_description_header = _("Description")
        encoder_name_header = _("Encoder Name")
        encoder_options_header = _("Encoder Options")
        preset_name_column_width = max([len(preset.name) for preset in encoding_presets] + [len(preset_name_header)])
        preset_description_column_width = max([len(preset.description) for preset in encoding_presets] + [len(preset_description_header)])
        encoder_name_column_width = max([len(preset.encoder_name) for preset in encoding_presets] + [len(encoder_name_header)])
        encoder_options_column_width = max([len(preset.encoder_options) for preset in encoding_presets] + [len(encoder_options_header)])
        if show_encoder_details:
            s += f"\n\t{preset_name_header.ljust(preset_name_column_width)}\t{preset_description_header.ljust(preset_description_column_width)}\t{encoder_name_header.ljust(encoder_name_column_width)}\t{encoder_options_header}"
            s += f"\n\t{preset_name_column_width * "-"}\t{preset_description_column_width * "-"}\t{encoder_name_column_width * "-"}\t{encoder_options_column_width * "-"}"
            for preset in encoding_presets:
                s += f"\n\t{preset.name.ljust(preset_name_column_width)}\t{preset.description.ljust(preset_description_column_width)}\t{preset.encoder_name.ljust(encoder_name_column_width)}\t{preset.encoder_options}"
        else:
            s += f"\n\t{preset_name_header.ljust(preset_name_column_width)}\t{preset_description_header}"
            s += f"\n\t{preset_name_column_width * "-"}\t{preset_description_column_width * "-"}"
            for preset in encoding_presets:
                s += f"\n\t{preset.name.ljust(preset_name_column_width)}\t{preset.description.ljust(preset_description_column_width)}"
    print(s)

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
    def __init__(self, video_metadata: VideoMetadata, frame_restorer: FrameRestorer):
        self.frame_processing_durations_buffer = []
        self.video_metadata = video_metadata
        self.frame_processing_durations_buffer_min_len = min(video_metadata.frames_count - 1, int(video_metadata.video_fps * 15))
        self.frame_processing_durations_buffer_max_len = min(video_metadata.frames_count - 1, int(video_metadata.video_fps * 120))
        self.error = False

        # Use {unit} instead of {postfix} as tqdm will add an additional comma without a way to overwrite this behavior (https://github.com/tqdm/tqdm/issues/712)
        BAR_FORMAT = _("Processing video: {done_percent}%|{bar}|Processed: {time_done} ({frames_done}f){bar_suffix}")
        BAR_FORMAT_TQDM = BAR_FORMAT.format(done_percent="{percentage:3.0f}", bar="{bar}", time_done="{elapsed}", frames_done="{n_fmt}", bar_suffix="{desc}")
        initial_estimating_bar_suffix = _(" | Remaining: ? | Speed: ?")
        self.tqdm_iterable = tqdm(frame_restorer, dynamic_ncols=True, total=video_metadata.frames_count, bar_format=BAR_FORMAT_TQDM, desc=initial_estimating_bar_suffix)
        orig_close = self.tqdm_iterable.close
        def ensure_completed_bar_then_close():
            # On some media files the frame count, which is used to set up total of the progressbar, is not correct.
            # To prevent not showing not 100% completed bar update total to actual number of frames and refresh before closing
            if not self.error and self.tqdm_iterable.total != self.tqdm_iterable.n:
                self.tqdm_iterable.total = self.tqdm_iterable.n
                self.update_time_remaining_and_speed(completed=True)
                self.tqdm_iterable.refresh()
            orig_close()
        self.tqdm_iterable.close = ensure_completed_bar_then_close
        self.duration_start = None

    def __iter__(self):
        self.duration_start = time.time()
        return self.tqdm_iterable.__iter__()

    def update(self):
        duration_end = time.time()
        duration = duration_end - self.duration_start
        self.duration_start = duration_end

        if len(self.frame_processing_durations_buffer) >= self.frame_processing_durations_buffer_max_len:
            self.frame_processing_durations_buffer.pop(0)
        self.frame_processing_durations_buffer.append(duration)

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

    def update_time_remaining_and_speed(self, completed=False) -> float | None:
        frames_remaining = self.tqdm_iterable.format_dict['total']-self.tqdm_iterable.format_dict['n'] if not completed else 0
        enough_datapoints =  len(self.frame_processing_durations_buffer) > self.frame_processing_durations_buffer_min_len
        if enough_datapoints:
            mean_duration = self._get_mean_processing_duration()
            time_remaining_s = frames_remaining * mean_duration
            time_remaining = self._format_duration(time_remaining_s)
            speed_fps = f"{1. / mean_duration:.1f}"
            self.tqdm_iterable.desc = _(" | Remaining: {time_remaining} ({frames_remaining}f) | Speed: {speed_fps}fps").format(time_remaining=time_remaining, frames_remaining=frames_remaining, speed_fps=speed_fps)