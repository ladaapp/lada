import argparse
import mimetypes
import os
import pathlib
import sys
import time

import av
import torch
from tqdm import tqdm

from lada import DETECTION_MODEL_NAMES_TO_FILES, RESTORATION_MODEL_NAMES_TO_FILES, \
    get_available_restoration_models, get_available_detection_models
from lada.lib import VideoMetadata
from lada.lib.frame_restorer import FrameRestorer

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

def dump_pyav_codecs():
    print(_("PyAV version:"))
    print(f"\t{av.__version__}")

    from lada.lib.video_utils import get_available_video_encoder_codecs
    print(_("Available video encoders:"))
    for short_name, long_name in get_available_video_encoder_codecs():
        print("\t%-18s %s" % (short_name, long_name))

    try:
        from av.codec.hwaccel import hwdevices_available
        print(_("Encoders with support for hardware acceleration (GPU):"))
        for x in hwdevices_available():
            print(f"\t{x}")
    except ImportError:
        print("Unable to list available hwdevices, ImportError")

    try:
        from av.codec.codec import dump_hwconfigs
        dump_hwconfigs()
    except ImportError:
        print("Unable to list hwdevice configs, ImportError")

def dump_torch_devices():
    device_header = _("Device")
    description_header = _("Description")
    s = _("Available devices:")
    s += f"\n\t{device_header}\t{description_header}"
    s += f"\n\t{len(device_header)*"-"}\t{len(description_header)*"-"}"
    s += "\n\tcpu\tCPU"
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_properties(i).name
        s += f"\n\tcuda:{i}\t{gpu_name}"
    print(s)

def dump_available_detection_models():
    s = _("Available detection models:")
    detection_model_names = get_available_detection_models()
    if len(detection_model_names) == 0:
        s += f"\n\t{_("None!")}"
    else:
        model_name_header = _("Name")
        model_path_header = _("Path")
        s += f"\n\t{model_name_header}\t{model_path_header}"
        s += f"\n\t{len(model_name_header) * "-"}\t{len(model_path_header) * "-"}"
        for name in detection_model_names:
            s += f"\n\t{name}\t{DETECTION_MODEL_NAMES_TO_FILES[name]}"
    print(s)

def dump_available_restoration_models():
    s = _("Available restoration models:")
    restoration_model_names = get_available_restoration_models()
    if len(restoration_model_names) == 0:
        s += f"\n\t{_("None!")}"
    else:
        model_name_header = _("Name")
        model_path_header = _("Path")
        s += f"\n\t{model_name_header}\t{model_path_header}"
        s += f"\n\t{len(model_name_header) * "-"}\t{len(model_path_header) * "-"}"
        for name in restoration_model_names:
            s += f"\n\t{name}\t{RESTORATION_MODEL_NAMES_TO_FILES[name]}"
    print(s)

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

        # Use {unit} instead of {postfix} as tqdm will add an additional comma without a way to overwrite this behavior (https://github.com/tqdm/tqdm/issues/712)
        BAR_FORMAT = _("Processing video: {done_percent}%|{bar}|Processed: {time_done} ({frames_done}f){bar_suffix}")
        BAR_FORMAT_TQDM = BAR_FORMAT.format(done_percent="{percentage:3.0f}", bar="{bar}", time_done="{elapsed}", frames_done="{n_fmt}", bar_suffix="{desc}")
        self.tqdm_iterable = tqdm(frame_restorer, total=video_metadata.frames_count, bar_format=BAR_FORMAT_TQDM, desc=" | Remaining: ? | Speed: ?")
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

    def update_time_remaining_and_speed(self) -> float | None:
        frames_remaining = self.tqdm_iterable.format_dict['total']-self.tqdm_iterable.format_dict['n']
        enough_datapoints =  len(self.frame_processing_durations_buffer) > self.frame_processing_durations_buffer_min_len
        if enough_datapoints:
            mean_duration = self._get_mean_processing_duration()
            time_remaining_s = frames_remaining * mean_duration
            time_remaining = self._format_duration(time_remaining_s)
            speed_fps = f"{1. / mean_duration:.1f}"
            self.tqdm_iterable.desc = _(" | Remaining: {time_remaining} ({frames_remaining}f) | Speed: {speed_fps}fps").format(time_remaining=time_remaining, frames_remaining=frames_remaining, speed_fps=speed_fps)