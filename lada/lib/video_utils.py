import json
import os
import re
import subprocess
from contextlib import contextmanager
from fractions import Fraction
from typing import Callable

import av
import cv2
import numpy as np

from lada.lib import Image, Mask, VideoMetadata, os_utils


def read_video_frames(path: str, float32: bool = True, start_idx: int = 0, end_idx: int | None = None, normalize_neg1_pos1 = False, binary_frames=False) -> list[np.ndarray]:
    with VideoReaderOpenCV(path) as video_reader:
        frames = []
        i = 0
        while video_reader.isOpened():
            ret, frame = video_reader.read()
            if ret and (end_idx is None or i < end_idx):
                if binary_frames:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = np.expand_dims(frame, axis=-1)
                if i >= start_idx:
                    if float32:
                        if normalize_neg1_pos1:
                            frame = (frame.astype(np.float32) / 255.0 - 0.5) / 0.5
                        else:
                            frame = frame.astype(np.float32) / 255.
                    frames.append(frame)
                i += 1
            else:
                break
    return frames

def resize_video_frames(frames: list, size: int | tuple[int, int]):
    resized = []
    target_size = size if isinstance(size, (list, tuple)) else (size, size)
    for frame in frames:
        if frame.shape[:2] == target_size:
            resized.append(frame)
        else:
            resized.append(cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR))
    return resized

def pad_to_compatible_size_for_video_codecs(imgs):
    # dims need to be divisible by 2 by most codecs. given the chroma / pix format dims must be divisible by 4
    h, w = imgs[0].shape[:2]
    pad_h = 0 if h % 4 == 0 else 4 - (h % 4)
    pad_w = 0 if w % 4 == 0 else 4 - (w % 4)
    if pad_h == 0 and pad_w == 0:
        return imgs
    else:
        return [np.pad(img, ((0, pad_h), (0, pad_w), (0,0))).astype(np.uint8) for img in imgs]

@contextmanager
def VideoReaderOpenCV(*args, **kwargs):
    cap = cv2.VideoCapture(*args, **kwargs)
    if not cap.isOpened():
        raise Exception(f"Unable to open video file:", *args)
    try:
        yield cap
    finally:
        cap.release()

class VideoReader:
    def __init__(self, file):
        self.file = file
        self.container = None

    def __enter__(self):
        self.container = av.open(self.file)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.container.close()

    def frames(self):
        for frame in self.container.decode(video=0):
            frame_img = frame.to_ndarray(format='bgr24')
            yield frame_img, frame.pts, frame.dts

    def seek(self, offset_ns):
        offset = int((offset_ns / 1_000_000_000) * av.time_base)
        self.container.seek(offset)

def get_video_meta_data(path: str) -> VideoMetadata:
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-select_streams', 'v', '-show_streams', '-show_format', path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=os_utils.get_subprocess_startup_info())
    out, err =  p.communicate()
    if p.returncode != 0:
        raise Exception(f"error running ffprobe: {err.strip()}. Code: {p.returncode}, cmd: {cmd}")
    json_output = json.loads(out)
    json_video_stream = json_output["streams"][0]
    json_video_format = json_output["format"]

    value = [int(num) for num in json_video_stream['avg_frame_rate'].split("/")]
    average_fps = value[0]/value[1] if len(value) == 2 else value[0]

    value = [int(num) for num in json_video_stream['r_frame_rate'].split("/")]
    fps = value[0]/value[1] if len(value) == 2 else value[0]
    fps_exact = Fraction(value[0], value[1])

    value = [int(num) for num in json_video_stream['time_base'].split("/")]
    time_base = Fraction(value[0], value[1])

    frame_count = json_video_stream.get('nb_frames')
    if not frame_count:
        # print("frame count ffmpeg", frame_count)
        cap = cv2.VideoCapture(path)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        # print("frame count opencv", frame_count)
    frame_count=int(frame_count)

    start_pts = json_video_stream.get('start_pts')

    metadata = VideoMetadata(
        video_file=path,
        video_height=int(json_video_stream['height']),
        video_width=int(json_video_stream['width']),
        video_fps=fps,
        average_fps=average_fps,
        video_fps_exact=fps_exact,
        codec_name=json_video_stream['codec_name'],
        frames_count=frame_count,
        duration=float(json_video_stream.get('duration', json_video_format['duration'])),
        time_base=time_base,
        start_pts=start_pts
    )
    return metadata

def offset_ns_to_frame_num(offset_ns, video_fps_exact):
    return int(Fraction(offset_ns, 1_000_000_000) * video_fps_exact)

def write_frames_to_video_file(frames: list[Image], output_path, fps: int | float | Fraction, codec='x264', preset='medium', crf=None):
    assert frames[0].ndim == 3
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    ffmpeg_output = [
        'nice', '-n', '19', 'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', f"{fps.numerator}/{fps.denominator}" if type(fps) == Fraction else str(fps),
        '-i', '-', '-an', '-preset', preset
    ]
    if codec == 'x265':
        ffmpeg_output.extend(['-tag:v', 'hvc1', '-vcodec', 'libx265', '-crf', str(crf) if crf else '18'])
    elif codec == 'x264':
        ffmpeg_output.extend(['-vcodec', 'libx264', '-crf', str(crf) if crf else '15'])
    ffmpeg_output.append(output_path)

    ffmpeg_process = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE, startupinfo=os_utils.get_subprocess_startup_info())
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ffmpeg_process.stdin.write(frame.tobytes())
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    if ffmpeg_process.returncode != 0:
        print(f"ERROR when writing video via ffmpeg to file: {output_path}, return code: {ffmpeg_process.returncode}")
        print(f"stderr: {ffmpeg_process.stderr.read()}")

def write_masks_to_video_file(frames: list[Mask], output_path, fps: int | float | Fraction):
    #assert frames[0].ndim == 2
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    ffmpeg_output = [
        'nice', '-n', '19', 'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{width}x{height}', '-r', f"{fps.numerator}/{fps.denominator}" if type(fps) == Fraction else str(fps),
        '-i', '-', '-an', '-vcodec', 'ffv1', '-level', '3', '-tag:v', 'ffv1',  output_path
    ]

    ffmpeg_process = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=os_utils.get_subprocess_startup_info())
    for frame in frames:
        try:
            ffmpeg_process.stdin.write(frame.tobytes())
        except Exception as e:
            print(f"ERROR when writing video via ffmpeg to file: {output_path}")
            print(f"exception: {e}")
            print(f"stderr: {ffmpeg_process.stderr.read()}")
            print(f"stdout: {ffmpeg_process.stdout.read()}")
            raise e
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    if ffmpeg_process.returncode != 0:
        print(f"ERROR when writing video via ffmpeg to file: {output_path}, return code: {ffmpeg_process.returncode}")
        print(f"stderr: {ffmpeg_process.stderr.read()}")
        print(f"stdout: {ffmpeg_process.stdout.read()}")

def process_video_v3(input_path, output_path, frame_processor: Callable[[Image], Image]):
    video_metadata = get_video_meta_data(input_path)
    video_reader = cv2.VideoCapture(input_path)
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps=video_metadata.video_fps, frameSize=(video_metadata.video_width, video_metadata.video_height))
    while video_reader.isOpened():
        ret, frame = video_reader.read()
        if ret:
            processed_frame = frame_processor(frame)
            video_writer.write(processed_frame)
        else:
            break
    video_reader.release()
    video_writer.release()

def approx_memory(video_metadata: VideoMetadata, frames_count, assume_images=True, assume_masks=True):
    size = 0
    frame_size_image = video_metadata.video_width * video_metadata.video_height * 3 * 1
    frame_size_mask = video_metadata.video_width * video_metadata.video_height * 1 * 1
    if assume_images:
        size += frame_size_image * frames_count
    if assume_masks:
        size += frame_size_mask * frames_count
    return size

def approx_max_length_by_memory_limit(video_metadata: VideoMetadata, limit_in_megabytes, assume_images=True, assume_masks=True):
    frame_size_image = approx_memory(video_metadata, 1, assume_images=assume_images, assume_masks=assume_masks)
    max_length_frames = (limit_in_megabytes * 1024 * 1024) / frame_size_image
    max_length_seconds = int(max_length_frames / video_metadata.video_fps)
    return max_length_seconds

class VideoWriter:
    def parse_custom_options(self, custom_encoder_options):
        # squeeze spaces
        custom_encoder_options = ' '.join(custom_encoder_options.split())
        regex = re.compile(r"-(\w+ \w+)")
        matches = regex.findall(custom_encoder_options)
        encoder_options = {}
        for match in matches:
            option, value = match.split()
            encoder_options[option] = value
        return encoder_options

    def get_default_encoder_options(self):
        libx264 = {
            'preset': 'medium',
            'crf': '20'
        }
        libx265 = {
            'preset': 'medium',
            'crf': '23',
            'x265-params': 'log_level=error'
        }
        encoder_defaults = {}
        encoder_defaults['libx264'] = libx264
        encoder_defaults['h264'] = libx264
        encoder_defaults['libx265'] = libx265
        encoder_defaults['hevc'] = libx265
        return encoder_defaults

    def __init__(self, output_path, width, height, fps, codec, crf=None, preset=None, time_base=None, moov_front=False, custom_encoder_options=None):
        container_options = {"movflags": "+frag_keyframe+empty_moov+faststart"} if moov_front else {}
        encoder_defaults = self.get_default_encoder_options()
        encoder_options = encoder_defaults.get(codec, {})

        if crf is not None:
            if codec in ('hevc_nvenc', 'h264_nvenc'):
                encoder_options['rc'] = 'constqp'
                encoder_options['qp'] = str(crf)
            else:
                encoder_options['crf'] = str(crf)
        if preset:
            encoder_options['preset'] = preset

        if custom_encoder_options:
            encoder_options.update(self.parse_custom_options(custom_encoder_options))

        output_container = av.open(output_path, "w", options=container_options)
        video_stream_out: av.VideoStream = output_container.add_stream(codec, fps)

        video_stream_out.width = width
        video_stream_out.height = height
        video_stream_out.thread_count = 0
        video_stream_out.thread_type = 3
        video_stream_out.time_base = time_base

        # up until PyAV 15.5.0 it was enough to set these settings on the stream only.
        video_stream_out.codec_context.width = width
        video_stream_out.codec_context.height = height
        video_stream_out.codec_context.thread_count = 0
        video_stream_out.codec_context.thread_type = 3
        video_stream_out.codec_context.time_base = time_base

        video_stream_out.options = encoder_options
        self.output_container = output_container
        self.video_stream = video_stream_out
        self.frame_number = 0
        self.fps = fps

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def calculate_pts(self, frame_number):
        return round(frame_number * self.video_stream.time_base.denominator / self.fps)

    def write(self, frame, original_pts = None, original_dts = None, bgr2rgb=False):
        if bgr2rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')

        if original_pts is not None:
            out_frame.pts = original_pts
        if original_dts is not None:
            out_frame.dts = original_dts

        out_packet = self.video_stream.encode(out_frame)
        self.output_container.mux(out_packet)
        self.frame_number += 1

    def release(self):
        out_packet = self.video_stream.encode(None)
        self.output_container.mux(out_packet)
        self.output_container.close()

def is_video_file(file_path):
    SUPPORTED_VIDEO_FILE_EXTENSIONS = {".asf", ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".ts", ".wmv",
                                       ".webm"}

    file_ext = os.path.splitext(file_path)[1]
    return file_ext.lower() in SUPPORTED_VIDEO_FILE_EXTENSIONS

def get_available_video_encoder_codecs():
    codecs = set()
    for name in av.codec.codecs_available:
        try:
            e_codec = av.codec.Codec(name, "w")
        except ValueError:
            continue
        if e_codec.type != 'video':
            continue
        codecs.add((e_codec.name, e_codec.long_name))
    return sorted(list(codecs))
