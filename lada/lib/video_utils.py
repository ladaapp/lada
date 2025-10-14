import json
import os
import re
import subprocess
from contextlib import contextmanager
from fractions import Fraction
from typing import Callable, Iterator

import cv2
import numpy as np
import PyNvVideoCodec as nvc
import torch
import pycuda.driver as cuda
from lada.lib import Image, Mask, VideoMetadata
import av

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
    def __init__(self, file, batch_size=4, cuda_ctx=None, model_stream=None):
        self.file = file
        self.container = None
        self.batch_size = batch_size
        self.cuda_ctx = cuda_ctx
        self.model_stream = model_stream

    def __enter__(self):
        self.demuxer = nvc.CreateDemuxer(filename=self.file)
        self.decoder: nvc.PyNvDecoder = nvc.CreateDecoder(
            gpuid=0,
            cudacontext=self.cuda_ctx.handle,
            cudastream=self.model_stream.cuda_stream,
            codec=self.demuxer.GetNvCodecId(),
            usedevicememory=True,
            outputColorType=nvc.OutputColorType.RGBP
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def frames(self) -> Iterator[tuple[torch.Tensor, int]]:
        for packet in self.demuxer:
            for frame in self.decoder.Decode(packet):
                yield torch.from_dlpack(frame).permute(1, 2, 0).clone(), packet.pts

    def seek(self, offset_ns):
        index = self.decoder.get_index_from_time_in_seconds(offset_ns)
        self.decoder.seek_to_index(index)

def get_video_meta_data(path: str) -> VideoMetadata:
    cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-select_streams', 'v', '-show_streams', '-show_format', path]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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

    ffmpeg_process = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
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

    ffmpeg_process = subprocess.Popen(ffmpeg_output, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
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

    def __init__(self, output_path, width, height, fps, cuda_ctx=None, modelstream=None):
        config = {
            "preset": "P6",
            "codec": "h264",
            # "tuning_info": "high_quality",
            # "rc": "vbr",
            # "gop": 30,
            # "bf": 3,
            # "bitrate": "10M",
            # "vbvinit" : 0,
            # "vbvbufsize" : 0,
            # "qmin"         : "0,0,0",
            # "qmax"         : "0,0,0",
            # "initqp"       : "0,0,0",
            "qp": "20"
        } 
        self.encoder = nvc.CreateEncoder(
            width=width,
            height=height,
            cudacontext=cuda_ctx.handle,
            cudastream=modelstream.cuda_stream,
            fmt="YUV444",
            usecpuinputbuffer=False,
            fps=fps,
            **config)
        self.output_file = open(output_path, "wb")

    def hwc_rgb_to_yuv_bt709(self, img_hwc: torch.Tensor) -> torch.Tensor:
        """
        Convert HWC RGB uint8 tensor [H,W,3] to planar YUV uint8 tensor [3*H,W].
        Uses BT.601 color space with limited range (Y: 16-235, UV: 16-240) to avoid contrast issues.
        Output format is planar: Y plane, then U plane, then V plane stacked vertically.
        
        Args:
            img_hwc: Input tensor with shape [H,W,3] in RGB order
            
        Returns:
            torch.Tensor: Output tensor with shape [3*H,W] in planar YUV format
        """
        assert img_hwc.ndim == 3 and img_hwc.shape[2] == 3, "Input must be HWC RGB with 3 channels"
        H, W, _ = img_hwc.shape

        # Convert to float for precise calculations
        rgb = img_hwc.float()
        
        # Extract RGB channels
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        
        # BT.601 RGB to YUV conversion (limited range 16-235/16-240)
        # Limited range is what most video encoders expect to avoid contrast issues
        # Y =  16 + (219/255) * (0.299*R + 0.587*G + 0.114*B)
        # U = 128 + (224/255) * (-0.169*R - 0.331*G + 0.500*B)
        # V = 128 + (224/255) * (0.500*R - 0.419*G - 0.081*B)
        
        # First convert RGB (0-255) to normalized (0-1)
        R_norm = R / 255.0
        G_norm = G / 255.0
        B_norm = B / 255.0
        
        # Apply BT.601 matrix to get YUV in normalized range
        Y_norm = 0.299 * R_norm + 0.587 * G_norm + 0.114 * B_norm
        U_norm = -0.169 * R_norm - 0.331 * G_norm + 0.500 * B_norm
        V_norm = 0.500 * R_norm - 0.419 * G_norm - 0.081 * B_norm
        
        # Convert to limited range
        Y = 16 + 219 * Y_norm          # Y: 16-235 range
        U = 128 + 224 * U_norm         # U: 16-240 range (128 ± 112)
        V = 128 + 224 * V_norm         # V: 16-240 range (128 ± 112)
        
        # Clamp to valid limited range and convert back to uint8
        Y = Y.clamp(16, 235).to(torch.uint8)
        U = U.clamp(16, 240).to(torch.uint8)
        V = V.clamp(16, 240).to(torch.uint8)
        
        # Create planar YUV format: stack Y, U, V planes vertically
        # Result shape: [3*H, W] where:
        # - Rows 0 to H-1: Y plane
        # - Rows H to 2*H-1: U plane  
        # - Rows 2*H to 3*H-1: V plane
        yuv_planar = torch.cat([Y, U, V], dim=0)  # [3*H, W]
        
        return yuv_planar

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def write(self, frame, frame_pts=None):
        # x = frame.cpu().numpy()
        # x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("debug_frame.png", x)
        nv12_frame = self.hwc_rgb_to_yuv_bt709(frame)
        bitstream = self.encoder.Encode(nv12_frame)
        self.output_file.write(bytearray(bitstream))

    def release(self):
        bitstream = self.encoder.EndEncode()
        self.output_file.write(bytearray(bitstream))
        self.output_file.close()

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
