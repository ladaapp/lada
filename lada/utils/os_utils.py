# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import subprocess
import sys
import av
import io

import torch
from functools import cache

def get_subprocess_startup_info():
    if sys.platform != "win32":
        return None
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startup_info

def has_modern_nvidia_gpu(device_index: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device_index)
    if major < 7:
        # No tensor cores
        return False
    if major > 7:
        return True
    name = torch.cuda.get_device_name(device_index).lower()
    if "gtx 16" in name:
        return False
    return True

def has_modern_intel_gpu(device_index: int = 0) -> bool:
    if not (hasattr(torch, 'xpu') and torch.xpu.is_available()):
        return False
    if device_index >= torch.xpu.device_count():
        return False
    return True

def gpu_has_fp16_acceleration(device: torch.device = None) -> bool:
    if device is None:
        if has_modern_intel_gpu(0):
            return True
        if torch.cuda.is_available():
            return has_modern_nvidia_gpu(0)
        return False
    if device.type == 'xpu':
        idx = device.index if device.index is not None else 0
        return has_modern_intel_gpu(idx)
    if device.type == 'cuda':
        idx = device.index if device.index is not None else 0
        return has_modern_nvidia_gpu(idx)
    return False

def get_default_torch_device() -> str:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda:0"
    if hasattr(torch, 'xpu') and torch.xpu.is_available() and torch.xpu.device_count() > 0:
        return "xpu:0"
    return "cpu"

def has_nvidia_hardware() -> bool:
    return torch.cuda.is_available()

def has_intel_arc_hardware() -> bool:
    return hasattr(torch, 'xpu') and torch.xpu.is_available()

@cache
def is_intel_qsv_encoding_available() -> bool:
    try:
        with av.logging.Capture():
            av.video.codeccontext.VideoCodecContext.create(
                av.codec.Codec('h264_qsv', 'w'),
                hwaccel=av.codec.hwaccel.HWAccel('qsv', allow_software_fallback=False),
                )
            return True
    except Exception:
        return False

@cache
def is_nvidia_cuda_encoding_available() -> bool:
    try:
        with av.logging.Capture():
            av.video.codeccontext.VideoCodecContext.create(
                av.codec.Codec('h264_nvenc', 'w'),
                hwaccel=av.codec.hwaccel.HWAccel('cuda', allow_software_fallback=False),
                )
        return True
    except Exception:
        return False