# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import subprocess
import sys

import torch

def get_subprocess_startup_info():
    if sys.platform != "win32":
        return None
    startup_info = subprocess.STARTUPINFO()
    startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startup_info

def gpu_has_tensor_cores(device_index: int = 0) -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(device_index)
    if major < 7:
        return False
    if major > 7:
        return True
    name = torch.cuda.get_device_name(device_index).lower()
    if "gtx 16" in name:
        return False
    return True