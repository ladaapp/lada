# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import re

import torch

from lada.utils import os_utils


class DeviceParseError(ValueError):
    pass


def _cuda_device_count() -> int:
    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _xpu_device_count() -> int:
    try:
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return 0
        return int(torch.xpu.device_count())
    except Exception:
        return 0


def detect_available_devices() -> list[str]:
    """Return devices for automatic export scheduling.

    Automatic multi-device scheduling is intentionally limited to CUDA. Other
    backends fall back to the existing default device as a single-device export.
    """
    cuda_count = _cuda_device_count()
    if cuda_count > 0:
        return [f"cuda:{i}" for i in range(cuda_count)]

    try:
        return [os_utils.get_default_torch_device()]
    except Exception:
        return ["cpu"]


def get_available_torch_devices() -> list[str]:
    devices = ["cpu"]

    cuda_count = _cuda_device_count()
    devices.extend(f"cuda:{i}" for i in range(cuda_count))

    try:
        if os_utils.has_mps():
            devices.append("mps")
    except Exception:
        pass

    xpu_count = _xpu_device_count()
    devices.extend(f"xpu:{i}" for i in range(xpu_count))

    return devices


def normalize_device(device: str) -> str:
    device = device.strip().lower()
    if device == "cuda":
        return "cuda:0"
    if device == "xpu":
        return "xpu:0"
    return device


def is_torch_device_available(device: str) -> bool:
    device = normalize_device(device)
    if device == "cpu":
        return True
    if device == "mps":
        try:
            return os_utils.has_mps()
        except Exception:
            return False

    match = re.fullmatch(r"(cuda|xpu):(\d+)", device)
    if not match:
        return False

    device_type = match.group(1)
    device_index = int(match.group(2))
    if device_type == "cuda":
        return device_index < _cuda_device_count()
    if device_type == "xpu":
        return device_index < _xpu_device_count()
    return False


def validate_torch_device(device: str) -> str:
    normalized_device = normalize_device(device)
    if not is_torch_device_available(normalized_device):
        available = ", ".join(get_available_torch_devices())
        raise DeviceParseError(
            f"Device '{device}' is not available. Available devices: {available}"
        )
    return normalized_device


def parse_devices_arg(devices_arg: str) -> list[str]:
    value = devices_arg.strip().lower()
    if not value:
        raise DeviceParseError("--devices must not be empty")

    if value == "auto":
        return detect_available_devices()

    devices = [normalize_device(part) for part in value.split(",") if part.strip()]
    if not devices:
        raise DeviceParseError("--devices must not be empty")

    duplicate_devices = sorted({device for device in devices if devices.count(device) > 1})
    if duplicate_devices:
        raise DeviceParseError(f"Duplicate device in --devices: {', '.join(duplicate_devices)}")

    validated_devices = [validate_torch_device(device) for device in devices]
    if len(validated_devices) > 1 and any(not device.startswith("cuda:") for device in validated_devices):
        raise DeviceParseError("Multiple export devices are currently supported for CUDA devices only")

    return validated_devices


def build_worker_device_slots(
    devices: list[str],
    jobs_per_device: int = 1,
    parallel: int | None = None,
    allow_parallel_cpu: bool = False,
) -> list[str]:
    if jobs_per_device < 1:
        raise ValueError("jobs_per_device must be greater than 0")
    if parallel is not None and parallel < 1:
        raise ValueError("parallel must be greater than 0")
    if not devices:
        raise ValueError("At least one device is required")

    slots = []
    for device in devices:
        slots.extend([device] * jobs_per_device)

    if (
        allow_parallel_cpu
        and parallel is not None
        and len(devices) == 1
        and devices[0] == "cpu"
        and parallel > len(slots)
    ):
        slots = ["cpu"] * parallel

    if parallel is not None:
        slots = slots[:parallel]

    return slots


def safe_device_name(device: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", device)
