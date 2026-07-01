# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import InterpolationMode
import torch.nn.functional as F

class PyTorchLetterBox:
    def __init__(self, imgsz: int | tuple[int, int], original_shape: tuple[int, int], stride: int = 32) -> None:
        if isinstance(imgsz, int):
            new_shape: tuple[int, int] = (imgsz, imgsz)
        else:
            new_shape = imgsz

        self.original_shape = original_shape
        h, w = original_shape
        new_h, new_w = new_shape

        r = min(new_h / h, new_w / w)
        new_unpad_w = int(round(w * r))
        new_unpad_h = int(round(h * r))

        dw = new_w - new_unpad_w
        dh = new_h - new_unpad_h
        dw = int(dw % stride)
        dh = int(dh % stride)

        self.resize = None if (h, w) == (new_unpad_h, new_unpad_w) else Resize(size=(new_unpad_h, new_unpad_w), interpolation=InterpolationMode.BILINEAR, antialias=False)
        self.padding = (dw // 2, dw - (dw // 2), dh // 2, dh - (dh // 2))

    def __call__(self, image: torch.Tensor) -> torch.Tensor: # (B,C,H,W)
        if self.resize is not None:
            image = self.resize(image)
        pad_value = 114.0 / 255.0 if torch.is_floating_point(image) else 114
        return F.pad(image, self.padding, value=pad_value)
