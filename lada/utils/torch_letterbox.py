# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
from torchvision.transforms.v2 import Resize, Pad
from torchvision.transforms.v2.functional import InterpolationMode

class PyTorchLetterBox:
    def __init__(self, imgsz: int | tuple[int, int], original_shape: tuple[int, int], stride: int = 32) -> None:
        if isinstance(imgsz, int):
            new_shape: tuple[int, int] = (imgsz, imgsz)
        else:
            new_shape = imgsz

        self.original_shape = original_shape
        pad_value: float = 114.0/255.0
        h, w = original_shape
        new_h, new_w = new_shape

        r = min(new_h / h, new_w / w)
        new_unpad_w = int(round(w * r))
        new_unpad_h = int(round(h * r))

        dw = new_w - new_unpad_w
        dh = new_h - new_unpad_h
        dw = int(dw % stride)
        dh = int(dh % stride)

        resize = None if (h, w) == (new_unpad_h, new_unpad_w) else Resize(size=(new_unpad_h, new_unpad_w), interpolation=InterpolationMode.BILINEAR, antialias=False)
        pad = Pad(padding=(dw // 2, dh // 2, dw - (dw // 2), dh - (dh // 2)), fill=pad_value)
        self.transform = torch.nn.Sequential(resize, pad) if resize is not None else pad

    def __call__(self, image: torch.Tensor) -> torch.Tensor: # (B,C,H,W)
        return self.transform(image)