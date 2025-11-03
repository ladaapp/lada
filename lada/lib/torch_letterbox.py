# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn.functional as F

class PyTorchLetterBox:
    def __init__(self, imgsz: int | tuple[int, int], stride: int = 32) -> None:
        if isinstance(imgsz, int):
            self.new_shape: tuple[int, int] = (imgsz, imgsz)
        else:
            self.new_shape = imgsz
        self.stride: int = stride
        self.pad_value: float = 144.0/255.0

    def __call__(self, image: torch.Tensor) -> torch.Tensor: # (B,C,H,W)
        h, w = image.shape[-2:]
        new_h, new_w = self.new_shape

        r = min(new_h / h, new_w / w)
        new_unpad_w = int(round(w * r))
        new_unpad_h = int(round(h * r))

        dw = new_w - new_unpad_w
        dh = new_h - new_unpad_h
        dw = int(dw % self.stride)
        dh = int(dh % self.stride)

        if (h, w) != (new_unpad_h, new_unpad_w):
            image = F.interpolate(
                image,
                size=(new_unpad_h, new_unpad_w),
                mode="bilinear",
                align_corners=False,
            )

        image = F.pad(image, (dw // 2, dw - (dw // 2), dh // 2, dh - (dh // 2)), value=self.pad_value)

        return image