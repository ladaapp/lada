# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur as tv_gaussian_blur
from lada.utils import Box, Mask, box_utils
from lada.utils import image_utils


def get_box(mask: Mask) -> Box:
    points = cv2.findNonZero(mask)
    return box_utils.convert_from_opencv(cv2.boundingRect(points))

def morph(mask: Mask, iterations=1, operator=cv2.MORPH_DILATE) -> Mask:
    if get_mask_area(mask) < 0.01:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, operator, kernel, iterations=iterations)

def dilate_mask(mask: Mask, dilatation_size=11, iterations=2):
    if iterations == 0:
        return mask
    element = np.ones((dilatation_size, dilatation_size), np.uint8)
    mask_img = cv2.dilate(mask, element, iterations=iterations).reshape(mask.shape)
    return mask_img

def extend_mask(mask: Mask, value) -> Mask:
    # value between 0 and 3 -> higher values mean more extension of mask area. 0 does not change mask at all
    if value == 0:
        return mask

    # Dilations are slow when using huge kernels (which we would need for high-res masks). therefore we downscale mask to perform morph operations on much smaller pixel space with smaller kernels
    target_size = 256
    extended_mask = image_utils.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    extended_mask = morph(extended_mask, iterations=value, operator=cv2.MORPH_DILATE)
    extended_mask = image_utils.resize(extended_mask, mask.shape[:2], interpolation=cv2.INTER_NEAREST)
    extended_mask = extended_mask.reshape(mask.shape)
    assert mask.shape == extended_mask.shape
    return extended_mask

def clean_mask(mask: Mask, box: Box) -> tuple[Mask, Box]:
    t, l, b, r = box
    # Masks from YOLO prediction extend detection area in some cases. Let's crop
    mask[:t + 1, :, :] = 0
    mask[b:, :, :] = 0
    mask[:, :l + 1, :] = 0
    mask[:, r:, :] = 0

    # Mask from YOLO prediction can sometimes contain additional disconnected (tiny) segments. Keep only the largest
    edited_mask = np.zeros_like(mask, dtype=mask.dtype)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) != 0
    if len(contours) > 1:
        contours = sorted(contours, key=lambda contour: cv2.contourArea(contour), reverse=True)[0]
    largest_contour = contours[0]
    cv_box = cv2.boundingRect(largest_contour)
    box = box_utils.convert_from_opencv(cv_box)
    cv2.drawContours(edited_mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
    return edited_mask, box

def get_mask_area(mask: Mask) -> float:
    pixels = cv2.countNonZero(mask)
    return pixels / (mask.shape[0] * mask.shape[1])

def smooth_mask(mask: Mask, kernel_size: int) -> Mask:
    return cv2.medianBlur(mask, kernel_size).reshape(mask.shape)

def create_blend_mask(crop_mask):
    mask = crop_mask.squeeze().float()
    h, w = mask.shape
    border_ratio = 0.05
    h_inner, w_inner = int(h * (1.0 - border_ratio)), int(w * (1.0 - border_ratio))
    h_outer, w_outer = h - h_inner, w - w_inner
    border_size = min(h_outer, w_outer)
    if border_size < 5:
        return torch.ones_like(mask)
    blur_size = int(border_size)
    if blur_size % 2 == 0:
        blur_size += 1
    inner = torch.ones((h_inner, w_inner), device=mask.device, dtype=mask.dtype)
    pad_top = h_outer // 2
    pad_bottom = h_outer - pad_top
    pad_left = w_outer // 2
    pad_right = w_outer - pad_left
    blend = F.pad(inner, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    mask4 = (mask > 0)
    blend = torch.maximum(mask4, blend)
    blend = tv_gaussian_blur(blend.unsqueeze(0), [blur_size, blur_size]).squeeze(0)
    assert blend.shape == mask.shape
    return blend

def apply_random_mask_extensions(mask: Mask) -> Mask:
    value = np.random.choice([0, 0, 1, 1, 2])
    return extend_mask(mask, value)
