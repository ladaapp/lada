import cv2
import math
import numpy as np

from lada.lib import Box, Mask
from lada.lib import image_utils
import torch
from torchvision.transforms import GaussianBlur

def get_box(mask: Mask) -> Box:
    points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(points)
    t, l, b, r = y, x, y+h, x+w
    box = t, l, b, r
    return box

def morph(mask: Mask, iterations=1) -> Mask:
    if get_mask_area(mask) < 0.01:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=iterations)

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
    target_size = 256
    # Dilations are slow when using huge kernels (which we would need for high-res masks). therefore we downscale mask to perform morph operations on much smaller pixel space with smaller kernels
    extended_mask = clean_up_boundaries(image_utils.resize(morph(image_utils.resize(mask, target_size, interpolation=cv2.INTER_NEAREST), iterations=value), mask.shape[:2], interpolation=cv2.INTER_NEAREST)).reshape(mask.shape)
    assert mask.shape == extended_mask.shape
    return extended_mask

def clean_up_boundaries(mask: Mask, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) -> Mask:
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def fill_holes(mask: Mask) -> Mask:
    edited_mask = np.zeros_like(mask, dtype=mask.dtype)
    contour, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contour:
        cv2.drawContours(edited_mask, [cnt], 0, 255, -1)

    return edited_mask

def get_mask_area(mask: Mask) -> float:
    pixels = cv2.countNonZero(mask)
    return pixels / (mask.shape[0] * mask.shape[1])


# input is H, W
def create_blend_mask(crop_mask: torch.Tensor) -> torch.Tensor:
    crop_mask = torch.squeeze(crop_mask) > 0
    h, w = crop_mask.shape
    border_ratio = 0.05
    h_inner, w_inner = int(h * (1.0-border_ratio)), int(w * (1.-border_ratio))
    h_outer, w_outer = h - h_inner, w - w_inner
    border_size = min(h_outer, w_outer)
    if border_size < 5:
        return torch.ones_like(crop_mask, dtype=torch.float32, device=crop_mask.device)
    blur_size = border_size
    blend_mask = torch.ones((h_inner, w_inner), device=crop_mask.device, dtype=torch.float32)
    
    # PyTorch padding: (left, right, top, bottom)
    pad_left = math.floor(w_outer / 2)
    pad_right = math.ceil(w_outer / 2)
    pad_top = math.floor(h_outer / 2)
    pad_bottom = math.ceil(h_outer / 2)
    blend_mask = torch.nn.functional.pad(blend_mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    blend_mask = torch.maximum(crop_mask.float(), blend_mask)
    
    # Apply Gaussian blur using torchvision
    # GaussianBlur expects (C, H, W) format, so add channel dimension
    blend_mask = blend_mask.unsqueeze(0)  # Add channel dimension
    gaussian_blur = GaussianBlur(kernel_size=blur_size if blur_size % 2 == 1 else blur_size + 1, sigma=blur_size/3)
    blend_mask = gaussian_blur(blend_mask)
    blend_mask = blend_mask.squeeze(0)  # Remove channel dimension
    
    assert blend_mask.shape == crop_mask.shape
    return blend_mask
