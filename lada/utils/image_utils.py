# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import math
import os

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid
from torchvision.transforms.v2 import Resize, InterpolationMode
from typing import Sequence

from lada.utils import Image, Pad

# pad image with reflect mode even if pad size is greater than image size
def torch_pad_reflect(image: torch.Tensor, paddings: Sequence[int]) -> torch.Tensor:
    paddings = np.array(paddings, dtype=int)

    assert np.all(np.array(image.shape[-2:]) > 1),  "Image shape should be more than 1 pixel"
    assert np.all(paddings >= 0), "Negative paddings not supported"

    while np.any(paddings):
        image_limits = np.repeat(image.shape[::-1][:len(paddings)//2], 2) - 1
        possible_paddings = np.minimum(paddings, image_limits)
        image = torch.nn.functional.pad(image, tuple(possible_paddings), mode='reflect')
        paddings = paddings - possible_paddings

    return image

def pad_image(img: Image|torch.Tensor, max_height, max_width, mode='zero') -> tuple[Image|torch.Tensor, Pad]:
    height, width = img.shape[:2]
    if height == max_height and width == max_width:
        return img, (0, 0, 0, 0)
    pad_h = max_height - height
    pad_w = max_width - width
    pad_h_t = math.ceil(pad_h / 2)
    pad_h_b = math.floor(pad_h / 2)
    pad_w_l = math.ceil(pad_w / 2)
    pad_w_r = math.floor(pad_w / 2)
    pad = (pad_h_t, pad_h_b,pad_w_l, pad_w_r)
    if isinstance(img, torch.Tensor):
        if mode == 'reflect':
            padded_image = torch_pad_reflect(img.permute(2, 0, 1), (pad_w_l, pad_w_r, pad_h_t, pad_h_b)).permute(1, 2, 0)
        else:
            padded_image = F.pad(img.permute(2, 0, 1), (pad_w_l, pad_w_r, pad_h_t, pad_h_b), mode='constant').permute(1, 2, 0)
    else:
        padded_image =  pad_image_by_pad(img, pad, mode)
    assert padded_image.shape[:2] == (max_height, max_width)
    return padded_image, pad

def pad_image_by_pad(img: Image, pad: Pad, mode='zero'):
    (pad_h_t, pad_h_b,pad_w_l, pad_w_r) = pad
    if img.ndim == 3:
        if mode == 'zero':
            padded_img = np.pad(img, ((pad_h_t, pad_h_b),(pad_w_l, pad_w_r),(0,0)), mode='constant', constant_values=0)
        elif mode == 'reflect':
            padded_img = np.pad(img, ((pad_h_t, pad_h_b),(pad_w_l, pad_w_r),(0,0)), mode='reflect')
        else:
            raise NotImplementedError()
    else:
        assert mode == 'zero'
        padded_img = np.pad(img, ((pad_h_t, pad_h_b),(pad_w_l, pad_w_r)), mode='constant', constant_values=0)
    return padded_img

def repad_image(imgs: list[Image], pads: list[Pad], mode='reflect'):
    assert len(imgs) == len(pads)
    padded_imgs = []
    for img, pad in zip(imgs, pads):
        (pad_h_t, pad_h_b, pad_w_l, pad_w_r) = pad
        h, w = img.shape[:2]
        if img.ndim == 3:
            if mode == 'zero':
                padded_img = np.pad(img[pad_h_t:h-pad_h_b, pad_w_l:w-pad_w_r], ((pad_h_t, pad_h_b),(pad_w_l, pad_w_r),(0,0)), mode='constant', constant_values=0)
            elif mode == 'reflect':
                padded_img = np.pad(img[pad_h_t:h-pad_h_b, pad_w_l:w-pad_w_r], ((pad_h_t, pad_h_b),(pad_w_l, pad_w_r),(0,0)), mode='reflect')
            else:
                raise NotImplementedError()
        else:
            padded_img = np.pad(img[pad_h_t:h-pad_h_b, pad_w_l:w-pad_w_r], ((pad_h_t, pad_h_b),(pad_w_l, pad_w_r)), mode='constant', constant_values=0)
        assert padded_img.shape[0] == h and padded_img.shape[1] == w
        padded_imgs.append(padded_img)
    return padded_imgs

def scale_pad(pad: Pad, scale_h: float, scale_w: float):
    if scale_h == 1 and scale_w == 1:
        return pad
    (pad_h_t, pad_h_b, pad_w_l, pad_w_r) = pad
    scaled_pad = (math.ceil(pad_h_t/scale_h), math.ceil(pad_h_b/scale_h), math.ceil(pad_w_l/scale_w), math.ceil(pad_w_r/scale_w))
    return scaled_pad

def unpad_image(img: Image, pad: Pad):
    (pad_h_t, pad_h_b, pad_w_l, pad_w_r) = pad
    h, w = img.shape[:2]
    unpadded_img = img[pad_h_t:h - pad_h_b, pad_w_l:w - pad_w_r]
    return unpadded_img

def img2tensor(imgs, bgr2rgb=True, float32=True, normalize_neg1_pos1 = False):
    """Numpy array to tensor. HWC to CHW

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
            if normalize_neg1_pos1:
                img = (img/ 255.0 - 0.5) / 0.5
            else:
                img = img / 255.
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor)):
        raise TypeError(f'list of tensors expected, got {type(tensor)}')

    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    return result

def resize(img: Image|torch.Tensor, size: int|tuple[int, int], interpolation=cv2.INTER_LINEAR) -> Image|torch.Tensor:
    if type(size) == int:
        h, w = img.shape[:2]
        if max(w, h) == size:
            return img
        if w >= h:
            scale_factor = size / w
            new_h = size
            new_w = math.ceil(h * scale_factor) if scale_factor < 1.0 else math.floor(h * scale_factor)
        else:
            scale_factor = size / h
            new_w = size
            new_h = math.ceil(w * scale_factor) if scale_factor < 1.0 else math.floor(w * scale_factor)
    else:
        if img.shape[:2] == size:
            return img
        new_h, new_w = size

    if isinstance(img, torch.Tensor):
        if interpolation == cv2.INTER_LINEAR:
            interpolation = InterpolationMode.BILINEAR
        elif interpolation == cv2.INTER_NEAREST:
            interpolation = InterpolationMode.NEAREST
        else:
            raise NotImplementedError(f"Interpolation {interpolation} not supported")

        img = img.permute(2, 0, 1)
        resized_img = Resize(size=(new_h, new_w), interpolation=interpolation, antialias=False)(img)
        resized_img = resized_img.permute(1, 2, 0)
    else:
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    assert size == max(resized_img.shape[:2]) if type(size) == int else size == resized_img.shape[:2]
    return resized_img

def resize_simple(img: Image, size: int, interpolation=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    if np.min((w,h)) == size:
        return img
    if w >= h:
        res = cv2.resize(img,(int(size*w/h), size),interpolation=interpolation)
    else:
        res = cv2.resize(img,(size, int(size*h/w)),interpolation=interpolation)
    return res

def is_image_file(file_path):
    SUPPORTED_IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    file_ext = os.path.splitext(file_path)[1]
    return file_ext in SUPPORTED_IMAGE_FILE_EXTENSIONS


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D
    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        # raise ValueError('Wrong kernel size')
        img = F.pad(img, (k // 2, k // 2 - 1, k // 2, k // 2 - 1), mode='reflect')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


class UnsharpMaskingSharpener(torch.nn.Module):
    def __init__(self, radius=50, sigma=0):
        super(UnsharpMaskingSharpener, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img


def rotate(img: Image, deg):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2),deg,1)
    img = cv2.warpAffine(img,M,(w,h))
    return img