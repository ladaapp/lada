import math
import os

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid

from lada.lib import Image, Pad


def pad_image(img: torch.Tensor, max_height, max_width, mode='zero'):
    # For HWC format, height and width are at indices 0 and 1
    height, width = img.shape[0:2]
    if height == max_height and width == max_width:
        return img, [0, 0, 0, 0]
    
    pad_h = max_height - height
    pad_w = max_width - width
    pad_h_t = math.ceil(pad_h / 2)
    pad_h_b = math.floor(pad_h / 2)
    pad_w_l = math.ceil(pad_w / 2)
    pad_w_r = math.floor(pad_w / 2)
    
    pad = [pad_h_t, pad_h_b, pad_w_l, pad_w_r]
    
    padded_image = pad_image_by_pad(img, pad, mode)
    # For HWC format, check height and width at indices 0 and 1
    assert padded_image.shape[0:2] == (max_height, max_width)
    return padded_image, pad

def pad_image_by_pad(img: torch.Tensor, pad: list, mode='zero'):
    pad_h_t, pad_h_b, pad_w_l, pad_w_r = pad
    
    if img.ndim == 3:
        # For 3D tensor (H, W, C) - need to permute to (C, H, W) for F.pad
        img_chw = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        img_bchw = img_chw.unsqueeze(0)  # (1, C, H, W)
        if mode == 'zero':
            # F.pad format: (pad_left, pad_right, pad_top, pad_bottom)
            padded_img = F.pad(img_bchw, (pad_w_l, pad_w_r, pad_h_t, pad_h_b), mode='constant', value=0)
        elif mode == 'reflect':
            padded_img = F.pad(img_bchw, (pad_w_l, pad_w_r, pad_h_t, pad_h_b), mode='reflect')
        else:
            raise NotImplementedError()
        # Remove batch dimension and permute back to (H, W, C)
        padded_img = padded_img.squeeze(0).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
    else:
        raise NotImplementedError("pad_image_by_pad currently only supports 3D tensors (H, W, C)")
    
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

def resize(img: torch.Tensor, size: int|tuple[int, int], interpolation='bilinear'):
    """
    Resize a torch.Tensor image using PyTorch's F.interpolate.
    
    Args:
        img (torch.Tensor): Input tensor of shape (H, W, C) or (B, H, W, C)
        size (int|tuple[int, int]): Target size. If int, resize keeping aspect ratio 
                                   with max dimension equal to size. If tuple, exact (H, W) size.
        interpolation (str|int): Interpolation mode. Can be PyTorch mode string 
                               ('bilinear', 'bicubic', 'nearest', 'area') or OpenCV constant
                               (cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_NEAREST)
    
    Returns:
        torch.Tensor: Resized tensor in (H, W, C) or (B, H, W, C) format
    """
    # Map OpenCV interpolation constants to PyTorch interpolation modes
    if isinstance(interpolation, int):
        cv2_to_torch_interp = {
            cv2.INTER_LINEAR: 'bilinear',
            cv2.INTER_CUBIC: 'bicubic', 
            cv2.INTER_NEAREST: 'nearest',
            cv2.INTER_AREA: 'area'
        }
        interpolation = cv2_to_torch_interp.get(interpolation, 'bilinear')
    
    # Handle uint8 input - F.interpolate only works with float tensors
    original_dtype = img.dtype
    convert_back_to_uint8 = original_dtype == torch.uint8
    if convert_back_to_uint8:
        img = img.float()
    
    # Handle batch dimension and format conversion
    original_shape = img.shape
    if len(original_shape) == 3:  # (H, W, C)
        h, w, c = img.shape
        # Convert to (C, H, W) then add batch dimension: (1, C, H, W)
        img = img.permute(2, 0, 1).unsqueeze(0)
        squeeze_output = True
    else:  # (B, H, W, C)
        b, h, w, c = img.shape
        # Convert to (B, C, H, W)
        img = img.permute(0, 3, 1, 2)
        squeeze_output = False
    
    if type(size) == int:
        # Keep aspect ratio, resize so max dimension equals size
        if max(w, h) == size:
            # Convert back to original format
            if squeeze_output:
                return img.squeeze(0).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            else:
                return img.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
            
        if w >= h:
            scale_factor = size / w
            new_w = size
            new_h = math.ceil(h * scale_factor) if scale_factor < 1.0 else math.floor(h * scale_factor)
        else:
            scale_factor = size / h
            new_h = size
            new_w = math.ceil(w * scale_factor) if scale_factor < 1.0 else math.floor(w * scale_factor)
        new_size = (new_h, new_w)
    else:
        # Exact size
        new_h, new_w = size
        if (h, w) == (new_h, new_w):
            # Convert back to original format
            if squeeze_output:
                return img.squeeze(0).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
            else:
                return img.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        new_size = (new_h, new_w)
    
    # Use F.interpolate for resizing
    resized_img = F.interpolate(img, size=new_size, mode=interpolation, align_corners=False if interpolation in ['bilinear', 'bicubic'] else None)
    
    # Convert back to original dtype if needed
    if convert_back_to_uint8:
        resized_img = resized_img.round().clamp(0, 255).to(torch.uint8)
    
    # Convert back to original format (H, W, C) or (B, H, W, C)
    if squeeze_output:
        resized_img = resized_img.squeeze(0).permute(1, 2, 0)  # (1, C, H, W) -> (H, W, C)
    else:
        resized_img = resized_img.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    
    # Verify output size
    if type(size) == int:
        assert size == max(resized_img.shape[-3:-1]), f"Expected max dimension {size}, got {max(resized_img.shape[-3:-1])}"
    else:
        assert resized_img.shape[-3:-1] == torch.Size(size), f"Expected size {size}, got {resized_img.shape[-3:-1]}"
    
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
    SUPPORTED_IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", "png", ".bmp"}

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
