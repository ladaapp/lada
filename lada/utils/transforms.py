# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import io
import math
import random

import av
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from lada.utils import video_utils, Image, image_utils, mosaic_utils, mask_utils, Mask
from lada.utils.degradations import circular_lowpass_kernel, random_mixed_kernels, random_add_gaussian_noise_pt, \
    random_add_poisson_noise_pt
from lada.utils.image_utils import filter2D, UnsharpMaskingSharpener
from lada.utils.jpeg_utils import DiffJPEG


class Blur(torch.nn.Module):
    def __init__(self, kernel_range: list[int], kernel_list: list[str], kernel_prob: list[float], sinc_prob: float, blur_sigma, betag_range, betap_range, device, p:float):
        super().__init__()
        use_sync_filter = np.random.uniform() < sinc_prob
        kernel_size = random.choice(kernel_range)
        self.kernel = self._generate_kernel(kernel_size, use_sync_filter, kernel_list, kernel_prob, blur_sigma, betag_range, betap_range).to(device)
        self.should_apply = np.random.uniform() < p

    def _generate_kernel(self, kernel_size: int, use_sync_filter: bool, kernel_list: list[str], kernel_prob: list[float], blur_sigma, betag_range, betap_range):
        if use_sync_filter:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                kernel_list,
                kernel_prob,
                kernel_size,
                blur_sigma,
                blur_sigma,
                [-math.pi, math.pi],
                betag_range,
                betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
        return torch.FloatTensor(kernel)

    def forward(self, img):
        if not self.should_apply:
            return img
        return filter2D(img, self.kernel)


class SincFilter(torch.nn.Module):
    def __init__(self, kernel_range: list[int], sinc_prob: float, device, p: float):
        super().__init__()
        use_sync_filter = np.random.uniform() < sinc_prob
        kernel_size = random.choice(kernel_range)
        self.kernel = self._generate_kernel(kernel_size, use_sync_filter).to(device)
        self.should_apply = np.random.uniform() < p

    def _generate_kernel(self, kernel_size: int, use_sync_filter: bool):
        if use_sync_filter:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            # TODO: kernel range is now hard-coded, should be in the configure file
            pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
            pulse_tensor[10, 10] = 1
            sinc_kernel = pulse_tensor

        return sinc_kernel

    def forward(self, img):
        if not self.should_apply:
            return img
        return filter2D(img, self.kernel)

class Resize(torch.nn.Module):
    def __init__(self, resize_range: list[float], resize_prob: list[float], target_base_h: float|int, target_base_w: float|int, p: float):
        super().__init__()
        resize_operation = random.choices(['up', 'down', 'keep'], resize_prob)[0]
        if resize_operation == 'up':
            scale_factor = np.random.uniform(1, resize_range[1])
        elif resize_operation == 'down':
            scale_factor = np.random.uniform(resize_range[0], 1)
        else:
            scale_factor = 1
        self.size = (int(target_base_h * scale_factor), int(target_base_w * scale_factor))
        self.interpolation_mode = random.choice(['area', 'bilinear', 'bicubic'])
        self.should_apply = np.random.uniform() < p

    def forward(self, img):
        if not self.should_apply:
            return img
        img = F.interpolate(img, size=self.size, mode=self.interpolation_mode)

        return img

class Sharpen(torch.nn.Module):
    def __init__(self, sharpener: UnsharpMaskingSharpener, p: float):
        super().__init__()
        self.sharpener = sharpener
        self.should_apply = np.random.uniform() < p

    def forward(self, img):
        return self.sharpener(img) if self.should_apply else img

class GaussianPoissonNoise(torch.nn.Module):
    def __init__(self, sigma_range: list[float], poisson_scale_range: list[float], gaussian_noise_prob: float, gray_noise_prob: float, p:float):
        super().__init__()
        self.use_gaussian_noise = np.random.uniform() < gaussian_noise_prob
        self.sigma_range = sigma_range
        self.poisson_scale_range = poisson_scale_range
        self.gray_noise_prob = gray_noise_prob
        self.should_apply = np.random.uniform() < p

    def forward(self, img):
        if not self.should_apply:
            return img
        if self.use_gaussian_noise:
            img = random_add_gaussian_noise_pt(img, sigma_range=self.sigma_range, clip=True, rounds=False,
                                               gray_prob=self.gray_noise_prob)
        else:
            img = random_add_poisson_noise_pt(
                img,
                scale_range=self.poisson_scale_range,
                gray_prob=self.gray_noise_prob,
                clip=True,
                rounds=False)

        return img

class GaussianNoise(torch.nn.Module):
    def __init__(self, snr: int, p: float):
        super().__init__()
        self.snr = snr
        self.mean = 0
        self.should_apply = np.random.random() < p

    def apply_noise(self, img):
        img = img / 255.0
        noise = np.random.normal(self.mean, 10 ** (-self.snr / 20), img.shape)
        img = np.clip(img + noise, 0, 1)
        img = (img * 255).astype(np.uint8)
        return img

    def forward(self, img):
        if not self.should_apply:
            return img
        imgs_in = img if isinstance(img, list) else [img]
        imgs_out = [self.apply_noise(_img) for _img in imgs_in]
        return imgs_out if isinstance(img, list) else imgs_out[0]

class GaussianBlur(torch.nn.Module):
    def __init__(self, sigma_range: list[float], p: float):
        super().__init__()
        self.sigma = np.random.randint(sigma_range[0], sigma_range[1])
        self.should_apply = np.random.random() < p

    def forward(self, img):
        if not self.should_apply:
            return img
        imgs_in = img if isinstance(img, list) else [img]
        imgs_out = [cv2.GaussianBlur(_img, (13,13), self.sigma) for _img in imgs_in]
        return imgs_out if isinstance(img, list) else imgs_out[0]

class ResizeFrames(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, imgs):
        frames = imgs if isinstance(imgs, list) else [imgs]
        resized_frames = video_utils.resize_video_frames(frames, self.size)
        return resized_frames if isinstance(imgs, list) else resized_frames[0]

class JPEGCompression(torch.nn.Module):
    def __init__(self, jpeger: DiffJPEG, jpeg_range: list[int], p: float):
        super().__init__()
        self.jpeger = jpeger
        self.jpeg_range = jpeg_range
        self.should_apply = np.random.random() < p

    def forward(self, img):
        if not self.should_apply:
            return img
        jpeg_p = img.new_zeros(img.size(0)).uniform_(*self.jpeg_range)
        img = torch.clamp(img, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        img = self.jpeger(img, quality=jpeg_p)

        return img

class VideoCompression(torch.nn.Module):
    def __init__(self, p:float, codecs: list[str], codec_probs: list[float], crf_ranges: dict, bitrate_ranges: dict):
        super().__init__()
        self.should_apply = np.random.random() < p
        codec = str(np.random.choice(codecs, p=codec_probs))
        self.crf = np.random.randint(crf_ranges[codec][0], crf_ranges[codec][1] + 1) if codec in crf_ranges else None
        self.bitrate = np.random.randint(bitrate_ranges[codec][0], bitrate_ranges[codec][1] + 1) if codec in bitrate_ranges else None
        self.codec = codec

    def forward(self, imgs: list[Image] | Image):
        if not self.should_apply:
            return imgs
        multiplier = 3
        frames = imgs if isinstance(imgs, list) else [imgs]*multiplier
        h, w = frames[0].shape[:2]
        frames = video_utils.pad_to_compatible_size_for_video_codecs(frames)
        try:
            degraded_frames = self._apply_video_compression(frames, self.codec, self.bitrate, self.crf)
        except Exception as e:
            print("ERROR while applying video compression, ignoring...", e)
            degraded_frames = frames
        unpadded_frames = [img[0:h, 0:w, :] for img in degraded_frames]
        return unpadded_frames if isinstance(imgs, list) else unpadded_frames[np.random.randint(0, multiplier)]

    def _apply_video_compression(self, imgs: list[Image], codec, bitrate, crf=None) -> list[Image]:
        buf = io.BytesIO()
        with av.open(buf, 'w', 'mp4') as container:
            options = {}
            if crf: options["crf"] = str(crf)
            if codec == 'libx265': options['x265-params'] = 'log_level=error'
            if codec == 'libx264' or codec == 'libx265': options['preset'] = 'veryfast'
            stream = container.add_stream(codec, rate=1, options=options)
            stream.height = imgs[0].shape[0]
            stream.width = imgs[0].shape[1]
            stream.pix_fmt = 'yuv420p'
            if bitrate: stream.bit_rate = bitrate

            for img in imgs:
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                frame.pict_type = av.video.frame.PictureType.NONE
                for packet in stream.encode(frame):
                    container.mux(packet)

            # Flush stream
            for packet in stream.encode():
                container.mux(packet)

        outputs = []
        with av.open(buf, 'r', 'mp4') as container:
            if container.streams.video:
                for frame in container.decode(**{'video': 0}):
                    img = frame.to_rgb().to_ndarray().astype(np.uint8)
                    outputs.append(img)

        return outputs

class Tensor2Image(torch.nn.Module):
    def __init__(self, rgb2bgr, squeeze):
        super().__init__()
        self.rgb2bgr = rgb2bgr
        self.squeeze = squeeze
    def forward(self, tensor):
        img =  image_utils.tensor2img([tensor], rgb2bgr=self.rgb2bgr)[0]
        return img.squeeze() if self.squeeze else tensor

class Image2Tensor(torch.nn.Module):
    def __init__(self, bgr2rgb, unsqueeze, device):
        super().__init__()
        self.device= device
        self.bgr2rgb = bgr2rgb
        self.unsqueeze = unsqueeze
    def forward(self, img):
        tensor = image_utils.img2tensor(img, bgr2rgb=self.bgr2rgb).to(self.device)
        return tensor.unsqueeze(0) if self.unsqueeze else tensor

class Mosaic(torch.nn.Module):
    def __init__(self,
                 mask_area_calc_methods=['normal', 'bounding'],
                 mask_area_calc_method_props=[0.5, 0.5],
                 mask_dilation_iteration_range: list[int]=[0,2],
                 base_block_size_scale_factor_range: list[float]=[0.7, 2.2],
                 block_shapes=['squa_mid', 'squa_avg', 'rect_avg'],
                 block_shape_probs=[0.25, 0.3, 0.45],
                 rectangular_block_ratio_range=[1.1, 1.8],
                 feather_prob=0.7,
                 feather_size_range=[0., 2.5],
                 reuse_input_mask_value=False
                 ):
        super().__init__()

        self.mask_area_calc_method = np.random.choice(mask_area_calc_methods, p=np.array(mask_area_calc_method_props).ravel())
        self.mask_dilation_iterations = np.random.randint(mask_dilation_iteration_range[0], mask_dilation_iteration_range[1])
        self.mosaic_mod = np.random.choice(block_shapes, p=np.array(block_shape_probs).ravel())
        self.mosaic_rectangle_ratio = random.uniform(rectangular_block_ratio_range[0], rectangular_block_ratio_range[1])
        self.mosaic_block_size_scale_factor = np.random.uniform(base_block_size_scale_factor_range[0], base_block_size_scale_factor_range[1])
        self.feather_size_scale_factor = random.uniform(feather_size_range[0], feather_size_range[1])
        self.should_apply_feathering =  random.random() < feather_prob
        self.reuse_input_mask_value = reuse_input_mask_value

    def _crop_to_box(self, img, box):
        t, l, b, r = box
        cropped_img = img[t:b + 1, l:r + 1]
        return cropped_img

    def _paste_img(self, outer_image, cropped_image, box):
        img = outer_image.copy()
        t, l, b, r = box
        img[t:b + 1, l:r + 1, :] = cropped_image
        return img

    def _paste_mask(self, outer_mask, cropped_mask, box):
        mask = np.zeros_like(outer_mask, dtype=outer_mask.dtype)
        t, l, b, r = box
        mask[t:b + 1, l:r + 1, :] = cropped_mask
        return mask

    def forward(self, img: Image, mask: Mask):
        single_image = not isinstance(img, list)
        imgs_gt = [img] if single_image else img
        masks_gt = [mask] if single_image else mask

        base_block_size = mosaic_utils.get_mosaic_block_size_v4(masks_gt[0], area_type=self.mask_area_calc_method)
        mosaic_size = int(base_block_size * self.mosaic_block_size_scale_factor)
        mosaic_feather_size = int(mosaic_size * self.feather_size_scale_factor) if self.should_apply_feathering else -1

        img_lqs = []
        mask_lqs = []
        for img_gt, mask_gt in zip(imgs_gt, masks_gt):
            mask_gt = mask_utils.dilate_mask(mask_gt, iterations=self.mask_dilation_iterations)
            img_lq, mask_lq = mosaic_utils.addmosaic_base(img_gt,
                                                          mask_gt,
                                                          mosaic_size,
                                                          model=self.mosaic_mod,
                                                          rect_ratio=self.mosaic_rectangle_ratio,
                                                          feather=mosaic_feather_size,
                                                          reuse_input_mask_value=self.reuse_input_mask_value)
            img_lqs.append(img_lq)
            mask_lqs.append(mask_lq)

        return (img_lqs[0], mask_lqs[0], mosaic_size) if single_image else (img_lqs, mask_lqs, mosaic_size)