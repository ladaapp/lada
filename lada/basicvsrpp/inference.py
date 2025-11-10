# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import os.path

import numpy as np
import torch
from lada.basicvsrpp.mmagic.registry import MODELS
from lada.basicvsrpp import register_all_modules
from mmengine.config import Config
from mmengine.runner import load_checkpoint

logger = logging.getLogger(__name__)

def get_default_gan_inference_config() -> dict:
    return dict(
        type='BasicVSRPlusPlusGan',
        generator=dict(
            type='BasicVSRPlusPlusGanNet',
            mid_channels=64,
            num_blocks=15,
            spynet_pretrained=None),
        pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
        is_use_ema=True,
        data_preprocessor=dict(
            type='DataPreprocessor',
            mean=[0., 0., 0.],
            std=[255., 255., 255.],
        ))


def load_model(config: str | dict | None, checkpoint_path, device, fp16, clip_length):
    register_all_modules()
    if device and type(device) == str:
        device = torch.device(device)
    if type(config) == str:
        config = Config.fromfile(config).model
    elif type(config) == dict:
        pass
    else:
        raise Exception("unsupported value for 'config', Must be either a file path to a config file or a dict definition of the model")
    model = MODELS.build(config)
    load_checkpoint(model, checkpoint_path, map_location='cpu', logger=logger)
    model.cfg = config
    model = model.to(device).eval()
    model.device = device
    model.cpu_buffer = torch.empty(1, clip_length, 3, 256, 256, dtype=torch.uint8, device='cpu', pin_memory=True)
    if fp16:
        model.dtype = torch.float16
        model = model.half()
    else:
        model.dtype = torch.float32

    model.inference_buffer = torch.empty(1, clip_length, 3, 256, 256, dtype=model.dtype, device=device, memory_format=torch.channels_last_3d)
    return model


def inference(model, video: list[torch.Tensor], max_frames=-1):
    input_frame_count = len(video)
    input_frame_shape = video[0].shape
    with torch.inference_mode():
        cpu_buffer_view = model.cpu_buffer[0][:input_frame_count]
        inference_view = model.inference_buffer[:, :input_frame_count]

        torch.stack([x.permute(2, 0, 1) for x in video], dim=0, out=cpu_buffer_view)
        inference_view.copy_(cpu_buffer_view, non_blocking=True)
        inference_view.div_(255.0)

        if max_frames > 0:
            for i in range(0, input.shape[1], max_frames):
                output = model(inputs=model.cpu_buffer[:, i:i + max_frames])
                result.append(output)
            result = torch.cat(result, dim=1)
        else:
            result = model(inputs=inference_view)

        # (H, W, C[BGR]) uint8 images to (B, T, C, H, W) float in [0,1]
        result = result.squeeze(0)[:input_frame_count] # -> (T, C, H, W)
        result = result.mul_(255.0).round_().clamp_(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1) # (T, H, W, C)
        result = list(torch.unbind(result, 0)) # (T, H, W, C) to list of (H, W, C)
        output_frame_count = len(result)
        output_frame_shape = result[0].shape
        assert input_frame_count == output_frame_count and input_frame_shape == output_frame_shape

    return result


def test():
    device = "cuda:0"

    clip_length = 255
    model = load_model("configs/basicvsrpp/mosaic_restoration_generic_stage2.py",
                       "experiments/basicvsrpp/mosaic_restoration_generic_stage2/iter_100000.pth", device, clip_length=clip_length)

    frame1 = torch.randint(0, clip_length, (256, 256, 3), dtype=torch.uint8, device=device)
    frame2 = torch.randint(0, clip_length, (256, 256, 3), dtype=torch.uint8, device=device)
    frame3 = torch.randint(0, clip_length, (256, 256, 3), dtype=torch.uint8, device=device)
    frame4 = torch.randint(0, clip_length, (256, 256, 3), dtype=torch.uint8, device=device)
    video = [frame1, frame2, frame3, frame4]
    result = inference(model, video, device)
    print(len(result), result[0].shape)


if __name__ == '__main__':
    test()
