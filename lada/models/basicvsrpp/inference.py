# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint

from lada.utils import Image
from lada.utils import image_utils
from lada.models.basicvsrpp import register_all_modules
from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan
from lada.models.basicvsrpp.mmagic.basicvsr import BasicVSR
from lada.models.basicvsrpp.mmagic.registry import MODELS

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


def load_model(config: str | dict | None, checkpoint_path, device, fp16=False) -> BasicVSRPlusPlusGan | BasicVSR:
    register_all_modules()
    if device and type(device) == str:
        device = torch.device(device)
    if config is None:
        config = get_default_gan_inference_config()
    elif type(config) == str:
        config = Config.fromfile(config).model
    elif type(config) == dict:
        pass
    else:
        raise Exception("unsupported value for 'config', Must be either a file path to a config file or a dict definition of the model")
    model = MODELS.build(config)
    assert isinstance(model, BasicVSRPlusPlusGan) or isinstance(model, BasicVSR), "Unknown model config. Must be either stage1 (BasicVSR) or stage2 (BasicVSRPlusPlusGan)"
    load_checkpoint(model, checkpoint_path, map_location='cpu', logger=logger)
    model.cfg = config
    model = model.to(device).eval()
    if fp16:
        model = model.half()
    return model

def inference(model: BasicVSRPlusPlusGan | BasicVSR, video: list[Image], device) -> list[Image]:
    input_frame_count = len(video)
    input_frame_shape = video[0].shape
    if device and type(device) == str:
        device = torch.device(device)
    with torch.no_grad():
        input = torch.stack(image_utils.img2tensor(video, bgr2rgb=False, float32=True), dim=0)
        input = torch.unsqueeze(input, dim=0)  # TCHW -> BTCHW
        result = model(inputs=input.to(device))
        result = torch.squeeze(result, dim=0)  # BTCHW -> TCHW
        result = list(torch.unbind(result, 0))
        output = image_utils.tensor2img(result, rgb2bgr=False, out_type=np.uint8, min_max=(0, 1))
        output_frame_count = len(output)
        output_frame_shape = output[0].shape
        assert input_frame_count == output_frame_count and input_frame_shape == output_frame_shape
        return output
