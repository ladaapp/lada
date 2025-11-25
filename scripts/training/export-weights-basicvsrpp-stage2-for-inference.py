# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import os

from mmengine.runner import load_checkpoint
import torch

from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan
from lada.models.basicvsrpp import register_all_modules

register_all_modules()

MODEL_WEIGHTS_IN_PATH = 'experiments/basicvsrpp/mosaic_restoration_generic_stage2.6/iter_100000.pth'
MODEL_WEIGHTS_OUT_PATH = 'experiments/basicvsrpp/mosaic_restoration_generic_stage2.6/lada_mosaic_restoration_model_generic_v1.2.pth'
pretrained_models_dir = 'model_weights'

model = BasicVSRPlusPlusGan(
    generator=dict(
        type='BasicVSRPlusPlusGanNet',
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained=os.path.join(pretrained_models_dir, "3rd_party", "spynet_20210409-c6c1bd09.pth")),
    discriminator=dict(
        type='UNetDiscriminatorWithSpectralNorm',
        in_channels=3,
        mid_channels=64,
        skip_connection=True),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={
            '2': 0.1,
            '7': 0.1,
            '16': 1.0,
            '25': 1.0,
            '34': 1.0,
        },
        vgg_type='vgg19',
        perceptual_weight=0.2, # was 1.0
        pretrained=os.path.join(pretrained_models_dir, "3rd_party", "vgg19-dcbb9e9d.pth"),
        style_weight=0,
        norm_img=False),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=5e-2,
        real_label_val=1.0,
        fake_label_val=0),
    is_use_ema=True,
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

load_checkpoint(model, MODEL_WEIGHTS_IN_PATH, strict=True)

model.discriminator = None
model.perceptual_loss = None
model.gan_loss = None


torch.save(model.state_dict(), MODEL_WEIGHTS_OUT_PATH)
