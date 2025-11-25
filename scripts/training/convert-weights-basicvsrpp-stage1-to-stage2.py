# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from mmengine.runner import load_checkpoint
import torch

from lada.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGan
from lada.basicvsrpp.mmagic.basicvsr import BasicVSR
from lada.basicvsrpp import register_all_modules

register_all_modules()

BASICVSRPP_WEIGHTS_PATH = 'experiments/basicvsrpp/mosaic_restoration_generic_stage1/iter_10000.pth'
BASICVSRPP_GAN_WEIGHTS_PATH = 'experiments/basicvsrpp/mosaic_restoration_generic_stage1/iter_10000_converted.pth'

gan_model = BasicVSRPlusPlusGan(
    generator=dict(
        type='BasicVSRPlusPlusGanNet',
        mid_channels=64,
        num_blocks=15,
        spynet_pretrained='model_weights/3rd_party/spynet_20210409-c6c1bd09.pth'),
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
        pretrained='model_weights/3rd_party/vgg19-dcbb9e9d.pth',
        perceptual_weight=0.2,  # was 1.0
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
    )
)

basicvsr = BasicVSR(dict(
    type='BasicVSRPlusPlusGanNet',
    mid_channels=64,
    num_blocks=15,
    spynet_pretrained='model_weights/3rd_party/spynet_20210409-c6c1bd09.pth'),
    dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'))

load_checkpoint(basicvsr, BASICVSRPP_WEIGHTS_PATH, strict=True)

gan_model.generator = basicvsr.generator
gan_model.generator_ema = basicvsr.generator

torch.save(gan_model.state_dict(), BASICVSRPP_GAN_WEIGHTS_PATH)
