# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from lada.basicvsrpp.mmagic.registry import MODELS
from lada.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet
from lada.basicvsrpp.mmagic.real_basicvsr import RealBasicVSR

@MODELS.register_module()
class BasicVSRPlusPlusGanNet(BasicVSRPlusPlusNet):
    def __init__(self,
                **kwargs):

        super().__init__(**kwargs)
        self.spynet.requires_grad_(False)


    def forward(self, lqs, return_lqs=False):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            return_lqs (bool): Whether to return LQ sequence. Default: False.

        Returns:
            Tensor: Output HR sequence.
        """
        outputs = super().forward(lqs)

        if return_lqs:
            return outputs, lqs
        else:
            return outputs

@MODELS.register_module()
class BasicVSRPlusPlusGan(RealBasicVSR):
    """RealBasicVSR model for real-world video super-resolution.

    Ref:
    Investigating Tradeoffs in Real-World Video Super-Resolution, arXiv

    Args:
        generator (dict): Config for the generator.
        discriminator (dict, optional): Config for the discriminator.
            Default: None.
        gan_loss (dict, optional): Config for the gan loss.
            Note that the loss weight in gan loss is only for the generator.
        pixel_loss (dict, optional): Config for the pixel loss. Default: None.
        perceptual_loss (dict, optional): Config for the perceptual loss.
            Default: None.
        train_cfg (dict): Config for training. Default: None.
            You may change the training of gan by setting:
            `disc_steps`: how many discriminator updates after one generate
            update;
            `disc_init_steps`: how many discriminator updates at the start of
            the training.
            These two keys are useful when training with WGAN.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`. Default: None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. Default: None.
    """

    def __init__(self,
                 generator,
                 discriminator=None,
                 gan_loss=None,
                 pixel_loss=None,
                 perceptual_loss=None,
                 is_use_ema=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None):

        super().__init__(
            generator=generator,
            discriminator=discriminator,
            gan_loss=gan_loss,
            pixel_loss=pixel_loss,
            perceptual_loss=perceptual_loss,
            is_use_sharpened_gt_in_pixel=False,
            is_use_sharpened_gt_in_percep=False,
            is_use_sharpened_gt_in_gan=False,
            is_use_ema=is_use_ema,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)


    def extract_gt_data(self, data_samples):
        gt = data_samples.gt_img
        gt_pixel, gt_percep, gt_gan = gt.clone(), gt.clone(), gt.clone()
        n, t, c, h, w = gt_pixel.size()
        gt_pixel = gt_pixel.view(-1, c, h, w)
        gt_percep = gt_percep.view(-1, c, h, w)
        gt_gan = gt_gan.view(-1, c, h, w)

        return gt_pixel, gt_percep, gt_gan