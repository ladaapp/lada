# SPDX-FileCopyrightText: OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND AGPL-3.0
# Code vendored from: https://github.com/open-mmlab/mmagic

from typing import Dict, List
from copy import deepcopy

import torch
import torch.nn.functional as F
from mmengine.optim import OptimWrapperDict

from .model_utils import set_requires_grad
from .registry import MODELS
from .base_edit_model import BaseEditModel


@MODELS.register_module()
class RealBasicVSR(BaseEditModel):
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
        is_use_sharpened_gt_in_pixel (bool, optional): Whether to use the image
            sharpened by unsharp masking as the GT for pixel loss.
            Default: False.
        is_use_sharpened_gt_in_percep (bool, optional): Whether to use the
            image sharpened by unsharp masking as the GT for perceptual loss.
            Default: False.
        is_use_sharpened_gt_in_gan (bool, optional): Whether to use the
            image sharpened by unsharp masking as the GT for adversarial loss.
            Default: False.
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
                 is_use_sharpened_gt_in_pixel=False,
                 is_use_sharpened_gt_in_percep=False,
                 is_use_sharpened_gt_in_gan=False,
                 is_use_ema=False,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 data_preprocessor=None):

        super().__init__(
            generator=generator,
            pixel_loss=pixel_loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        # discriminator
        self.discriminator = MODELS.build(
            discriminator) if discriminator else None

        # loss
        self.gan_loss = MODELS.build(gan_loss) if gan_loss else None
        self.perceptual_loss = MODELS.build(
            perceptual_loss) if perceptual_loss else None

        self.disc_steps = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_steps', 1)
        self.disc_repeat = 1 if self.train_cfg is None else self.train_cfg.get(
            'disc_repeat', 1)
        self.disc_init_steps = (0 if self.train_cfg is None else
                                self.train_cfg.get('disc_init_steps', 0))

        self.register_buffer('step_counter', torch.tensor(0), False)

        if self.discriminator is None or self.gan_loss is None:
            # No GAN model or loss.
            self.disc_repeat = 0

        self.is_use_sharpened_gt_in_pixel = is_use_sharpened_gt_in_pixel
        self.is_use_sharpened_gt_in_percep = is_use_sharpened_gt_in_percep
        self.is_use_sharpened_gt_in_gan = is_use_sharpened_gt_in_gan
        self.is_use_ema = is_use_ema

        if is_use_ema:
            self.generator_ema = deepcopy(self.generator)
        else:
            self.generator_ema = None

        if train_cfg is not None:  # used for initializing from ema model
            self.start_iter = train_cfg.get('start_iter', -1)
        else:
            self.start_iter = -1

    def extract_gt_data(self, data_samples):
        """extract gt data from data samples.

        Args:
            data_samples (list): List of DataSample.

        Returns:
            Tensor: Extract gt data.
        """

        gt_pixel, gt_percep, gt_gan = self.super_extract_gt_data(data_samples)
        n, t, c, h, w = gt_pixel.size()
        gt_pixel = gt_pixel.view(-1, c, h, w)
        gt_percep = gt_percep.view(-1, c, h, w)
        gt_gan = gt_gan.view(-1, c, h, w)

        return gt_pixel, gt_percep, gt_gan

    def g_step(self, batch_outputs, batch_gt_data):
        """G step of GAN: Calculate losses of generator.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tuple[Tensor]): Batch GT data.

        Returns:
            dict: Dict of losses.
        """

        gt_pixel, gt_percep, gt_gan = batch_gt_data
        fake_g_output, fake_g_lq = batch_outputs
        fake_g_output = fake_g_output.view(gt_pixel.shape)

        losses = self.g_step_super(
            batch_outputs=fake_g_output,
            batch_gt_data=(gt_pixel, gt_percep, gt_gan))

        return losses

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapperDict) -> Dict[str, torch.Tensor]:
        """Train step of GAN-based method.

        Args:
            data (List[dict]): Data sampled from dataloader.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """

        g_optim_wrapper = optim_wrapper['generator']

        data = self.data_preprocessor(data, True)
        batch_inputs = data['inputs']
        data_samples = data['data_samples']
        batch_gt_data = self.extract_gt_data(data_samples)

        log_vars = dict()

        with g_optim_wrapper.optim_context(self):
            batch_outputs = self.forward_train(batch_inputs, data_samples)

        if self.if_run_g():
            set_requires_grad(self.discriminator, False)

            log_vars_d = self.g_step_with_optim(
                batch_outputs=batch_outputs,
                batch_gt_data=batch_gt_data,
                optim_wrapper=optim_wrapper)

            log_vars.update(log_vars_d)

        if self.if_run_d():
            set_requires_grad(self.discriminator, True)

            gt_pixel, gt_percep, gt_gan = batch_gt_data
            fake_g_output, fake_g_lq = batch_outputs
            fake_g_output = fake_g_output.view(gt_pixel.shape)

            for _ in range(self.disc_repeat):
                # detach before function call to resolve PyTorch2.0 compile bug
                log_vars_d = self.d_step_with_optim(
                    batch_outputs=fake_g_output.detach(),
                    batch_gt_data=(gt_pixel, gt_percep, gt_gan),
                    optim_wrapper=optim_wrapper)

            log_vars.update(log_vars_d)

        if 'loss' in log_vars:
            log_vars.pop('loss')

        self.step_counter += 1

        return log_vars

    def forward_train(self, batch_inputs, data_samples=None):
        """Forward Train.

        Run forward of generator with ``return_lqs=True``

        Args:
            batch_inputs (Tensor): Batch inputs.
            data_samples (List[DataSample]): Data samples of Editing.
                Default:None

        Returns:
            Tuple[Tensor]: Result of generator.
                (outputs, lqs)
        """

        return self.generator(batch_inputs, return_lqs=True)



##################################
##################################
##################################
##################################

    def forward_tensor(self, inputs, data_samples=None, training=False):
        """Forward tensor. Returns result of simple forward.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.
            training (bool): Whether is training. Default: False.

        Returns:
            Tensor: result of simple forward.
        """

        if training or not self.is_use_ema:
            feats = self.generator(inputs)
        else:
            feats = self.generator_ema(inputs)

        return feats

    def g_step_super(self, batch_outputs, batch_gt_data):
        """G step of GAN: Calculate losses of generator.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tuple[Tensor]): Batch GT data.

        Returns:
            dict: Dict of losses.
        """

        gt_pixel, gt_percep, _ = batch_gt_data
        losses = dict()

        # pix loss
        if self.pixel_loss:
            losses['loss_pix'] = self.pixel_loss(batch_outputs, gt_pixel)

        # perceptual loss
        if self.perceptual_loss:
            loss_percep, loss_style = self.perceptual_loss(
                batch_outputs, gt_percep)
            if loss_percep is not None:
                losses['loss_perceptual'] = loss_percep
            if loss_style is not None:
                losses['loss_style'] = loss_style

        # gan loss for generator
        if self.gan_loss and self.discriminator:
            fake_g_pred = self.discriminator(batch_outputs)
            losses['loss_gan'] = self.gan_loss(
                fake_g_pred, target_is_real=True, is_disc=False)

        return losses

    def d_step_real(self, batch_outputs, batch_gt_data: torch.Tensor):
        """Real part of D step.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tuple[Tensor]): Batch GT data.

        Returns:
            Tensor: Real part of gan_loss for discriminator.
        """

        _, _, gt_gan = batch_gt_data

        # real
        real_d_pred = self.discriminator(gt_gan)
        loss_d_real = self.gan_loss(
            real_d_pred, target_is_real=True, is_disc=True)

        return loss_d_real

    def d_step_fake(self, batch_outputs, batch_gt_data):
        """Fake part of D step.

        Args:
            batch_outputs (Tensor): Output of generator.
            batch_gt_data (Tuple[Tensor]): Batch GT data.

        Returns:
            Tensor: Fake part of gan_loss for discriminator.
        """

        # fake
        fake_d_pred = self.discriminator(batch_outputs.detach())
        loss_d_fake = self.gan_loss(
            fake_d_pred, target_is_real=False, is_disc=True)

        return loss_d_fake

    def super_extract_gt_data(self, data_samples):
        """extract gt data from data samples.

        Args:
            data_samples (list): List of DataSample.

        Returns:
            Tensor: Extract gt data.
        """

        gt = data_samples.gt_img
        gt_unsharp = data_samples.gt_unsharp / 255.

        gt_pixel, gt_percep, gt_gan = gt.clone(), gt.clone(), gt.clone()
        if self.is_use_sharpened_gt_in_pixel:
            gt_pixel = gt_unsharp.clone()
        if self.is_use_sharpened_gt_in_percep:
            gt_percep = gt_unsharp.clone()
        if self.is_use_sharpened_gt_in_gan:
            gt_gan = gt_unsharp.clone()

        return gt_pixel, gt_percep, gt_gan


##################################
##################################

    def if_run_g(self):
        """Calculates whether need to run the generator step."""

        return (self.step_counter % self.disc_steps == 0
                and self.step_counter >= self.disc_init_steps)

    def if_run_d(self):
        """Calculates whether need to run the discriminator step."""

        return self.discriminator and self.gan_loss

    def g_step_with_optim(self, batch_outputs: torch.Tensor,
                          batch_gt_data: torch.Tensor,
                          optim_wrapper: OptimWrapperDict):
        """G step with optim of GAN: Calculate losses of generator and run
        optim.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.
            optim_wrapper (OptimWrapperDict): Optim wrapper dict.

        Returns:
            dict: Dict of parsed losses.
        """

        g_optim_wrapper = optim_wrapper['generator']

        with g_optim_wrapper.optim_context(self):
            losses_g = self.g_step(batch_outputs, batch_gt_data)

        parsed_losses_g, log_vars_g = self.parse_losses(losses_g)
        g_optim_wrapper.update_params(parsed_losses_g)

        return log_vars_g

    def d_step_with_optim(self, batch_outputs: torch.Tensor,
                          batch_gt_data: torch.Tensor,
                          optim_wrapper: OptimWrapperDict):
        """D step with optim of GAN: Calculate losses of discriminator and run
        optim.

        Args:
            batch_outputs (Tensor): Batch output of generator.
            batch_gt_data (Tensor): Batch GT data.
            optim_wrapper (OptimWrapperDict): Optim wrapper dict.

        Returns:
            dict: Dict of parsed losses.
        """

        log_vars = dict()
        d_optim_wrapper = optim_wrapper['discriminator']

        with d_optim_wrapper.optim_context(self):
            loss_d_real = self.d_step_real(batch_outputs, batch_gt_data)

        parsed_losses_dr, log_vars_dr = self.parse_losses(
            dict(loss_d_real=loss_d_real))
        log_vars.update(log_vars_dr)
        loss_dr = d_optim_wrapper.scale_loss(parsed_losses_dr)
        d_optim_wrapper.backward(loss_dr)

        with d_optim_wrapper.optim_context(self):
            loss_d_fake = self.d_step_fake(batch_outputs, batch_gt_data)

        parsed_losses_df, log_vars_df = self.parse_losses(
            dict(loss_d_fake=loss_d_fake))
        log_vars.update(log_vars_df)
        loss_df = d_optim_wrapper.scale_loss(parsed_losses_df)
        d_optim_wrapper.backward(loss_df)

        if d_optim_wrapper.should_update():
            d_optim_wrapper.step()
            d_optim_wrapper.zero_grad()

        return log_vars

