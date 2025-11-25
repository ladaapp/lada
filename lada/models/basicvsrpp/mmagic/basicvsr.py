# SPDX-FileCopyrightText: OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND AGPL-3.0
# Code vendored from: https://github.com/open-mmlab/mmagic

import torch

from .base_edit_model import BaseEditModel
from .registry import MODELS
from .data_sample import DataSample


@MODELS.register_module()
class BasicVSR(BaseEditModel):
    """BasicVSR model for video super-resolution.

    Note that this model is used for IconVSR.

    Paper:
        BasicVSR: The Search for Essential Components in Video Super-Resolution
        and Beyond, CVPR, 2021

    Args:
        generator (dict): Config for the generator structure.
        pixel_loss (dict): Config for pixel-wise loss.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
    """

    def __init__(self,
                 generator,
                 pixel_loss,
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

        # fix pre-trained networks
        self.fix_iter = train_cfg.get('fix_iter', 0) if train_cfg else 0
        self.is_weight_fixed = False

        # count training steps
        self.register_buffer('step_counter', torch.zeros(1))

    def forward_train(self, inputs, data_samples=None, **kwargs):
        """Forward training. Returns dict of losses of training.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            dict: Dict of losses.
        """

        # fix SPyNet and EDVR at the beginning
        if self.step_counter < self.fix_iter:
            if not self.is_weight_fixed:
                self.is_weight_fixed = True
                for k, v in self.generator.named_parameters():
                    if 'spynet' in k or 'edvr' in k:
                        v.requires_grad_(False)
        elif self.step_counter == self.fix_iter:
            # train all the parameters
            self.generator.requires_grad_(True)

        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        batch_gt_data = data_samples.gt_img

        loss = self.pixel_loss(feats, batch_gt_data)
        self.step_counter += 1

        return dict(loss=loss)

    def forward_inference(self, inputs, data_samples=None, **kwargs):
        """Forward inference. Returns predictions of validation, testing.

        Args:
            inputs (torch.Tensor): batch input tensor collated by
                :attr:`data_preprocessor`.
            data_samples (List[BaseDataElement], optional):
                data samples collated by :attr:`data_preprocessor`.

        Returns:
            DataSample: predictions.
        """

        feats = self.forward_tensor(inputs, data_samples, **kwargs)
        # feats.shape = [b, t, c, h, w]
        feats = self.data_preprocessor.destruct(feats, data_samples)

        # If the GT is an image (i.e. the center frame), the output sequence is
        # turned to an image.
        gt = data_samples.gt_img[0]
        if gt is not None and gt.data.ndim == 3:
            t = feats.size(1)
            feats = feats[:, t // 2]

        # create a stacked data sample
        predictions = DataSample(
            pred_img=feats.cpu(), metainfo=data_samples.metainfo)

        return predictions
