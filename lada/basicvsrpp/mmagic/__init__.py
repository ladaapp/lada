# SPDX-FileCopyrightText: OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND AGPL-3.0
# Code vendored from: https://github.com/open-mmlab/mmagic

from mmengine import DefaultScope

SCOPE = 'lada.basicvsrpp.mmagic'

def register_all_modules():
    from .base_edit_model import BaseEditModel
    from .basicvsr_plusplus_net import BasicVSRPlusPlusNet
    from .basicvsr import BasicVSR
    from .concat_visualizer import ConcatImageVisualizer
    from .data_preprocessor import DataPreprocessor
    from .ema import ExponentialMovingAverageHook
    from .gan_loss import GANLoss
    from .iter_time_hook import IterTimerHook
    from .log_processor import LogProcessor
    from .multi_optimizer_constructor import MultiOptimWrapperConstructor
    from .perceptual_loss import PerceptualLoss
    from .pixelwise_loss import CharbonnierLoss
    from .real_basicvsr import RealBasicVSR
    from .unet_disc import UNetDiscriminatorWithSpectralNorm
    from .vis_backend import TensorboardVisBackend
    from .visualization_hook import VisualizationHook
    from .evaluator import Evaluator
    from .psnr import PSNR
    from .ssim import SSIM
    from .multi_loops import MultiValLoop

    never_created = DefaultScope.get_current_instance() is None or not DefaultScope.check_instance_created(SCOPE)
    if never_created:
        DefaultScope.get_instance(SCOPE, scope_name=SCOPE)
        return
