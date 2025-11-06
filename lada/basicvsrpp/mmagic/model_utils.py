# SPDX-FileCopyrightText: OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND AGPL-3.0
# Code vendored from: https://github.com/open-mmlab/mmagic

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        normal_init, update_init_info,
                                        xavier_init)
from mmengine.registry import Registry
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm


def default_init_weights(module, scale=1):
    """Initialize network weights.

    Args:
        modules (nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


def get_module_device(module):
    """Get the device of a module.

    Args:
        module (nn.Module): A module contains the parameters.

    Returns:
        torch.device: The device of the module.
    """
    try:
        next(module.parameters())
    except StopIteration:
        raise ValueError('The input module should contain parameters.')

    if next(module.parameters()).is_cuda:
        return next(module.parameters()).get_device()
    else:
        return torch.device('cpu')


def set_requires_grad(nets, requires_grad=False):
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def build_module(module: Union[dict, nn.Module], builder: Registry, *args,
                 **kwargs) -> Any:
    """Build module from config or return the module itself.

    Args:
        module (Union[dict, nn.Module]): The module to build.
        builder (Registry): The registry to build module.
        *args, **kwargs: Arguments passed to build function.

    Returns:
        Any: The built module.
    """
    if isinstance(module, dict):
        return builder.build(module, *args, **kwargs)
    elif isinstance(module, nn.Module):
        return module
    else:
        raise TypeError(
            f'Only support dict and nn.Module, but got {type(module)}.')
