# SPDX-FileCopyrightText: OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND AGPL-3.0
# Code vendored from: https://github.com/open-mmlab/mmagic

import os
from typing import Optional, Union

import cv2
import numpy as np
import torch
from mmengine.visualization import BaseVisBackend
from mmengine.visualization import \
    TensorboardVisBackend as BaseTensorboardVisBackend
from mmengine.visualization import WandbVisBackend as BaseWandbVisBackend
from mmengine.visualization.vis_backend import force_init_env

from .registry import VISBACKENDS

@VISBACKENDS.register_module()
class TensorboardVisBackend(BaseTensorboardVisBackend):

    @force_init_env
    def add_image(self, name: str, image: np.array, step: int = 0, **kwargs):
        """Record the image to Tensorboard. Additional support upload gif
        files.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Useless parameter. Wandb does not
                need this parameter. Default to 0.
        """

        if image.ndim == 4:
            n_skip = kwargs.get('n_skip', 1)
            fps = kwargs.get('fps', 60)

            frames_list = []
            for frame in image[::n_skip]:
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not (image.shape[0] % n_skip == 0):
                frames_list.append(image[-1])

            frames_np = np.transpose(
                np.stack(frames_list, axis=0), (0, 3, 1, 2))
            frames_tensor = torch.from_numpy(frames_np)[None, ...]
            self._tensorboard.add_video(
                name, frames_tensor, global_step=step, fps=fps)
        else:
            # write normal image
            self._tensorboard.add_image(name, image, step, dataformats='HWC')


@VISBACKENDS.register_module()
class PaviVisBackend(BaseVisBackend):
    """Visualization backend for Pavi."""

    def __init__(self,
                 save_dir: str,
                 exp_name: Optional[str] = None,
                 labels: Optional[str] = None,
                 project: Optional[str] = None,
                 model: Optional[str] = None,
                 description: Optional[str] = None):
        self.save_dir = save_dir

        self._name = exp_name
        self._labels = labels
        self._project = project
        self._model = model
        self._description = description

    def _init_env(self):
        """Init save dir."""
        try:
            import pavi
        except ImportError:
            raise ImportError(
                'To use \'PaviVisBackend\' Pavi must be installed.')
        self._pavi = pavi.SummaryWriter(
            name=self._name,
            labels=self._labels,
            project=self._project,
            model=self._model,
            description=self._description,
            log_dir=self.save_dir)

    @property  # type: ignore
    @force_init_env
    def experiment(self) -> 'VisBackend':
        """Return the experiment object associated with this visualization
        backend."""
        return self._pavi

    @force_init_env
    def add_image(self,
                  name: str,
                  image: np.array,
                  step: int = 0,
                  **kwargs) -> None:
        """Record the image to Pavi.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB. Default to None.
            step (int): Global step value to record. Default to 0.
        """
        assert image.dtype == np.uint8
        drawn_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self._pavi.add_image(name, drawn_image, step)

    @force_init_env
    def add_scalar(self,
                   name: str,
                   value: Union[int, float, torch.Tensor, np.ndarray],
                   step: int = 0,
                   **kwargs) -> None:
        """Record the scalar data to Pavi.

        Args:
            name (str): The scalar identifier.
            value (int, float, torch.Tensor, np.ndarray): Value to save.
            step (int): Global step value to record. Default to 0.
        """
        if isinstance(value, torch.Tensor):
            value = value.item()
        self._pavi.add_scalar(name, value, step)

    @force_init_env
    def add_scalars(self,
                    scalar_dict: dict,
                    step: int = 0,
                    file_path: Optional[str] = None,
                    **kwargs) -> None:
        """Record the scalars to Pavi.

        The scalar dict will be written to the default and
        specified files if ``file_path`` is specified.

        Args:
            scalar_dict (dict): Key-value pair storing the tag and
                corresponding values. The value must be dumped
                into json format.
            step (int): Global step value to record. Default to 0.
            file_path (str, optional): The scalar's data will be
                saved to the ``file_path`` file at the same time
                if the ``file_path`` parameter is specified.
                Default to None.
        """
        assert isinstance(scalar_dict, dict)
        for name, value in scalar_dict.items():
            self.add_scalar(name, value, step)


@VISBACKENDS.register_module()
class WandbVisBackend(BaseWandbVisBackend):
    """Wandb visualization backend for MMagic."""

    def _init_env(self):
        """Setup env for wandb."""
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir, exist_ok=True)  # type: ignore
        if self._init_kwargs is None:
            self._init_kwargs = {'dir': self._save_dir}
        else:
            self._init_kwargs.setdefault('dir', self._save_dir)
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')

        # add timestamp at the end of name
        timestamp = self._save_dir.split('/')[-2]
        orig_name = self._init_kwargs.get('name', None)
        if orig_name:
            self._init_kwargs['name'] = f'{orig_name}_{timestamp}'
        wandb.init(**self._init_kwargs)
        self._wandb = wandb

    @force_init_env
    def add_image(self, name: str, image: np.array, step: int = 0, **kwargs):
        """Record the image to wandb. Additional support upload gif files.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to be saved. The format
                should be RGB.
            step (int): Useless parameter. Wandb does not
                need this parameter. Default to 0.
        """
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')

        if image.ndim == 4:
            n_skip = kwargs.get('n_skip', 1)
            fps = kwargs.get('fps', 60)

            frames_list = []
            for frame in image[::n_skip]:
                frames_list.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not (image.shape[0] % n_skip == 0):
                frames_list.append(image[-1])

            frames_np = np.transpose(
                np.stack(frames_list, axis=0), (0, 3, 1, 2))
            self._wandb.log(
                {name: wandb.Video(frames_np, fps=fps, format='gif')},
                commit=self._commit)
        else:
            # write normal image
            self._wandb.log({name: wandb.Image(image)}, commit=self._commit)
