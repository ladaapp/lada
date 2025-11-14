# SPDX-FileCopyrightText: DeepMosaics Authors
# SPDX-License-Identifier: GPL-3.0 AND AGPL-3.0
# Code vendored from: https://github.com/HypoX64/DeepMosaics/

import torch
from lada.deepmosaics.models import model_util
from lada.deepmosaics.models.BVDNet import define_G as video_G

def video(device: torch.device, model_path: str, fp16: bool):
    dtype = torch.float16 if fp16 else torch.float32
    gpu_id = str(device.index) if device.type == 'cuda' else '-1'
    netG = video_G(N=2,n_blocks=4,gpu_id=gpu_id)
    netG.load_state_dict(torch.load(model_path))
    netG = model_util.todevice(netG,gpu_id)
    netG.eval()
    netG.to(dtype)
    netG.dtype = dtype
    return netG

