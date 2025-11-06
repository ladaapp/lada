# SPDX-FileCopyrightText: DeepMosaics Authors
# SPDX-License-Identifier: GPL-3.0 AND AGPL-3.0
# Code vendored from: https://github.com/HypoX64/DeepMosaics/

import numpy as np
import torch

from lada.deepmosaics.util import data

def restore_video_frames(gpu_id,netG, frames) -> list[np.ndarray[np.uint8]]:
    """
    T is numer of frames processed in a single step (center frame + N previous/next frames that come before/after it):
     T = 2N + 1. The paper authors use N = 2 in their network (T = 5).
     S is the stride that determines which neighboring frames (N) we chose. With 1 we would take the immediate neighboring frames.
     The bigger S the more changes we're expected to see as each frame is further apart.
     The paper authors use 3 in their network.
    """
    N,T,S = 2,5,3
    LEFT_FRAME = (N*S)
    POOL_NUM = LEFT_FRAME*2+1
    INPUT_SIZE = 256
    FRAME_POS = np.linspace(0, (T-1)*S,T,dtype=np.int64)
    img_pool = []
    previous_frame = None
    init_flag = True

    restored_clip_frames = []

    for i in range(len(frames)):
        input_stream = []
        # image read stream
        if i==0 :# init
            for j in range(POOL_NUM):
                img_pool.append(frames[np.clip(i+j-LEFT_FRAME,0,len(frames)-1)])
        else: # load next frame
            img_pool.pop(0)
            img_pool.append(frames[np.clip(i+LEFT_FRAME,0,len(frames)-1)])

        for pos in FRAME_POS:
            input_stream.append(img_pool[pos][:,:,::-1])
        if init_flag:
            init_flag = False
            previous_frame = input_stream[N]
            previous_frame = data.im2tensor(previous_frame,bgr2rgb=True,gpu_id=gpu_id)

        input_stream = np.array(input_stream).reshape(1,T,INPUT_SIZE,INPUT_SIZE,3).transpose((0,4,1,2,3))
        input_stream = data.to_tensor(data.normalize(input_stream),gpu_id=gpu_id)

        with torch.no_grad():
            unmosaic_pred = netG(input_stream,previous_frame)
        img_fake = data.tensor2im(unmosaic_pred,rgb2bgr = True)
        previous_frame = unmosaic_pred
        restored_clip_frames.append(img_fake.copy())
    return restored_clip_frames