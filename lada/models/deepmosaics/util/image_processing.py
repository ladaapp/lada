# SPDX-FileCopyrightText: DeepMosaics Authors
# SPDX-License-Identifier: GPL-3.0 AND AGPL-3.0
# Code vendored from: https://github.com/HypoX64/DeepMosaics/

import numpy as np

def psnr(img1,img2):
    mse = np.mean((img1/255.0-img2/255.0)**2)
    if mse < 1e-10:
        return 100
    psnr_v = 20*np.log10(1/np.sqrt(mse))
    return psnr_v

def splice(imgs,splice_shape):
    '''Stitching multiple images, all imgs must have the same size
    imgs : [img1,img2,img3,img4]
    splice_shape: (2,2)
    '''
    h,w,ch = imgs[0].shape
    output = np.zeros((h*splice_shape[0],w*splice_shape[1],ch),np.uint8)
    cnt = 0
    for i in range(splice_shape[0]):
        for j in range(splice_shape[1]):
            if cnt < len(imgs):
                output[h*i:h*(i+1),w*j:w*(j+1)] = imgs[cnt]
                cnt += 1
    return output

