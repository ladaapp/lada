# SPDX-FileCopyrightText: DeepMosaics Authors
# SPDX-License-Identifier: GPL-3.0 AND AGPL-3.0
# Code vendored from: https://github.com/HypoX64/DeepMosaics/

import numpy as np
import torch
import cv2

def to_tensor(data,gpu_id):
    data = torch.from_numpy(data)
    if gpu_id != '-1':
        data = data.cuda()
    return data

def normalize(data):
    '''
    normalize to -1 ~ 1
    '''
    return (data.astype(np.float32)/255.0-0.5)/0.5

def anti_normalize(data):
    return np.clip((data*0.5+0.5)*255,0,255).astype(np.uint8)

def tensor2im(image_tensor, gray=False, rgb2bgr = True ,is0_1 = False, batch_index=0):
    image_tensor =image_tensor.data
    image_numpy = image_tensor[batch_index].cpu().float().numpy()
    
    if not is0_1:
        image_numpy = (image_numpy + 1)/2.0
    image_numpy = np.clip(image_numpy * 255.0,0,255) 

    # gray -> output 1ch
    if gray:
        h, w = image_numpy.shape[1:]
        image_numpy = image_numpy.reshape(h,w)
        return image_numpy.astype(np.uint8)

    # output 3ch
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = image_numpy.transpose((1, 2, 0))  
    if rgb2bgr and not gray:
        image_numpy = image_numpy[...,::-1]-np.zeros_like(image_numpy)
    return image_numpy.astype(np.uint8)


def im2tensor(image_numpy, gray=False,bgr2rgb = True, reshape = True, gpu_id = '-1',is0_1 = False):
    if gray:
        h, w = image_numpy.shape
        image_numpy = (image_numpy/255.0-0.5)/0.5
        image_tensor = torch.from_numpy(image_numpy).float()
        if reshape:
            image_tensor = image_tensor.reshape(1,1,h,w)
    else:
        h, w ,ch = image_numpy.shape
        if bgr2rgb:
            image_numpy = image_numpy[...,::-1]-np.zeros_like(image_numpy)
        if is0_1:
            image_numpy = image_numpy/255.0
        else:
            image_numpy = (image_numpy/255.0-0.5)/0.5
        image_numpy = image_numpy.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image_numpy).float()
        if reshape:
            image_tensor = image_tensor.reshape(1,ch,h,w)
    if gpu_id != '-1':
        image_tensor = image_tensor.cuda()
    return image_tensor

def shuffledata(data,target):
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(target)

def showresult(img1,img2,img3,name,is0_1 = False):
    size = img1.shape[3]
    showimg=np.zeros((size,size*3,3))
    showimg[0:size,0:size] = tensor2im(img1,rgb2bgr = False, is0_1 = is0_1)
    showimg[0:size,size:size*2] = tensor2im(img2,rgb2bgr = False, is0_1 = is0_1)
    showimg[0:size,size*2:size*3] = tensor2im(img3,rgb2bgr = False, is0_1 = is0_1)
    cv2.imwrite(name, showimg)
