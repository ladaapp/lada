# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import math

from lada.lib import Box, Mask, Image


def crop_to_box_v3(box: Box, img: Image, mask_img: Mask, target_size: tuple[int, int], max_box_expansion_factor=1.0, border_size=0):
    """
    Crops Mask and Image by using Box. Will try to grow Box to better fit target size
    Parameters
    ----------
    box
    img
    mask_img
    target_size
    max_box_expansion_factor: Limits how much to grow the Box before cropping. Could be useful for tiny Boxes (compared to given target size)
    border_size: includes area outside of box. useful to additional context outside the box detection

    Returns
    -------
    img, mask_img, cropped_box, scale_factor
    """
    target_width, target_height = target_size
    t, l, b, r = box
    width, height = r - l + 1,  b - t + 1
    border_size = max(20, int(max(width, height) * border_size)) if border_size > 0. else 0
    t, l, b, r = max(0, t-border_size), max(0, l-border_size), min(img.shape[0]-1, b+border_size), min(img.shape[1]-1, r+border_size)
    width, height = r - l + 1,  b - t + 1
    down_scale_factor = min(target_width / width, target_height / height)
    if down_scale_factor > 1.0:
        # we ignore upscaling for now as we first want to try expanding the box.
        down_scale_factor = 1.0
    missing_width, missing_height = int((target_width - (width * down_scale_factor)) / down_scale_factor), int((target_height - (height * down_scale_factor)) / down_scale_factor)

    available_width_l = l
    available_width_r = (img.shape[1]-1) - r
    available_height_t = t
    available_height_b = (img.shape[0]-1) - b

    budget_width = int(max_box_expansion_factor * width)
    budget_height = int(max_box_expansion_factor * height)

    expand_width_lr = min(available_width_l, available_width_r, missing_width//2, budget_width)
    expand_width_l = min(available_width_l - expand_width_lr, missing_width - expand_width_lr * 2, budget_width - expand_width_lr)
    expand_width_r = min(available_width_r - expand_width_lr, missing_width - expand_width_lr * 2 - expand_width_l, budget_width - expand_width_lr - expand_width_l)

    expand_height_tb = min(available_height_t, available_height_b, missing_height//2, budget_height)
    expand_height_t = min(available_height_t - expand_height_tb, missing_height - expand_height_tb * 2, budget_height - expand_height_tb)
    expand_height_b = min(available_height_b - expand_height_tb, missing_height - expand_height_tb * 2 - expand_height_t, budget_height - expand_height_tb - expand_height_t)

    l, r = (l - math.floor(expand_width_lr/2) - expand_width_l,
            r + math.ceil(expand_width_lr/2) + expand_width_r)
    t, b = (t - math.floor(expand_height_tb/2) - expand_height_t,
            b + math.ceil(expand_height_tb/2) + expand_height_b)
    img = img[t:b + 1, l:r + 1]
    mask_img = mask_img[t:b + 1, l:r + 1]

    width, height = r - l + 1,  b - t + 1
    if down_scale_factor <= 1.0:
        scale_factor = down_scale_factor
    else:
        scale_factor = min(target_width / width, target_height / height)

    cropped_box = t, l, b, r
    assert img.shape[:2] == mask_img.shape[:2] == (cropped_box[2]-cropped_box[0]+1, cropped_box[3]-cropped_box[1]+1)
    return img, mask_img, cropped_box, scale_factor

