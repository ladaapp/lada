# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import cv2
import numpy as np
from lada.lib import Mask, Box, Image, Detection
from lada.lib import mask_utils
from lada.bpjdet.inference import get_model, inference

def _choose_biggest_detection(dets: list[Box]) -> Box | None:
    max_area = -1
    max_box: Box | None = None
    for i, box in enumerate(dets):
        t, l, b, r = box
        area = (r - l) * (b - t)
        if area > max_area:
            max_area = area
            max_box = box
    return max_box

def _create_mask(frame: Image, box: Box) -> Mask:
    t, l, b, r = box
    box_width, box_height = r - l + 1, b - t + 1

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Set the center of the ellipse at the center of the box
    center = (l + (box_width // 2), t + (box_height // 2))

    # Set the axes of the ellipse to half the width and half the height of the box
    axes = (box_width // 2, box_height // 2)

    angle = 0
    start_angle = 0
    end_angle = 360

    color = 255
    thickness = -1

    cv2.ellipse(mask, center, axes, angle, start_angle, end_angle, color, thickness)

    mask = np.expand_dims(mask, axis=-1)

    return mask

def _get_detection(dets: list[Box], frame, random_extend_masks: bool) -> Detection | None:
    box = _choose_biggest_detection(dets)
    object_detected = box is not None
    if not object_detected:
        return None
    mask = _create_mask(frame, box)

    if random_extend_masks:
        mask = mask_utils.apply_random_mask_extensions(mask)
        box = mask_utils.get_box(mask)

    t, l, b, r = box
    width, height = r - l + 1, b - t + 1
    if min(width, height) < 40:
        # skip tiny detections
        return None

    return Detection(frame, box, mask)

class HeadDetector:
    def __init__(self, model, data, conf_thres,  iou_thres, imgz=1024, random_extend_masks=False):
        self.model = model
        self.data = data
        self.random_extend_masks = random_extend_masks
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgz = imgz

    def detect(self, file_path: str) -> Detection | None:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR_RGB)
        dets = inference(self.model, file_path, data=self.data, conf_thres=self.conf_thres, iou_thres=self.iou_thres, imgz=self.imgz)
        return _get_detection(dets, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), random_extend_masks=self.random_extend_masks)
