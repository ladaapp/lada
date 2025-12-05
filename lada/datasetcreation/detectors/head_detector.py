# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import cv2
import numpy as np
from lada.utils import Mask, Box, Image, Detections, Detection, DETECTION_CLASSES
from lada.utils import box_utils
from lada.models.bpjdet.inference import inference

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

    color = DETECTION_CLASSES["sfw_head"]["mask_value"]
    thickness = -1

    cv2.ellipse(mask, center, axes, angle, start_angle, end_angle, color, thickness)

    mask = np.expand_dims(mask, axis=-1)

    return mask

def _get_detection(dets: list[Box], frame, random_extend_masks: bool) -> Detections | None:
    if len(dets) == 0:
        return None
    detections = []
    for box in dets:
        if random_extend_masks:
            box = box_utils.random_scale_box(frame, box, scale_range=(0.9, 1.2))
        mask = _create_mask(frame, box)

        t, l, b, r = box
        width, height = r - l + 1, b - t + 1
        if min(width, height) < 40:
            # skip tiny detections
            continue
        detections.append(Detection(DETECTION_CLASSES["sfw_head"]["cls"], box, mask))
    return Detections(frame, detections)

class HeadDetector:
    def __init__(self, model, data, conf_thres,  iou_thres, imgz=1536, random_extend_masks=False):
        self.model = model
        self.data = data
        self.random_extend_masks = random_extend_masks
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgz = imgz

    def detect(self, source: str | Image) -> Detections | None:
        if isinstance(source, str):
            image = cv2.imread(source)
        else:
            image = source
        dets = inference(self.model, source, data=self.data, conf_thres=self.conf_thres, iou_thres=self.iou_thres, imgz=self.imgz)
        return _get_detection(dets, image, random_extend_masks=self.random_extend_masks)
