# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import cv2
import numpy as np
from lada.lib import Mask, Box, Image, Detection, Detections, DETECTION_CLASSES
from lada.lib import box_utils
from lada.centerface.centerface import CenterFace

def convert_to_boxes(dets) -> list[Box]:
    boxes = []
    for i, det in enumerate(dets):
        box, score = det[:4], det[4]
        x1, y1, x2, y2 = box.astype(int)
        boxes.append((int(y1), int(x1), int(y2), int(x2)))
    return boxes

def create_mask(frame: Image, box: Box) -> Mask:
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

    color = DETECTION_CLASSES["sfw_face"]["mask_value"]
    thickness = -1

    cv2.ellipse(mask, center, axes, angle, start_angle, end_angle, color, thickness)

    mask = np.expand_dims(mask, axis=-1)

    return mask

def get_nsfw_frame(dets: list[Box], frame: Image, random_extend_masks: bool) -> Detections | None:
    if len(dets) == 0:
        return None
    detections = []
    for box in dets:

        if random_extend_masks:
            box = box_utils.random_scale_box(frame, box, scale_range=(1.2, 1.5))
        mask = create_mask(frame, box)

        t, l, b, r = box
        width, height = r - l + 1, b - t + 1
        if min(width, height) < 40:
            # skip tiny detections
            return None

        detections.append(Detection("sfw_face", box, mask))
    return Detections(frame, detections)

class FaceDetector:
    def __init__(self, model: CenterFace, random_extend_masks=False, conf=0.2):
        self.model = model
        self.random_extend_masks = random_extend_masks
        self.conf = conf

    def detect(self, file_path: str) -> Detections | None:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR_RGB)
        dets, _ = self.model(image, threshold=self.conf)
        dets_boxes = convert_to_boxes(dets)
        return get_nsfw_frame(dets_boxes, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), random_extend_masks=self.random_extend_masks)
