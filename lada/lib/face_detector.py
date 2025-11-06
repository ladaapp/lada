# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import cv2
import numpy as np
from lada.lib import Mask, Box, Image, Detection
from lada.lib import mask_utils
from lada.centerface.centerface import CenterFace

def scale_box(box, mask_scale=1.0) -> Box:
    s = mask_scale - 1.0
    t, l, b, r = box
    w, h = r - l + 1, b - t + 1
    t -= h * s
    b += h * s
    l -= w * s
    r += w * s
    return int(t), int(l), int(b), int(r)

def choose_biggest_detection(dets) -> Box | None:
    max_area = -1
    max_box: Box | None = None
    for i, det in enumerate(dets):
        boxes, score = det[:4], det[4]
        x1, y1, x2, y2 = boxes.astype(int)
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            max_box = (int(y1), int(x1), int(y2), int(x2))
    return max_box

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

    color = 255
    thickness = -1

    cv2.ellipse(mask, center, axes, angle, start_angle, end_angle, color, thickness)

    mask = np.expand_dims(mask, axis=-1)

    return mask

def get_nsfw_frame(dets, frame, random_extend_masks: bool, mask_scale: float) -> Detection | None:
    box = choose_biggest_detection(dets)
    object_detected = box is not None
    if not object_detected:
        return None
    box = scale_box(box, mask_scale)
    mask = create_mask(frame, box)

    if random_extend_masks:
        mask = mask_utils.apply_random_mask_extensions(mask)
        box = mask_utils.get_box(mask)

    t, l, b, r = box
    width, height = r - l + 1, b - t + 1
    if min(width, height) < 40:
        # skip tiny detections
        return None

    return Detection(frame, box, mask)

class FaceDetector:
    def __init__(self, model: CenterFace, random_extend_masks=False, conf=0.2, mask_scale=1.3):
        self.model = model
        self.random_extend_masks = random_extend_masks
        self.conf = conf
        self.mask_scale = mask_scale

    def detect(self, file_path: str) -> Detection | None:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR_RGB)
        dets, _ = self.model(image, threshold=self.conf)
        return get_nsfw_frame(dets, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), random_extend_masks=self.random_extend_masks, mask_scale=self.mask_scale)
