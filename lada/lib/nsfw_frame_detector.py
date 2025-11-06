# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import ultralytics.models
from lada.lib import Mask, Detection
from lada.lib import mask_utils
from lada.lib.ultralytics_utils import convert_yolo_box, convert_yolo_mask, choose_biggest_detection

def get_nsfw_frame(yolo_results: ultralytics.engine.results.Results, random_extend_masks: bool) -> Detection | None:
    yolo_box, yolo_mask = choose_biggest_detection(yolo_results, tracking_mode=False)
    object_detected = yolo_box is not None
    if not object_detected:
        return None

    mask = convert_yolo_mask(yolo_mask, yolo_results.orig_img.shape)
    mask = mask_utils.fill_holes(mask)
    # TODO: in a single yolo detection there could be multiple disconnected segments -> keep only the biggest area by contour and nuke the rest.
    #  Most often these are tiny false positive detections by NSFW detection model

    if random_extend_masks:
        mask = mask_utils.apply_random_mask_extensions(mask)
        box = mask_utils.get_box(mask)
    else:
        box = convert_yolo_box(yolo_box, yolo_results.orig_img.shape)

    t, l, b, r = box
    width, height = r - l + 1, b - t + 1
    if min(width, height) < 40:
        # skip tiny detections
        return None

    return Detection(yolo_results.orig_img, box, mask)

class NsfwImageDetector:
    def __init__(self, model: ultralytics.models.YOLO, device=None, random_extend_masks=False, conf=0.25):
        self.model = model
        self.device = device
        self.random_extend_masks = random_extend_masks
        self.conf = conf

    def detect(self, file_path: str) -> Detection | None:
        for results in self.model.predict(source=file_path, stream=False, verbose=False, device=self.device, conf=self.conf):
            return get_nsfw_frame(results, self.random_extend_masks)
