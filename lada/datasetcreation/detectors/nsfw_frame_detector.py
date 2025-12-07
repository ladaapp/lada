# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from ultralytics.engine.results import Results as UltralyticsResults
from lada.utils import Detections, Detection, Image, DETECTION_CLASSES
from lada.utils import mask_utils, ultralytics_utils
from lada.models.yolo.yolo import Yolo


def get_nsfw_frames(yolo_results: UltralyticsResults, random_extend_masks: bool) -> Detections | None:
    detections = []
    if not yolo_results.boxes:
        return None
    for yolo_box, yolo_mask in zip(yolo_results.boxes, yolo_results.masks):
        mask = ultralytics_utils.convert_yolo_mask(yolo_mask, yolo_results.orig_img.shape)
        box = ultralytics_utils.convert_yolo_box(yolo_box, yolo_results.orig_img.shape)
        conf = ultralytics_utils.convert_yolo_conf(yolo_box)
        mask, box = mask_utils.clean_mask(mask, box)
        mask = mask_utils.smooth_mask(mask, kernel_size=11)

        if random_extend_masks:
            mask = mask_utils.apply_random_mask_extensions(mask)
            mask = mask_utils.smooth_mask(mask, kernel_size=15)
            box = mask_utils.get_box(mask)

        t, l, b, r = box
        width, height = r - l + 1, b - t + 1
        if min(width, height) < 10:
            # skip tiny detections
            continue

        detections.append(Detection(DETECTION_CLASSES["nsfw"]["cls"], box, mask, conf))
    return Detections(yolo_results.orig_img, detections)

class NsfwImageDetector:
    def __init__(self, model: Yolo, device=None, random_extend_masks=False, conf=0.25):
        self.model = model
        self.device = device
        self.random_extend_masks = random_extend_masks
        self.conf = conf

    def detect(self, source: str | Image) -> Detections | None:
        for results in self.model.predict(source=source, stream=False, verbose=False, device=self.device, conf=self.conf, iou=0.):
            return get_nsfw_frames(results, self.random_extend_masks)