# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from typing import Optional

from lada.lib.ultralytics_utils import convert_yolo_boxes
from lada.lib.scene_utils import box_overlap
from lada.lib import Image, Box
from ultralytics import YOLO


MALE_GENITALIA_EXPOSED = 14
FEMALE_GENITALIA_EXPOSED = 4

NSFW_CLASS_IDS = (MALE_GENITALIA_EXPOSED, FEMALE_GENITALIA_EXPOSED)

class NudeNetNsfwDetector:
    def __init__(self, model: YOLO, device):
        self.model = model
        self.device = device
        self.batch_size = 4
        self.min_confidence = 0.15
        self.min_positive_detections = 6
        self.sampling_rate = 0.3

    def detect(self, images:list[Image], boxes:Optional[list[Box]]=None) -> tuple[bool, bool, bool]:
        num_samples = min(len(images), max(1, int(len(images)*self.sampling_rate)))
        indices_step_size = len(images) // num_samples
        indices_of_nsfw_elements = list(range(0, num_samples*indices_step_size, indices_step_size))
        samples = [images[i] for i in indices_of_nsfw_elements]
        samples_boxes = [boxes[i] for i in indices_of_nsfw_elements] if boxes else None

        batches = [samples[i:i + self.batch_size] for i in range(0, len(samples), self.batch_size)]
        positive_detections = 0
        positive_male_detections = 0
        positive_female_detections = 0
        for batch_idx, batch in enumerate(batches):
            batch_prediction_results = self.model.predict(source=batch, stream=False, verbose=False, device=self.device, conf=self.min_confidence)
            for result_idx, results in enumerate(batch_prediction_results):
                sample_idx = batch_idx * self.batch_size + result_idx
                detections = results.boxes.conf.size(dim=0)
                if detections == 0:
                    continue
                cls = results.boxes.cls.tolist()
                indices_of_nsfw_elements = [i for i in range(len(cls)) if cls[i] in NSFW_CLASS_IDS]
                if len(indices_of_nsfw_elements) == 0:
                    continue
                conf = results.boxes.conf.tolist()
                detection_boxes = convert_yolo_boxes(results.boxes, results.orig_shape)

                single_image_nsfw_male_detected = False
                single_image_nsfw_female_detected = False
                for i in indices_of_nsfw_elements:
                    if conf[i] > self.min_confidence and (not samples_boxes or box_overlap(detection_boxes[i], samples_boxes[sample_idx])):
                        if not single_image_nsfw_male_detected:
                            single_image_nsfw_male_detected = cls[i] == MALE_GENITALIA_EXPOSED
                        if not single_image_nsfw_female_detected:
                            single_image_nsfw_female_detected = cls[i] == FEMALE_GENITALIA_EXPOSED
                single_image_nsfw_detected = single_image_nsfw_male_detected or single_image_nsfw_female_detected
                if single_image_nsfw_detected:
                    positive_detections += 1
                if single_image_nsfw_male_detected:
                    positive_male_detections += 1
                if single_image_nsfw_female_detected:
                    positive_female_detections += 1
        nsfw_detected = positive_detections >= self.min_positive_detections
        nsfw_male_detected = positive_male_detections > self.min_positive_detections
        nsfw_female_detected = positive_female_detections > self.min_positive_detections
        #print(f"nudenet nsfw detector: nsfw {nsfw_detected}, detected {positive_detections}/{len(samples)}")
        return nsfw_detected, nsfw_male_detected, nsfw_female_detected