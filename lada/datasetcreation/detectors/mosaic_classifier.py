# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from typing import Optional

from lada.utils.ultralytics_utils import convert_yolo_boxes
from lada.utils.box_utils import box_overlap
from lada.utils import Image, Box, Detections, ultralytics_utils, Detection, DETECTION_CLASSES
from lada.models.yolo.yolo import Yolo

class MosaicClassifier:
    def __init__(self, model: Yolo, device, min_confidence=0.8):
        self.model = model
        self.device = device
        self.batch_size = 4
        self.min_confidence = min_confidence
        self.min_positive_detections = 4
        self.sampling_rate = 0.3

    def detect_batch(self, images:list[Image], boxes:Optional[list[Box]]=None) -> bool:
        num_samples = min(len(images), max(1, int(len(images) * self.sampling_rate)))
        indices_step_size = len(images) // num_samples
        indices = list(range(0, num_samples*indices_step_size, indices_step_size))
        samples = [images[i] for i in indices]
        samples_boxes = [boxes[i] for i in indices] if boxes else None

        batches = [samples[i:i + self.batch_size] for i in range(0, len(samples), self.batch_size)]
        positive_detections = 0
        for batch_idx, batch in enumerate(batches):
            batch_prediction_results = self.model.predict(source=batch, stream=False, verbose=False, device=self.device, conf=self.min_confidence, imgsz=640)
            for result_idx, results in enumerate(batch_prediction_results):
                sample_idx = batch_idx * self.batch_size + result_idx
                detections = results.boxes.conf.size(dim=0)
                if detections == 0:
                    continue
                detection_confidences = results.boxes.conf.tolist()
                detection_boxes = convert_yolo_boxes(results.boxes, results.orig_shape)
                single_image_watermark_detected = any(conf > self.min_confidence and (not samples_boxes or box_overlap(detection_boxes[i], samples_boxes[sample_idx])) for i, conf in enumerate(detection_confidences))
                if single_image_watermark_detected:
                    positive_detections += 1
                    if positive_detections >= self.min_positive_detections:
                        return True
        return False

    def detect(self, source: str | Image) -> Detections | None:
        for yolo_results in self.model.predict(source=source, stream=False, verbose=False, device=self.device, conf=self.min_confidence, imgsz=640):
            detections = []
            if not yolo_results.boxes:
                return None
            for yolo_box, yolo_mask in zip(yolo_results.boxes, yolo_results.masks):
                mask = ultralytics_utils.convert_yolo_mask(yolo_mask, yolo_results.orig_img.shape)
                box = ultralytics_utils.convert_yolo_box(yolo_box, yolo_results.orig_img.shape)
                t, l, b, r = box
                width, height = r - l + 1, b - t + 1
                if min(width, height) < 20:
                    # skip tiny detections
                    continue

                detections.append(Detection(DETECTION_CLASSES["mosaic"]["cls"], box, mask))
            return Detections(yolo_results.orig_img, detections)