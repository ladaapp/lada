# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from typing import Optional

from lada.lib.ultralytics_utils import convert_yolo_boxes
from lada.lib.scene_utils import box_overlap
from lada.lib import Image, Box
from ultralytics import YOLO


class WatermarkDetector:
    def __init__(self, model: YOLO, device):
        self.model = model
        self.device = device
        self.batch_size = 4
        self.min_confidence = 0.4
        self.min_positive_detections = 4
        self.sampling_rate = 0.3

    def detect(self, images:list[Image], boxes:Optional[list[Box]]=None) -> bool:
        num_samples = min(len(images), max(1, int(len(images) * self.sampling_rate)))
        indices_step_size = len(images) // num_samples
        indices = list(range(0, num_samples*indices_step_size, indices_step_size))
        samples = [images[i] for i in indices]
        samples_boxes = [boxes[i] for i in indices] if boxes else None

        batches = [samples[i:i + self.batch_size] for i in range(0, len(samples), self.batch_size)]
        positive_detections = 0
        for batch_idx, batch in enumerate(batches):
            # not exactly sure why but prediction accuracy is horrible if not setting imgsz to 640 even though model was trained with 512 in train-yolo-watermark-detector.py.
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
        watermark_detected = positive_detections >= self.min_positive_detections
        #print(f"watermark detector: watermark {watermark_detected}, detected {positive_detections}/{len(samples)}")
        return watermark_detected