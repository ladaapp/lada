# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
import os

from lada import MODEL_WEIGHTS_DIR
from lada.utils.ultralytics_utils import set_default_settings
from lada.models.yolo.yolo import Yolo

set_default_settings()

# "accurate" model
model = Yolo(os.path.join(MODEL_WEIGHTS_DIR, '3rd_party', 'yolo11s-seg.pt'))
model.train(data='configs/yolo/mosaic_detection_dataset_config.yaml', epochs=200, imgsz=640, name="train_mosaic_detection_yolo11s", augmentations=[])

# "fast" model
# model = Yolo(os.path.join(MODEL_WEIGHTS_DIR, '3rd_party', 'yolo11n-seg.pt'))
# model.train(data='configs/yolo/mosaic_detection_dataset_config.yaml', epochs=200, imgsz=640, name="train_mosaic_detection_yolo11n", augmentations=[])