# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from lada.utils.ultralytics_utils import set_default_settings
from lada.models.yolo.yolo import Yolo

set_default_settings()

model = Yolo('yolo11m-seg.pt')
model.train(data='configs/yolo/nsfw_detection_dataset_config.yaml', epochs=200, imgsz=640, name="train_nsfw_detection_yolo11m")
