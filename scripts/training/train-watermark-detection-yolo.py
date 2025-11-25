# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from lada.utils.ultralytics_utils import set_default_settings
from lada.models.yolo.yolo import Yolo

set_default_settings()

model = Yolo('yolo11s.pt')
model.train(data='configs/yolo/watermark_detection_dataset_config.yaml', epochs=100, imgsz=512, single_cls=True, name="train_watermark_detection_yolo11s")
