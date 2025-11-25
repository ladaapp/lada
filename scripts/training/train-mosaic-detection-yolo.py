# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from ultralytics import YOLO

from lada.lib.ultralytics_utils import set_default_settings

set_default_settings()

# !! uninstall albumentations, as it will blur and jpeg compress if installed and found by ultralytics. There seems to be no way to disable this in ultralytics

# "accurate" model
model = YOLO('yolo11s-seg.pt')
model.train(data='configs/yolo/mosaic_detection_dataset_config.yaml', epochs=200, imgsz=640, name="train_mosaic_detection_yolo11s")

# "fast" model
# model = YOLO('yolo11n-seg.pt')
# model.train(data='configs/yolo/mosaic_detection_dataset_config.yaml', epochs=200, imgsz=640, name="train_mosaic_detection_yolo11n")