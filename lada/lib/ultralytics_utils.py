# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics.engine
from ultralytics import settings
from ultralytics.utils.ops import scale_image

from lada.lib import Box, Mask, mask_utils

def set_default_settings():
    settings.update({'runs_dir': './experiments/yolo', 'datasets_dir': './datasets', 'tensorboard': True})

def convert_yolo_box(yolo_box: ultralytics.engine.results.Boxes, img_shape) -> Box:
    _box = yolo_box.xyxy[0]
    l = int(torch.clip(_box[0], 0, img_shape[1]).item())
    t = int(torch.clip(_box[1], 0, img_shape[0]).item())
    r = int(torch.clip(_box[2], 0, img_shape[1]).item())
    b = int(torch.clip(_box[3], 0, img_shape[0]).item())
    return t, l, b, r

def convert_yolo_boxes(yolo_box: ultralytics.engine.results.Boxes, img_shape) -> list[Box]:
    _boxes = yolo_box.xyxy
    boxes = []
    for _box in _boxes:
        l = int(torch.clip(_box[0], 0, img_shape[1]).item())
        t = int(torch.clip(_box[1], 0, img_shape[0]).item())
        r = int(torch.clip(_box[2], 0, img_shape[1]).item())
        b = int(torch.clip(_box[3], 0, img_shape[0]).item())
        box = t, l, b, r
        boxes.append(box)
    return boxes

def convert_yolo_mask(yolo_mask: ultralytics.engine.results.Masks, img_shape) -> Mask:
    mask_img = _to_mask_img(yolo_mask.data)
    if mask_img.ndim == 2:
        mask_img = np.expand_dims(mask_img, axis=-1)
    mask_img = scale_image(mask_img, img_shape)
    mask_img = np.where(mask_img > 127, 255, 0).astype(np.uint8)
    assert mask_img.ndim == 3 and mask_img.shape[2] == 1
    return mask_img


def _to_mask_img(masks, class_val=0, pixel_val=255) -> Mask:
    masks_tensor = (masks != class_val).int() * pixel_val
    mask_img = masks_tensor.cpu().numpy()[0].astype(np.uint8)
    return mask_img


def choose_biggest_detection(result: ultralytics.engine.results.Results, tracking_mode=True) -> tuple[
    ultralytics.engine.results.Boxes | None, ultralytics.engine.results.Masks | None]:
    """
    Returns the biggest detection box and mask of a YOLO Results set
    """
    box = None
    mask = None
    yolo_box: ultralytics.engine.results.Boxes
    yolo_mask: ultralytics.engine.results.Masks
    for i, yolo_box in enumerate(result.boxes):
        if tracking_mode and yolo_box.id is None:
            continue
        yolo_mask = result.masks[i]
        if box is None:
            box = yolo_box
            mask = yolo_mask
        else:
            box_dims = box.xywh[0]
            _box_dims = yolo_box.xywh[0]
            box_size = box_dims[2] * box_dims[3]
            _box_size = _box_dims[2] * _box_dims[3]
            if _box_size > box_size:
                box = yolo_box
                mask = yolo_mask
    return box, mask

def convert_segment_masks_to_yolo_segmentation_labels(masks_dir, output_dir, pixel_to_class_mapping):
    """
    pixel_to_class_mapping is a dict providing a mapping from pixel value to class id.
    e.g. if you only have a single class with id 0 and binary masks use pixel value 255 then this would be:
    pixel_to_class_mapping = {255: 0}

    source: ultralytics.data.converter.convert_segment_masks_to_yolo_seg
    """
    for mask_path in Path(masks_dir).iterdir():
        if mask_path.suffix in {".png", ".jpg"}:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            img_height, img_width = mask.shape

            unique_values = np.unique(mask)  # Get unique pixel values representing different classes
            yolo_format_data = []

            for value in unique_values:
                if value == 0:
                    continue  # Skip background
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    print(f"Unknown class for pixel value {value} in file {mask_path}, skipping.")
                    continue

                # Create a binary mask for the current class and find contours
                contours, _ = cv2.findContours(
                    (mask == value).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )  # Find contours

                for contour in contours:
                    if len(contour) >= 3:  # YOLO requires at least 3 points for a valid segmentation
                        contour = contour.squeeze()  # Remove single-dimensional entries
                        yolo_format = [class_index]
                        for point in contour:
                            # Normalize the coordinates
                            yolo_format.append(round(point[0] / img_width, 6))  # Rounding to 6 decimal places
                            yolo_format.append(round(point[1] / img_height, 6))
                        yolo_format_data.append(yolo_format)
            # Save Ultralytics YOLO format data to file
            output_path = Path(output_dir) / f"{mask_path.stem}.txt"
            with open(output_path, "w", encoding="utf-8") as file:
                for item in yolo_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")


def convert_binary_mask_to_yolo_detection_labels(masks_dir, output_dir, pixel_to_class_mapping):
    """
    pixel_to_class_mapping is a dict providing a mapping from pixel value to class id.
    e.g. if you only have a single class with id 0 and binary masks use pixel value 255 then this would be:
    pixel_to_class_mapping = {255: 0}

    currently only binary masks are supported so single key-value pair in pixel_to_class_mapping

    """
    assert len(pixel_to_class_mapping.keys()) == 1, 'only single class / mapping currently supported'
    class_id = list(pixel_to_class_mapping.values())[0]

    def _convert_binary_mask_to_yolo_detection_labels(mask: Mask) -> tuple[float]:
        t, l, b, r = mask_utils.get_box(mask)
        h, w = mask.shape[:2]
        box_width = r - l
        box_height = b - t
        box_center_x = l + box_width / 2
        box_center_y = t + box_height / 2
        yolo_box = box_center_x / w, box_center_y / h, box_width / w, box_height / h
        return yolo_box

    for mask_path in Path(masks_dir).iterdir():
        if mask_path.suffix in {".png", ".jpg"}:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            yolo_box = _convert_binary_mask_to_yolo_detection_labels(mask)
            label_file_path = Path(output_dir).joinpath(Path(mask_path).with_suffix('.txt').name)
            with open(label_file_path, 'a') as file:
                file.write(f"{class_id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}")
