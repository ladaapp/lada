# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import settings
from ultralytics.engine.results import Boxes as UltralyticsBoxes
from ultralytics.engine.results import Masks as UltralyticsMasks
from ultralytics.engine.results import Results as UltralyticsResults
from ultralytics.utils.ops import scale_image

from lada.utils import Box, Mask

def set_default_settings():
    settings.update({'runs_dir': './experiments/yolo', 'datasets_dir': './datasets', 'tensorboard': True})

def convert_yolo_box(yolo_box: UltralyticsBoxes, img_shape) -> Box:
    _box = yolo_box.xyxy[0]
    l = int(torch.clip(_box[0], 0, img_shape[1]).item())
    t = int(torch.clip(_box[1], 0, img_shape[0]).item())
    r = int(torch.clip(_box[2], 0, img_shape[1]).item())
    b = int(torch.clip(_box[3], 0, img_shape[0]).item())
    return t, l, b, r

def convert_yolo_boxes(yolo_box: UltralyticsBoxes, img_shape) -> list[Box]:
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

def scale_and_unpad_image(masks, im0_shape):
    h0, w0 = im0_shape[:2]
    h1, w1, _ = masks.shape
    if h1 == h0 and w1 == w0:
        return masks
    g = min(h1 / h0, w1 / w0)
    pw, ph = (w1 - w0 * g) / 2, (h1 - h0 * g) / 2
    t, l = round(ph - 0.1), round(pw - 0.1)
    b, r = h1 - round(ph + 0.1), w1 - round(pw + 0.1)
    x = masks[t:b, l:r].permute(2, 0, 1).unsqueeze(0).float()
    y = F.interpolate(x, size=(h0, w0), mode='bilinear', align_corners=False)
    return y.squeeze(0).permute(1, 2, 0).round_().clamp_(0, 255).to(masks.dtype)

def convert_yolo_mask_tensor(yolo_mask: UltralyticsMasks, img_shape) -> torch.Tensor:
    mask_img = _to_mask_img_tensor(yolo_mask.data)
    if mask_img.ndim == 2:
        mask_img = mask_img.unsqueeze(-1)
    mask_img = scale_and_unpad_image(mask_img, img_shape)
    mask_img = torch.where(mask_img > 127, 255, 0).to(torch.uint8)
    assert mask_img.ndim == 3 and mask_img.shape[2] == 1
    return mask_img

def _to_mask_img_tensor(masks: torch.Tensor, class_val=0, pixel_val=255) -> torch.Tensor:
    masks_tensor = torch.where(masks != class_val, pixel_val, 0).to(torch.uint8)
    return masks_tensor[0]

def convert_yolo_mask(yolo_mask: UltralyticsMasks, img_shape) -> Mask:
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

def choose_biggest_detection(result: UltralyticsResults, tracking_mode=True) -> tuple[UltralyticsBoxes | None, UltralyticsMasks | None]:
    """
    Returns the biggest detection box and mask of a YOLO Results set
    """
    box = None
    mask = None
    yolo_box: UltralyticsBoxes
    yolo_mask: UltralyticsMasks
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

def _get_unique_pixel_values(mask: Mask) -> list[int]:
    # get unique values except background (0)
    unique_values = np.unique(mask).tolist()
    if 0 in unique_values: unique_values.remove(0)  # remove background class
    return unique_values

def convert_segment_masks_to_yolo_labels(masks_dir, output_dir_segmentation_labels, output_dir_detection_labels, pixel_to_class_mapping):
    """
    pixel_to_class_mapping is a dict providing a mapping from pixel value to class id.
    e.g. if you only have a single class with id 0 and binary masks use pixel value 255 then this would be:
    pixel_to_class_mapping = {255: 0}

    Based of: ultralytics.data.converter.convert_segment_masks_to_yolo_seg
    """
    def get_yolo_box(contour) -> tuple[float]:
        x, y, w, h = cv2.boundingRect(contour)
        h, w = mask.shape[:2]
        center_x = x + w / 2
        center_y = y + h / 2
        yolo_box = center_x / w, center_y / h, w / w, h / h
        return yolo_box

    for mask_path in Path(masks_dir).iterdir():
        if mask_path.suffix in {".png", ".jpg"}:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            img_height, img_width = mask.shape

            unique_values = _get_unique_pixel_values(mask)
            yolo_segmentation_format_data = []
            yolo_detection_format_data = []

            for value in unique_values:
                class_index = pixel_to_class_mapping.get(value, -1)
                if class_index == -1:
                    print(f"Unknown class for pixel value {value} in file {mask_path}, skipping.")
                    continue

                # Create a binary mask for the current class and find contours
                binary_mask_for_current_class = (mask == value).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask_for_current_class, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if len(contour) >= 3:  # YOLO requires at least 3 points for a valid segmentation
                        contour = contour.squeeze()  # Remove single-dimensional entries
                        yolo_segmentation_format = [class_index]
                        for point in contour:
                            # Normalize the coordinates
                            yolo_segmentation_format.append(round(point[0] / img_width, 6))  # Rounding to 6 decimal places
                            yolo_segmentation_format.append(round(point[1] / img_height, 6))
                        yolo_segmentation_format_data.append(yolo_segmentation_format)
                        yolo_detection_format_data.append(get_yolo_box(contour))

            # Save Ultralytics YOLO format data to file
            output_path = Path(output_dir_segmentation_labels) / f"{mask_path.stem}.txt"
            with open(output_path, "w", encoding="utf-8") as file:
                for item in yolo_segmentation_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")
            output_path = Path(output_dir_detection_labels) / f"{mask_path.stem}.txt"
            with open(output_path, "w", encoding="utf-8") as file:
                for item in yolo_detection_format_data:
                    line = " ".join(map(str, item))
                    file.write(line + "\n")
