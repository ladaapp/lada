# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import os.path
from pathlib import Path

import cv2
import numpy as np

from lada.models.yolo.yolo import Yolo
from lada.utils import image_utils, video_utils
from lada.utils.video_utils import process_video_v3

def process_frame(input: np.ndarray[np.uint8], model, conf, classes, imgsz, iou, negate=False):
    result = model.predict(input, conf=conf, imgsz=imgsz, iou=iou, verbose=False, classes=classes)
    detected = len(result[0].boxes) > 0
    if detected ^ negate:
        return result[0].plot()
    return input

def process_image(input: str, output_path, model, conf, classes, imgsz, iou, negate=False):
    in_image = cv2.imread(input)
    result = model.predict(in_image, conf=conf, imgsz=imgsz, iou=iou, verbose=False, classes=classes)
    detected = len(result[0].boxes) > 0
    if detected ^ negate:
        out_image = result[0].plot()
    else:
        out_image = in_image
    cv2.imwrite(output_path, out_image)

def process_file(input_path, args):
    output_path = str(Path(args.output_dir).joinpath(Path(input_path).name))

    frame = cv2.imread(input_path)
    if frame is None:
        process_video_v3(input_path, output_path, lambda in_frame: process_frame(in_frame, model, args.conf, classes=[args.class_id] if args.class_id is not None else None, imgsz=args.imgsz, iou=args.iou, negate=args.negate))
    else:
        # input is an image file
        process_image(input_path, output_path, model, args.conf, classes=[args.class_id] if args.class_id is not None else None, imgsz=args.imgsz, iou=args.iou, negate=args.negate)

def get_files(dir):
    file_list = []
    for r, d, f in os.walk(dir):
        for file in f:
            file_path = os.path.join(r, file)
            if image_utils.is_image_file(file_path) or video_utils.is_video_file(file_path):
                file_list.append(Path(file_path))
    return file_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to image or video file or directory containing image or video files")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory path to save predictions")
    parser.add_argument('--model-path', type=str, required=True, help="Path to YOLO model weights file")
    parser.add_argument('--conf', type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument('--iou', type=float, default=0.7, help="IoU (Intersection over union) used for NMS (Non-Maximum-Suppression)")
    parser.add_argument('--imgsz', type=int, default=640, help="Target Image/Frame resolution. Image/Frame will be scaled up/down accordingly. Needs to be a multiple of 32 to match YOLO stride size")
    parser.add_argument('--class-id', type=int, default=None, help="If set will only consider detections of this single class id. Can be used together with --negate to consider all BUT this single class id")
    parser.add_argument('--negate', default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    model = Yolo(args.model_path)
    os.makedirs(args.output_dir, exist_ok=True)

    input_path = Path(args.input)
    if input_path.is_file():
        process_file(args.input, args)
    elif input_path.is_dir():
        files = get_files(input_path)
        print(f"Found {len(files)} files")
        for file_index, dir_entry in enumerate(get_files(input_path)):
            print(f"{file_index}, Processing {Path(dir_entry).name}")
            process_file(str(dir_entry), args)
