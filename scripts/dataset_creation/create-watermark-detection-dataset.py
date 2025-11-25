# SPDX-FileCopyrightText: Lada Authors
# SPDX-FileCopyrightText: Watermarking Detection - DNN Authors
# SPDX-License-Identifier: AGPL-3.0
# Code vendored from: https://github.com/tgenlis83/dnn-watermark

import argparse

import glob
import math
import os
import random
from logging import getLogger
from typing import List
from tqdm import tqdm

from PIL import Image

from lada.utils.watermark_creation_utils import add_logo_watermark, add_text_watermark, load_fonts, convert_to_yolo

logger = getLogger(__name__)

def create_dataset(
    image_directory: str, logo_directory: str, yolo_labels_path: str, yolo_images_path: str, dataset_min_size: int
) -> None:
    font_paths_english = load_fonts("en")
    assert len(font_paths_english) > 0, "No English fonts found"
    font_paths_japanese = load_fonts("ja")
    assert len(font_paths_japanese) > 0, "No Japanese fonts found"
    logo_paths: List[str] = os.listdir(logo_directory)
    images_paths: List[str] = glob.glob(os.path.join(image_directory, '*'))
    image_multiplier = math.ceil(dataset_min_size / len(images_paths)) if len(images_paths) < dataset_min_size else 1
    images_paths = (images_paths * image_multiplier)[:dataset_min_size]
    random.shuffle(logo_paths)

    for idx, image_path in enumerate(tqdm(images_paths, desc="Generating watermarks")):
        image_pil: Image = Image.open(image_path)
        try:
            if idx % 2 == 0:
                logo_path: str = logo_directory + "/" + logo_paths[idx % len(logo_paths)]
                logo_pil: Image = Image.open(logo_path)
                assert logo_pil.mode == "RGBA"
                watermarked_image, bbox, category = add_logo_watermark(image_pil, logo_pil)
            else:
                lang = random.choice(["en", "ja"])
                font_paths = font_paths_japanese if lang == "ja" else font_paths_english
                font = font_paths[idx % len(font_paths)]
                watermarked_image, bbox, category = add_text_watermark(image_pil, font, lang=lang)
        except Exception as e:
            logger.warning(e)
            continue

        if bbox is None:
            # logger.warning("Empty bbox in image %s", image_properties["file_name"])
            continue

        stem, suffix = os.path.splitext(os.path.basename(image_path))
        file_name = f"{stem}_{idx}{suffix}"

        convert_to_yolo(
            file_name=file_name,
            bbox=bbox,
            category_id=category,
            yolo_labels_path=yolo_labels_path,
            yolo_images_path=yolo_images_path,
            watermarked_image=watermarked_image,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-images-dir', type=str)
    parser.add_argument('--val-images-dir', type=str)
    parser.add_argument('--logos-dir', type=str)
    parser.add_argument('--yolo-dir', type=str, default='datasets/watermark_detection')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    create_dataset(
        image_directory=args.train_images_dir,
        logo_directory=args.logos_dir,
        yolo_labels_path=os.path.join(args.yolo_dir, "train", "labels"),
        yolo_images_path=os.path.join(args.yolo_dir, "train", "images"),
        dataset_min_size=50_000
    )

    create_dataset(
        image_directory=args.val_images_dir,
        logo_directory=args.logos_dir,
        yolo_labels_path=os.path.join(args.yolo_dir, "val", "labels"),
        yolo_images_path=os.path.join(args.yolo_dir, "val", "images"),
        dataset_min_size=2_000
    )

if __name__ == "__main__":
    main()

