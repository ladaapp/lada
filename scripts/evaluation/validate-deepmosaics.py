# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import glob
import os.path

import cv2
import torch

from lada.utils.image_utils import pad_image, resize
from lada.utils.video_utils import read_video_frames, get_video_meta_data, write_frames_to_video_file
from lada.models.deepmosaics.inference import restore_video_frames
from lada.models.deepmosaics.models import loadmodel

def validate(in_dir, out_dir, device, model_path):
    model = loadmodel.video(device, model_path, False)
    for video_path in glob.glob(os.path.join(in_dir, '*')):
        video_metadata = get_video_meta_data(video_path)
        images = read_video_frames(video_path, float32=False)

        if images[0].shape[:2] != (256, 256):
            size = 256
            for i, _ in enumerate(images):
                images[i] = resize(images[i], size, interpolation=cv2.INTER_LINEAR)
                images[i], _ = pad_image(images[i], size, size, mode='reflect')

        restored_images = restore_video_frames(device.index, model, images)
        filename = os.path.basename(video_path)
        out_path = os.path.join(out_dir, filename)
        fps = video_metadata.video_fps
        write_frames_to_video_file(restored_images, out_path, fps)

def parse_args():
    parser = argparse.ArgumentParser(description='Validate a model on a validation dataset')
    parser.add_argument('--out-dir')
    parser.add_argument('--in-dir')
    parser.add_argument('--model-path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    validate(args.in_dir, args.out_dir, torch.device("cuda:0"), args.model_path)