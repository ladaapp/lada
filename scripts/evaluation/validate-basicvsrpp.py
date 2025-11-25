# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import glob
import os.path

import torch
import cv2

from lada.basicvsrpp.inference import load_model, inference
from lada.lib.image_utils import pad_image, resize
from lada.lib.video_utils import read_video_frames, get_video_meta_data, write_frames_to_video_file

def validate(in_dir, out_dir, config_path, model_path, device):
    model = load_model(config_path, model_path, device)
    with torch.no_grad():
        for video_path in glob.glob(os.path.join(in_dir, '*')):
            video_metadata = get_video_meta_data(video_path)
            orig_images = read_video_frames(video_path, float32=False)
            orig_images = [torch.from_numpy(image) for image in orig_images]

            if orig_images[0].shape[:2] != (256, 256):
                size = 256
                for i, _ in enumerate(orig_images):
                    orig_images[i] = resize(orig_images[i], size, interpolation=cv2.INTER_LINEAR)
                    orig_images[i], _ = pad_image(orig_images[i], size, size, mode='zero')

            restored_images = inference(model, orig_images)
            restored_images = [image.cpu().numpy() for image in restored_images]
            filename = os.path.basename(video_path)
            out_path = os.path.join(out_dir, filename)
            fps = video_metadata.video_fps
            write_frames_to_video_file(restored_images, out_path, fps)


def parse_args():
    parser = argparse.ArgumentParser(description='Validate a model on a validation dataset')
    parser.add_argument('--out-dir', help='the dir to save logs and models')
    parser.add_argument('--in-dir')
    parser.add_argument('--model-path')
    parser.add_argument('--config-path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    validate(args.in_dir, args.out_dir, args.config_path, args.model_path, "cuda")