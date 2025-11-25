# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
import glob
import os.path
from lada.deepmosaics.models import loadmodel
from lada.deepmosaics.inference import restore_video_frames
from lada.lib.video_utils import read_video_frames, get_video_meta_data, write_frames_to_video_file

def validate(in_dir, out_dir, gpu_id, model_path):
    model = loadmodel.video(gpu_id, model_path)
    for video_path in glob.glob(os.path.join(in_dir, '*')):
        video_metadata = get_video_meta_data(video_path)
        images = read_video_frames(video_path, float32=False)
        restored_images = restore_video_frames(gpu_id, model, images)
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
    validate(args.in_dir, args.out_dir, '0', args.model_path)