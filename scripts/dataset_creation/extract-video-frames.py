# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import os
from os import walk
import os.path as osp
import argparse
import cv2
import math
from tqdm import tqdm
from multiprocessing import Pool
import hashlib
import random

from lada.utils import video_utils

class FrameExtractor:
    def __init__(self, video_file, output_dir, sampling=-1, skip_start=None, skip_end=None):
        self.video_file = video_file
        self.sampling = sampling
        self.output_dir = output_dir
        self.video_capture = cv2.VideoCapture(self.video_file)
        self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.video_length = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_start = 0 if skip_start is None else math.ceil(skip_start * self.video_fps)
        self.frame_end = (self.video_length - 1) if skip_end is None else max(self.frame_start+1, (self.video_length - 1) - math.floor(skip_end * self.video_fps))

    def extract(self):
        if self.frame_start > self.frame_end or self.frame_start > self.video_length or self.frame_end < 0:
            return
        frame_cnt = self.frame_start
        self.video_capture.set(1, frame_cnt)
        success, frame = self.video_capture.read()
        while success and frame_cnt <= self.frame_end:
            filename = f"{hashlib.sha256(frame).hexdigest()}.jpg"
            file_path = osp.join(self.output_dir, filename)
            cv2.imwrite(file_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            if self.sampling != -1:
                frame_cnt += math.ceil(self.sampling * self.video_fps)
                self.video_capture.set(1, frame_cnt)
            else:
                frame_cnt += 1

            success, frame = self.video_capture.read()

def extract_video_frames(input):
    video_file = input[0]
    args = input[1]

    extractor = FrameExtractor(video_file=osp.join(args.input, video_file),
                               output_dir=args.output,
                               sampling=args.sampling,
                               skip_start=args.skip_start,
                               skip_end=args.skip_end)
    extractor.extract()

def main():
    parser = argparse.ArgumentParser("Extract video frames")
    parser.add_argument('--input', type=str, help='directory or single video file')
    parser.add_argument('--sampling', type=int, default=-1, help="extract 1 frame every --sampling seconds. Set to -1 to extract all frames")
    parser.add_argument('--output', type=str, default='extracted_frames', help="output root directory")
    parser.add_argument('--workers', type=int, default=None, help="worker count")
    parser.add_argument('--skip-start', type=float, default=None, help="skip first number of seconds")
    parser.add_argument('--skip-end', type=float, default=None, help="skip last number of seconds")
    args = parser.parse_args()

    if not osp.exists(args.output):
        os.makedirs(args.output)

    if osp.isfile(args.input):
        assert video_utils.is_video_file(args.input)
        extractor = FrameExtractor(video_file=args.input, output_dir=args.output, sampling=args.sampling)
        extractor.extract()
    else:
        root_input_dir = args.input
        assert args.sampling == -1 or args.sampling > 0
        video_list = []
        for r, d, f in walk(root_input_dir):
            for file in f:
                if video_utils.is_video_file(osp.join(r, file)):
                    video_list.append([osp.join(osp.relpath(r, root_input_dir), file), args])
        print(f"Detected files: {len(video_list)}")
        random.shuffle(video_list)
        with Pool(processes=args.workers) as p:
            with tqdm(total=len(video_list)) as pbar:
                for i, _ in enumerate(p.imap_unordered(extract_video_frames, video_list)):
                    pbar.update()


if __name__ == '__main__':
    main()
