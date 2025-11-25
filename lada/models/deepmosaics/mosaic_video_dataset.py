# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import json
import os.path
import random
import glob

import numpy as np
import torch
import torch.utils.data as data
import lada.utils.video_utils as video_utils
from lada.utils import image_utils


class MosaicVideoDataset(data.Dataset):
    def __init__(self, opt):
        super(MosaicVideoDataset, self).__init__()
        self.lq_size = opt.get('lq_size', 256)
        self.gt_size = opt.get('gt_size', 256)
        self.gt_root, self.lq_root, self.meta_root = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_meta']
        self.max_frame_count = opt['num_frame']
        self.min_frame_count = opt['min_num_frame'] if 'min_num_frame' in opt else opt['num_frame']
        self.S = opt.get('S', 3)
        self.T = opt.get('T', 2)

        self.clip_names = []
        self.total_num_frames = []
        for meta_path in glob.glob(os.path.join(self.meta_root, '*')):
            with open(meta_path, 'r') as meta_file:
                meta_json = json.load(meta_file)
                filename = f"{os.path.splitext(os.path.basename(meta_path))[0]}.mp4"
                frame_num = meta_json["frame_count"]
                if frame_num < self.min_frame_count:
                    continue
                self.clip_names.append(filename)
                self.total_num_frames.append(frame_num)


    def __getitem__(self, index):
        clip_name = self.clip_names[index]
        total_num_frames = self.total_num_frames[index]

        if self.max_frame_count == -1:
            # select the full clip
            start_frame_idx = 0
            end_frame_idx = total_num_frames - 1
        else:
            # randomly select shorter clip of length num_frame
            start_frame_idx = random.randint(0, total_num_frames - self.max_frame_count)
            end_frame_idx = start_frame_idx + self.max_frame_count

        # get the neighboring LQ and GT frames
        vid_lq_path = os.path.join(self.lq_root, clip_name)
        vid_gt_path = os.path.join(self.gt_root, clip_name)
        img_lqs = video_utils.read_video_frames(vid_lq_path, float32=True, start_idx=start_frame_idx, end_idx=end_frame_idx, normalize_neg1_pos1=True)
        img_gts = video_utils.read_video_frames(vid_gt_path, float32=True, start_idx=start_frame_idx, end_idx=end_frame_idx, normalize_neg1_pos1=True)

        img_gts = torch.stack(image_utils.img2tensor(img_gts), dim=0)
        img_lqs = torch.stack(image_utils.img2tensor(img_lqs), dim=0)

        img_lqs_batch = []
        img_gts_batch = []
        for i in range(start_frame_idx, end_frame_idx + 1):
            frame_indices = np.linspace(0, (self.T-1)*self.S,self.T,dtype=np.int64)
            # flip some dims -> T,C,H,W -> C,T,H,W
            img_gts_batch_item = img_gts[frame_indices]
            img_gts_batch_item = torch.transpose(img_gts_batch_item, 0, 1)
            img_lqs_batch_item = img_lqs[frame_indices]
            img_lqs_batch_item = torch.transpose(img_lqs_batch_item, 0, 1)

            img_gts_batch.append(img_gts_batch_item)
            img_lqs_batch.append(img_lqs_batch_item)

        # img_lqs_batch: (B,C,T,H,W)
        # img_gts_batch: (B,C,T,H,W)
        #print(f"est data sample item size in MB: {sum([arr.nbytes for arr in img_gts_batch] + [arr.nbytes for arr in img_lqs_batch]) / 1024 / 1024}")
        img_gts_batch = torch.stack(img_gts_batch, dim=0)
        img_lqs_batch = torch.stack(img_lqs_batch, dim=0)
        return img_gts_batch, img_lqs_batch

    def __len__(self):
        return len(self.clip_names)