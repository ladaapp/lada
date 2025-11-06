# SPDX-FileCopyrightText: DOVER-video-quality-assessment-replicate Authors
# SPDX-License-Identifier: Apache-2.0 AND AGPL-3.0
# Code vendored from: https://github.com/turian/DOVER-video-quality-assessment-replicate

import os
import pathlib
import random

import cv2
import numpy as np
import torch
import yaml
# Importing necessary modules from the DOVER package
from lada.dover.datasets import (
    UnifiedFrameSampler,
    spatial_temporal_view_decomposition,
)
from lada.dover.models import DOVER


class VideoQualityEvaluator:
    def __init__(self, device=None):
        """Setup to load the model and configurations"""
        if device is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), "dover.yml")
        with open(config_path, "r") as f:
            self.opt = yaml.safe_load(f)

        # Initialize and load the model
        self.model = DOVER(**self.opt["model"]["args"]).to(self.device)
        self.model.load_state_dict(
            torch.load(
                os.path.join(os.path.dirname(__file__), self.opt["test_load_path"]),
                map_location=self.device,
            )
        )
        self.model.eval()

        self.mean = torch.FloatTensor([123.675, 116.28, 103.53]).to(self.device)
        self.std = torch.FloatTensor([58.395, 57.12, 57.375]).to(self.device)

    def evaluate(
            self, video, seed: int = 42
    ) -> dict:
        """Predict method to process video and output scores"""
        # Set seed for reproducibility
        self.set_seed(seed)

        video_path = str(video) if type(video) == str else "unspecified"

        dopt = self.opt["data"]["val-l1080p"]["args"]
        dopt["anno_file"] = None
        dopt["data_prefix"] = os.path.dirname(video_path)

        temporal_samplers = {}
        for stype, sopt in dopt["sample_types"].items():
            if "t_frag" not in sopt:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

        # View Decomposition
        views, _ = spatial_temporal_view_decomposition(
            video, dopt["sample_types"], temporal_samplers
        )

        for k, v in views.items():
            num_clips = dopt["sample_types"][k].get("num_clips", 1)
            views[k] = (
                ((v.to(self.device).permute(1, 2, 3, 0) - self.mean) / self.std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
            )

        results = [np.mean(r.cpu().numpy()) for r in self.model(views)]
        fused_results = self.fuse_results(results)

        return fused_results

    def fuse_results(self, results) -> dict:
        """Fuse aesthetic and technical results into final scores"""
        t, a = (results[0] - 0.1107) / 0.07355, (results[1] + 0.08285) / 0.03774
        x = t * 0.6104 + a * 0.3896
        return {
            "aesthetic": float(1 / (1 + np.exp(-a))),
            "technical": float(1 / (1 + np.exp(-t))),
            "overall": float(1 / (1 + np.exp(-x))),
        }

    def set_seed(self, seed):
        """Set the seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    evaluator = VideoQualityEvaluator()
    prediction = evaluator.evaluate(
        "mosaic_qual/fac955013dde39b939290afdcacaabcb60cfbf7d.m4v-000001.mp4")
    print(prediction)

    imgs_path = pathlib.Path("testme")
    imgs = []
    for img_path in imgs_path.glob("*"):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    print(f"loaded {len(imgs)} frames")

    prediction = evaluator.evaluate(imgs)
    print(prediction)
