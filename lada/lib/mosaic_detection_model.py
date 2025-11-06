# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils import nms, ops
from ultralytics.engine.results import Results
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics import YOLO
from lada.lib.torch_letterbox import PyTorchLetterBox
from typing import List

class MosaicDetectionModel:
    def __init__(self, model_path: str, device, imgsz=640, fp16=False, **kwargs):
        yolo_model = YOLO(model_path)
        assert yolo_model.task == 'segment'
        self.stride = 32
        self.imgsz = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        self.letterbox = PyTorchLetterBox(self.imgsz, stride=self.stride)

        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict", "device": device, "half": fp16}
        args = {**yolo_model.overrides, **custom, **kwargs}  # highest priority args on the right
        self.args = get_cfg(DEFAULT_CFG, args)

        self.model = AutoBackend(
            model=yolo_model.model,
            device=torch.device(device),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            fuse=True,
            verbose=False,
        )
        self.device = self.model.device
        self.args.half = self.model.fp16
        self.model.eval()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        self.dtype = torch.float16 if fp16 else torch.float32

        self.is_segmentation_model = yolo_model.task == 'segment'

    def preprocess(self, imgs: list[torch.Tensor]) -> torch.Tensor:
        im = torch.stack([x.permute(2, 0, 1) for x in imgs]).to(dtype=self.dtype, memory_format=torch.channels_last).div_(255.0) # (H, W, C) to (C, H, W)
        return self.letterbox(im)

    def inference(self, image_batch: torch.Tensor):
        return self.model(image_batch, augment=False, visualize=False, embed=None)

    def postprocess(self, preds, img, orig_imgs: List[torch.Tensor]) -> List[Results]:
        protos = preds[1][-1]
        preds = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
        )
        return [self.construct_result(pred, img, orig_img, proto) for pred, orig_img, proto in zip(preds, orig_imgs, protos)]

    def construct_result(self, preds: torch.tensor, img: torch.tensor, orig_img: torch.Tensor, proto: torch.tensor):
        if not len(preds):  # save empty boxes
            masks = None
        else:
            masks = ops.process_mask(proto, preds[:, 6:], preds[:, :4], img.shape[2:], upsample=True)  # HWC
            preds[:, :4] = ops.scale_boxes(img.shape[2:], preds[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            preds, masks = preds[keep], masks[keep]
        return Results(orig_img, path='', names=self.model.names, boxes=preds[:, :6].cpu(), masks=masks)
