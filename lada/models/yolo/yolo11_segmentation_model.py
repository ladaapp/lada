# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import nms, ops
from ultralytics.utils.checks import check_imgsz

from lada.utils import ImageTensor
from lada.utils.torch_letterbox import PyTorchLetterBox
from lada.utils.ultralytics_utils import UltralyticsResults

class Yolo11SegmentationModel:
    def __init__(self, model_path: str, device, imgsz=640, fp16=False, **kwargs):
        yolo_model = YOLO(model_path)
        assert yolo_model.task == 'segment'
        self.stride = 32
        self.imgsz = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        self.letterbox: PyTorchLetterBox|LetterBox = LetterBox(self.imgsz, auto=True, stride=self.stride)

        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict", "device": device, "half": fp16}
        args = {**yolo_model.overrides, **custom, **kwargs}  # highest priority args on the right
        self.args = get_cfg(DEFAULT_CFG, args)

        self.device: torch.device = torch.device(device)
        self.is_cuda_device: bool = self.device.type == 'cuda'
        self.model = AutoBackend(
            model=yolo_model.model,
            device=self.device,
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            fuse=True,
            verbose=False,
        )
        self.args.half = self.model.fp16
        self.model.eval()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))
        self.dtype = torch.float16 if fp16 else torch.float32

    def _preprocess_cpu(self, imgs: list[ImageTensor]) -> torch.Tensor:
        im = np.stack([self.letterbox(image=x.numpy()) for x in imgs])
        im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        return torch.from_numpy(im)
    
    def _preprocess_gpu(self, imgs: list[ImageTensor]) -> torch.Tensor:
        return self.letterbox(torch.stack(imgs, dim=0))

    def preprocess(self, imgs: list[ImageTensor]) -> list[torch.Tensor]:
        is_cpu_input = imgs[0].device.type == 'cpu'
        if is_cpu_input:
            return self._preprocess_cpu(imgs)
        else:
            if self.letterbox is None or imgs[0].shape[:2] != self.letterbox.original_shape:
                self.letterbox = PyTorchLetterBox(self.imgsz, imgs[0].shape[:2], stride=self.stride)
            return self._preprocess_gpu(imgs)

    def inference(self, image_batch: torch.Tensor):
        return self.model(image_batch, augment=False, visualize=False, embed=None)

    def inference_and_postprocess(self, imgs: torch.Tensor, orig_imgs: list[ImageTensor]) -> list[UltralyticsResults]:

        with torch.inference_mode():
            input = imgs.to(device=self.device).to(dtype=self.dtype).div_(255.0)
            preds = self.inference(input)
            return self.postprocess(preds, input, orig_imgs)

    def postprocess(self, preds, img, orig_imgs: list[ImageTensor]) -> list[Results]:
        protos = preds[0][-1]
        preds = nms.non_max_suppression(
            preds[0],
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
        )
        return [self.construct_result(pred, img, orig_img, proto) for pred, orig_img, proto in zip(preds, orig_imgs, protos)]

    def construct_result(self, preds: torch.tensor, img: torch.tensor, orig_img: ImageTensor, proto: torch.tensor):
        if not len(preds):  # save empty boxes
            masks = None
        else:
            masks = ops.process_mask(proto, preds[:, 6:], preds[:, :4], img.shape[2:], upsample=True)  # HWC
            preds[:, :4] = ops.scale_boxes(img.shape[2:], preds[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            preds, masks = preds[keep], masks[keep]
        return Results(orig_img, path='', names=self.model.names, boxes=preds[:, :6].cpu(), masks=masks)
