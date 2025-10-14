import threading

import torch
from ultralytics.utils.checks import check_imgsz
import numpy as np
from ultralytics.utils import nms, ops
from ultralytics.engine.results import Results
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics import YOLO
from lada.lib import Image
from typing import List, Union, Tuple
import torch.nn as nn
import torchvision.transforms.v2 as T
import torch.nn.functional as F

class OptimizedLetterBox:
    """Optimized LetterBox for torch.Tensor images with auto=True and no label support."""
    
    def __init__(
        self,
        new_shape: tuple[int, int] = (640, 640),
        scaleup: bool = True,
        stride: int = 32,
        padding_value: float = 0.447,  # 114/255 for normalized images
    ):
        """
        Initialize OptimizedLetterBox for torch.Tensor images.
        
        Args:
            new_shape (tuple[int, int]): Target size (height, width) for the resized image.
            scaleup (bool): If True, allow scaling up. If False, only scale down.
            stride (int): Stride of the model (e.g., 32 for YOLOv5).
            padding_value (float): Normalized padding value (0-1 range).
        """
        self.new_shape = new_shape
        self.scaleup = scaleup
        self.stride = stride
        self.padding_value = padding_value

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Resize and pad a torch.Tensor image with letterboxing.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (H, W, C)
            
        Returns:
            torch.Tensor: Resized and padded image tensor of shape (H, W, C).
        """
        # Transpose from HWC to CHW for processing
        image = image.permute(2, 0, 1)
        c, h, w = image.shape

        # Calculate scale ratio
        r = min(self.new_shape[0] / h, self.new_shape[1] / w)
        if not self.scaleup:
            r = min(r, 1.0)

        # Calculate new unpadded size
        new_unpad = (int(round(w * r)), int(round(h * r)))
        
        # Resize if needed
        if (h, w) != (int(round(h * r)), int(round(w * r))):
            image = F.interpolate(
                image.unsqueeze(0), 
                size=(int(round(h * r)), int(round(w * r))), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)

        # Calculate padding (auto mode with stride alignment)
        dw = self.new_shape[1] - new_unpad[0]
        dh = self.new_shape[0] - new_unpad[1]
        
        # Apply stride alignment (auto mode)
        dw = dw % self.stride
        dh = dh % self.stride
        
        # Center the padding
        dw /= 2
        dh /= 2
        
        # Calculate padding values
        top = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left = int(round(dw - 0.1))
        right = int(round(dw + 0.1))
        
        # Apply padding
        if top > 0 or bottom > 0 or left > 0 or right > 0:
            image = F.pad(image, (left, right, top, bottom), value=self.padding_value)
        
        # Transpose back from CHW to HWC
        return image.permute(1, 2, 0)

class MosaicDetectionModel:
    def __init__(self, model_path: str, device, imgsz=640, **kwargs):
        yolo_model = YOLO(model_path)
        assert yolo_model.task == 'segment'
        self.stride = 32
        self.imgsz = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        self.letterbox = OptimizedLetterBox(
            self.imgsz,
            scaleup=True,
            stride=self.stride,
            padding_value=0.447  # 114/255 for normalized images
        )

        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict", "device": device}
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
        self.model = self.model# torch.compile(self.model, mode="reduce-overhead")
        self.device = self.model.device
        self.args.half = self.model.fp16
        self.model.eval()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

        self.is_segmentation_model = yolo_model.task == 'segment'

    def preprocess(self, imgs: List[torch.Tensor]):
        im = torch.stack([self.letterbox(image=x.float().div_(255)) for x in imgs], 0)
        return im.permute(0, 3, 1, 2)  # BCHW

    def inference(self, image_batch: torch.Tensor):
        return self.model(image_batch, augment=False, visualize=False, embed=None)

    def postprocess(self, preds, img, orig_imgs):
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

    def construct_result(self, preds: torch.tensor, img: torch.tensor, orig_img: list[torch.tensor], proto: torch.tensor):
        if not len(preds):  # save empty boxes
            masks = None
        else:
            masks = ops.process_mask(proto, preds[:, 6:], preds[:, :4], img.shape[2:], upsample=True)  # HWC
            preds[:, :4] = ops.scale_boxes(img.shape[2:], preds[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            preds, masks = preds[keep], masks[keep]
        return Results(orig_img, path='', names=self.model.names, boxes=preds[:, :6], masks=masks)