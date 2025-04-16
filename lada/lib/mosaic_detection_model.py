import threading

import torch
from ultralytics.utils.checks import check_imgsz
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.engine.results import Results
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics import YOLO
from lada.lib import Image

class MosaicDetectionModel:
    def __init__(self, model_path: str, device, imgsz=640, **kwargs):
        yolo_model = YOLO(model_path)
        assert yolo_model.task == 'segment'
        self.stride = 32
        self.imgsz = check_imgsz(imgsz, stride=self.stride, min_dim=2)
        self.letterbox = LetterBox(
            self.imgsz,
            auto=True,
            stride=self.stride
        )

        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict", "device": device}
        args = {**yolo_model.overrides, **custom, **kwargs}  # highest priority args on the right
        self.args = get_cfg(DEFAULT_CFG, args)

        self.model = AutoBackend(
            weights=yolo_model.model,
            device=torch.device(device),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.half,
            batch=self.args.batch,
            fuse=True,
            verbose=False,
        )
        self.device = self.model.device
        self.args.half = self.model.fp16
        self.model.eval()
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

        self.is_segmentation_model = yolo_model.task == 'segment'
        self._lock = threading.Lock()

    def preprocess(self, imgs):
        im = np.stack([self.letterbox(image=x) for x in imgs])
        im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)
        im = im.to(self.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def inference(self, image_batch: torch.Tensor):
        with self._lock:
            return self.model(image_batch, augment=False, visualize=False, embed=False)

    def postprocess(self, preds, img, orig_imgs):
        protos = preds[1][-1]
        preds = ops.non_max_suppression(
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

    def construct_result(self, preds: torch.tensor, img: torch.tensor, orig_img: list[Image], proto: torch.tensor):
        if not len(preds):  # save empty boxes
            masks = None
        else:
            masks = ops.process_mask(proto, preds[:, 6:], preds[:, :4], img.shape[2:], upsample=True)  # HWC
            preds[:, :4] = ops.scale_boxes(img.shape[2:], preds[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            preds, masks = preds[keep], masks[keep]
        return Results(orig_img, path='', names=self.model.names, boxes=preds[:, :6], masks=masks)
