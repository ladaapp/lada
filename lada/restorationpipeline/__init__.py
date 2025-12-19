import logging

import torch

from lada import LOG_LEVEL, ModelFiles
from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

def load_models(
    device: torch.device,
    mosaic_restoration_model_name: str,
    mosaic_restoration_model_path: str,
    mosaic_restoration_config_path: str | None,
    mosaic_detection_model_path: str,
    fp16: bool,
    detect_face_mosaics: bool):
    if mosaic_restoration_model_name.startswith("deepmosaics"):
        from lada.models.deepmosaics.models import loadmodel
        from lada.restorationpipeline.deepmosaics_mosaic_restorer import DeepmosaicsMosaicRestorer
        _model = loadmodel.video(device, mosaic_restoration_model_path, fp16)
        mosaic_restoration_model = DeepmosaicsMosaicRestorer(_model, device)
        pad_mode = 'reflect'
    elif mosaic_restoration_model_name.startswith("basicvsrpp"):
        from lada.models.basicvsrpp.inference import load_model
        from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
        _model = load_model(mosaic_restoration_config_path, mosaic_restoration_model_path, device, fp16)
        mosaic_restoration_model = BasicvsrppMosaicRestorer(_model, device, fp16)
        pad_mode = 'zero'
    else:
        raise NotImplementedError()
    # setting classes=[0] will consider only detections of class id = 0 (nsfw mosaics) therefore filtering out sfw mosaics (heads, faces)
    if detect_face_mosaics:
        classes = [0]
        detection_model_name = ModelFiles.get_detection_model_by_path(mosaic_detection_model_path)
        if detection_model_name and detection_model_name == "v2":
            logger.info("Mosaic detection model v2 does not support detecting face mosaics. Use detection models v3 or newer. Ignoring...")
    else:
        classes = None
    mosaic_detection_model = Yolo11SegmentationModel(mosaic_detection_model_path, device, classes=classes, conf=0.15, fp16=fp16)
    return mosaic_detection_model, mosaic_restoration_model, pad_mode
