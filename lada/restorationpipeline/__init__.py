import torch

from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

def load_models(
    device: torch.device,
    mosaic_restoration_model_name: str,
    mosaic_restoration_model_path: str,
    mosaic_restoration_config_path: str | None,
    mosaic_detection_model_path: str,
    fp16: bool,
    clip_length: int):
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
        mosaic_restoration_model = BasicvsrppMosaicRestorer(_model, device, fp16, clip_length)
        pad_mode = 'zero'
    else:
        raise NotImplementedError()
    # setting classes=[0] will consider only for class id = 0 as detections (nsfw mosaics) therefore filtering out sfw mosaics (heads, faces)
    mosaic_detection_model = Yolo11SegmentationModel(mosaic_detection_model_path, device, classes=[0], conf=0.2, fp16=fp16)
    return mosaic_detection_model, mosaic_restoration_model, pad_mode
