# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
from dataclasses import dataclass

from lada import LOG_LEVEL, ModelFiles
from lada.utils import VideoMetadata

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)
from lada.restorationpipeline.frame_restorer import FrameRestorer
from lada.restorationpipeline import load_models
from lada.utils import video_utils

import gc
import torch

@dataclass
class FrameRestorerOptions:
    mosaic_restoration_model_name: str
    mosaic_detection_model_name: str
    video_metadata: VideoMetadata
    device: str
    max_clip_length: int
    mosaic_detection: bool
    passthrough: bool
    fp16_enabled: bool
    detect_face_mosaics: bool

class FrameRestorerOptionsBuilder:
    def __init__(self, initial: FrameRestorerOptions | None = None):
        self._properties = {}
        if initial:
            self._properties = {
                "mosaic_restoration_model_name": initial.mosaic_restoration_model_name,
                "mosaic_detection_model_name": initial.mosaic_detection_model_name,
                "video_metadata": initial.video_metadata,
                "device": initial.device,
                "max_clip_length": initial.max_clip_length,
                "mosaic_detection": initial.mosaic_detection,
                "passthrough": initial.passthrough,
                "fp16_enabled": initial.fp16_enabled,
                "detect_face_mosaics": initial.detect_face_mosaics,
            }

    def mosaic_restoration_model_name(self, value: str) -> 'FrameRestorerOptionsBuilder':
        self._properties["mosaic_restoration_model_name"] = value
        return self

    def mosaic_detection_model_name(self, value: str) -> 'FrameRestorerOptionsBuilder':
        self._properties["mosaic_detection_model_name"] = value
        return self

    def video_metadata(self, value: VideoMetadata) -> 'FrameRestorerOptionsBuilder':
        self._properties["video_metadata"] = value
        return self

    def device(self, value: str) -> 'FrameRestorerOptionsBuilder':
        self._properties["device"] = value
        return self

    def max_clip_length(self, value: int) -> 'FrameRestorerOptionsBuilder':
        self._properties["max_clip_length"] = value
        return self

    def mosaic_detection(self, value: bool) -> 'FrameRestorerOptionsBuilder':
        self._properties["mosaic_detection"] = value
        return self

    def passthrough(self, value: bool) -> 'FrameRestorerOptionsBuilder':
        self._properties["passthrough"] = value
        return self

    def fp16_enabled(self, value: bool) -> 'FrameRestorerOptionsBuilder':
        self._properties["fp16_enabled"] = value
        return self

    def detect_face_mosaics(self, value: bool) -> 'FrameRestorerOptionsBuilder':
        self._properties["mosaic_detection"] = value
        return self

    def build(self) -> FrameRestorerOptions:
        # Check that all properties have been set
        if any(value is None for value in self._properties.values()):
            raise ValueError("All properties must be set before building")
        return FrameRestorerOptions(**self._properties)

class FrameRestorerProvider:
    def __init__(self):
        self.models_cache: None | dict = None
        self.options: FrameRestorerOptions | None = None

    def init(self, options):
        if self.options is not None:
            if self.options.device != options.device:
                self._clear_cache()
        self.options = options

    def get(self):
        assert self.options is not None, "IllegalState: get called but options are not initialized. Call init before using get"
        if self.options.passthrough:
            return PassthroughFrameRestorer(self.options.video_metadata.video_file)

        is_empty_cache = self.models_cache is None
        cache_miss = False
        if is_empty_cache:
            cache_miss = True
        else:
            if self.models_cache["mosaic_restoration_model_name"] != self.options.mosaic_restoration_model_name:
                cache_miss = True
                logger.info(f"model {self.options.mosaic_restoration_model_name} not found in cache. Reloading models...")
            if self.models_cache["mosaic_detection_model_name"] != self.options.mosaic_detection_model_name:
                cache_miss = True
                logger.info(f"model {self.options.mosaic_detection_model_name} not found in cache. Reloading models...")
            if self.models_cache.get("fp16_enabled") != self.options.fp16_enabled:
                cache_miss = True
                logger.info(f"FP16 setting changed from {self.models_cache.get('fp16_enabled')} to {self.options.fp16_enabled}. Reloading models...")
            if self.models_cache.get("detect_face_mosaics") != self.options.detect_face_mosaics:
                cache_miss = True
                logger.info(f"Detect Face Mosaics setting changed from {self.models_cache.get('detect_face_mosaics')} to {self.options.detect_face_mosaics}. Reloading models...")

        if cache_miss:
            self._clear_cache()

            mosaic_restoration_model_path = ModelFiles.get_restoration_model_by_name(self.options.mosaic_restoration_model_name).path
            mosaic_detection_path = ModelFiles.get_detection_model_by_name(self.options.mosaic_detection_model_name).path
            mosaic_detection_model, mosaic_restoration_model, mosaic_restoration_model_preferred_pad_mode = load_models(
                torch.device(self.options.device), self.options.mosaic_restoration_model_name, mosaic_restoration_model_path, None,
                mosaic_detection_path, fp16=self.options.fp16_enabled, detect_face_mosaics=self.options.detect_face_mosaics,
            )

            self.models_cache = dict(mosaic_restoration_model_name=self.options.mosaic_restoration_model_name,
                                     mosaic_detection_model_name=self.options.mosaic_detection_model_name,
                                     fp16_enabled=self.options.fp16_enabled,
                                     mosaic_detection_model=mosaic_detection_model,
                                     mosaic_restoration_model=mosaic_restoration_model,
                                     mosaic_restoration_model_preferred_pad_mode=mosaic_restoration_model_preferred_pad_mode,
                                     detect_face_mosaics=self.options.detect_face_mosaics)

        return FrameRestorer(self.options.device,
                             self.options.video_metadata.video_file,
                             self.options.max_clip_length,
                             self.options.mosaic_restoration_model_name,
                             self.models_cache["mosaic_detection_model"],
                             self.models_cache["mosaic_restoration_model"],
                             self.models_cache["mosaic_restoration_model_preferred_pad_mode"],
                             self.options.mosaic_detection)

    def _clear_cache(self):
        if self.models_cache is None:
            return
        if "mosaic_detection_model" in self.models_cache: del self.models_cache["mosaic_detection_model"]
        if "mosaic_restoration_model" in self.models_cache: del self.models_cache["mosaic_restoration_model"]
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except:
            pass
        self.models_cache = None

class PassthroughFrameRestorer:
    def __init__(self, video_file):
        self.video_file = video_file
        self.video_reader: video_utils.VideoReader | None = None
        self.frame_restoration_queue = None
        self.stopped = False

    def start(self, start_ns=0):
        self.video_reader = video_utils.VideoReader(self.video_file)
        self.video_reader = self.video_reader.__enter__()
        if start_ns >= 0:
            self.video_reader.seek(start_ns)
        self.frame_restoration_queue = PassthroughFrameRestorer.PassthroughQueue(self)

    def stop(self):
        self.stopped = True
        self.video_reader.__exit__(None, None, None)

    def get_frame_restoration_queue(self):
        return self.frame_restoration_queue

    class PassthroughQueue:
        def __init__(self, frame_restorer):
            self.video_frames_generator = frame_restorer.video_reader.frames()
            self.frame_restorer = frame_restorer

        def get(self, block=True, timeout=None):
            if self.frame_restorer.stopped:
                return None
            try:
                return next(self.video_frames_generator)
            except StopIteration:
                return None

        def put(self, item, block=True, timeout=None):
            pass

FRAME_RESTORER_PROVIDER = FrameRestorerProvider()