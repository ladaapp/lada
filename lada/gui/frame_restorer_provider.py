# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
from dataclasses import dataclass

from lada import LOG_LEVEL
from lada.utils import VideoMetadata

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)
from lada import RESTORATION_MODEL_NAMES_TO_FILES, DETECTION_MODEL_NAMES_TO_FILES
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

    def with_mosaic_restoration_model_name(self, mosaic_restoration_model_name) -> 'FrameRestorerOptions':
        return FrameRestorerOptions(mosaic_restoration_model_name, self.mosaic_detection_model_name, self.video_metadata, self.device, self.max_clip_length, self.mosaic_detection, self.passthrough, self.fp16_enabled)

    def with_mosaic_detection_model_name(self, mosaic_detection_model_name) -> 'FrameRestorerOptions':
        return FrameRestorerOptions(self.mosaic_restoration_model_name, mosaic_detection_model_name, self.video_metadata, self.device, self.max_clip_length, self.mosaic_detection, self.passthrough, self.fp16_enabled)

    def with_video_metadata(self, video_metadata) -> 'FrameRestorerOptions':
        return FrameRestorerOptions(self.mosaic_restoration_model_name, self.mosaic_detection_model_name, video_metadata, self.device, self.max_clip_length, self.mosaic_detection, self.passthrough, self.fp16_enabled)

    def with_device(self, device) -> 'FrameRestorerOptions':
        return FrameRestorerOptions(self.mosaic_restoration_model_name, self.mosaic_detection_model_name, self.video_metadata, device, self.max_clip_length, self.mosaic_detection, self.passthrough, self.fp16_enabled)

    def with_max_clip_length(self, max_clip_length) -> 'FrameRestorerOptions':
        return FrameRestorerOptions(self.mosaic_restoration_model_name, self.mosaic_detection_model_name, self.video_metadata, self.device, max_clip_length, self.mosaic_detection, self.passthrough, self.fp16_enabled)

    def with_mosaic_detection(self, mosaic_detection) -> 'FrameRestorerOptions':
        return FrameRestorerOptions(self.mosaic_restoration_model_name, self.mosaic_detection_model_name, self.video_metadata, self.device, self.max_clip_length, mosaic_detection, self.passthrough, self.fp16_enabled)

    def with_passthrough(self, passthrough) -> 'FrameRestorerOptions':
        return FrameRestorerOptions(self.mosaic_restoration_model_name, self.mosaic_detection_model_name, self.video_metadata, self.device, self.max_clip_length, self.mosaic_detection, passthrough, self.fp16_enabled)

    def with_fp16(self, fp16_enabled: bool) -> 'FrameRestorerOptions':
        return FrameRestorerOptions(self.mosaic_restoration_model_name, self.mosaic_detection_model_name, self.video_metadata, self.device, self.max_clip_length, self.mosaic_detection, self.passthrough, fp16_enabled)

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
                logger.info(f"model {self.options.mosaic_restoration_model_name} not found in cache. Loading...")
            if self.models_cache["mosaic_detection_model_name"] != self.options.mosaic_detection_model_name:
                cache_miss = True
                logger.info(f"model {self.options.mosaic_detection_model_name} not found in cache. Loading...")

        if cache_miss:
            self._clear_cache()

            mosaic_restoration_model_path = RESTORATION_MODEL_NAMES_TO_FILES[self.options.mosaic_restoration_model_name]
            mosaic_detection_path = DETECTION_MODEL_NAMES_TO_FILES[self.options.mosaic_detection_model_name]
            mosaic_detection_model, mosaic_restoration_model, mosaic_restoration_model_preferred_pad_mode = load_models(
                torch.device(self.options.device), self.options.mosaic_restoration_model_name, mosaic_restoration_model_path, None,
                mosaic_detection_path, fp16=self.options.fp16_enabled, clip_length=self.options.max_clip_length
            )

            self.models_cache = dict(mosaic_restoration_model_name=self.options.mosaic_restoration_model_name,
                                     mosaic_detection_model_name=self.options.mosaic_detection_model_name,
                                     mosaic_detection_model=mosaic_detection_model,
                                     mosaic_restoration_model=mosaic_restoration_model,
                                     mosaic_restoration_model_preferred_pad_mode=mosaic_restoration_model_preferred_pad_mode)

        return FrameRestorer(self.options.device, self.options.video_metadata.video_file, self.options.max_clip_length,
                             self.options.mosaic_restoration_model_name,
                             self.models_cache["mosaic_detection_model"], self.models_cache["mosaic_restoration_model"],
                             self.models_cache["mosaic_restoration_model_preferred_pad_mode"],
                             mosaic_detection=self.options.mosaic_detection)

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