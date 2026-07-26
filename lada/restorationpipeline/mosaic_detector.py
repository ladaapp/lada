# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import os
import time
from typing import List, Tuple, Callable
from contextlib import nullcontext

import cv2
import numpy as np
import torch

from lada import LOG_LEVEL
from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel
from lada.utils import Box
from lada.utils import VideoMetadata, threading_utils, ImageTensor, MaskTensor, Pad
from lada.utils import image_utils
from lada.utils.perf_utils import StageTimer
from lada.utils import video_utils
from lada.utils.box_utils import box_overlap
from lada.utils.scene_utils import crop_to_box_v3
from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER, PipelineQueue, StopMarker, PipelineThread, ErrorMarker
from lada.utils.ultralytics_utils import convert_yolo_box, convert_yolo_mask_tensor, UltralyticsResults

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

DEFAULT_DETECTION_FRAME_DEVICE_MAX_MB = 4096

class Scene:
    def __init__(self, file_path: str, video_meta_data: VideoMetadata, clip_size: int):
        self.file_path = file_path
        self.video_meta_data = video_meta_data
        self.clip_size = clip_size
        self.frames: list[ImageTensor] = []
        self.masks: list[MaskTensor] = []
        self.boxes: list[Box] = []
        self.tracking_boxes: list[Box] = []
        self.crop_shapes: List[Tuple[int, int]] = []
        self.frame_start: int | None = None
        self.frame_end: int | None = None
        self._index: int = 0

    def __len__(self):
        return len(self.frames)

    @staticmethod
    def _owning_copy(img: ImageTensor | MaskTensor):
        if isinstance(img, torch.Tensor):
            return img.detach().clone().contiguous()
        if isinstance(img, np.ndarray):
            return img.copy()
        raise TypeError(f"Unsupported scene crop type: {type(img)}")

    @staticmethod
    def _merge_crop_masks(base_mask: MaskTensor, base_box: Box, new_mask: MaskTensor, new_box: Box) -> MaskTensor:
        overlap_t = max(base_box[0], new_box[0])
        overlap_l = max(base_box[1], new_box[1])
        overlap_b = min(base_box[2], new_box[2])
        overlap_r = min(base_box[3], new_box[3])
        if overlap_t > overlap_b or overlap_l > overlap_r:
            return new_mask

        base_t = overlap_t - base_box[0]
        base_l = overlap_l - base_box[1]
        base_b = base_t + (overlap_b - overlap_t + 1)
        base_r = base_l + (overlap_r - overlap_l + 1)
        new_t = overlap_t - new_box[0]
        new_l = overlap_l - new_box[1]
        new_b = new_t + (overlap_b - overlap_t + 1)
        new_r = new_l + (overlap_r - overlap_l + 1)

        if isinstance(new_mask, torch.Tensor):
            new_mask[new_t:new_b, new_l:new_r] = torch.maximum(
                new_mask[new_t:new_b, new_l:new_r],
                base_mask[base_t:base_b, base_l:base_r].to(device=new_mask.device),
            )
        else:
            new_mask[new_t:new_b, new_l:new_r] = np.maximum(
                new_mask[new_t:new_b, new_l:new_r],
                base_mask[base_t:base_b, base_l:base_r],
            )
        return new_mask

    def _crop_frame(self, img: ImageTensor, mask: MaskTensor, box: Box):
        cropped_img, cropped_mask, cropped_box, _ = crop_to_box_v3(
            box,
            img,
            mask,
            (self.clip_size, self.clip_size),
            max_box_expansion_factor=1.,
            border_size=0.06,
        )
        cropped_img = self._owning_copy(cropped_img)
        cropped_mask = self._owning_copy(cropped_mask)
        return cropped_img, cropped_mask, cropped_box, tuple(cropped_img.shape)

    def add_frame(self, frame_num: int, img: ImageTensor, mask: MaskTensor, box: Box):
        if self.frame_start is None:
            self.frame_start = frame_num
            self.frame_end = frame_num
        else:
            assert frame_num == self.frame_end + 1
            self.frame_end = frame_num

        cropped_img, cropped_mask, cropped_box, crop_shape = self._crop_frame(img, mask, box)
        self.frames.append(cropped_img)
        self.masks.append(cropped_mask)
        self.boxes.append(cropped_box)
        self.tracking_boxes.append(box)
        self.crop_shapes.append(crop_shape)

    def merge_mask_box(self, img: ImageTensor, mask: MaskTensor, box: Box):
        assert self.belongs(box)
        current_box = self.tracking_boxes[-1]
        t = min(current_box[0], box[0])
        l = min(current_box[1], box[1])
        b = max(current_box[2], box[2])
        r = max(current_box[3], box[3])
        new_box = (t, l, b, r)
        cropped_img, cropped_mask, cropped_box, crop_shape = self._crop_frame(img, mask, new_box)
        cropped_mask = self._merge_crop_masks(self.masks[-1], self.boxes[-1], cropped_mask, cropped_box)
        self.frames[-1] = cropped_img
        self.masks[-1] = cropped_mask
        self.boxes[-1] = cropped_box
        self.tracking_boxes[-1] = new_box
        self.crop_shapes[-1] = crop_shape

    def belongs(self, box: Box):
        if len(self.tracking_boxes) == 0:
            return False
        last_scene_box = self.tracking_boxes[-1]
        return box_overlap(last_scene_box, box)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self):
            item = self.frames[self._index], self.masks[self._index], self.boxes[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration


class Clip:
    def __init__(self, scene: Scene, size, pad_mode, id):
        self.id = id
        self.file_path = scene.file_path
        self.frame_start = scene.frame_start
        self.frame_end = scene.frame_end
        assert self.frame_start <= self.frame_end
        self.size = size
        self.pad_mode = pad_mode
        self.frames: list[ImageTensor] = []
        self.masks: list[MaskTensor] = []
        self.boxes: list[Box] = []
        self.crop_shapes: List[Tuple[int, int]] = []
        self.pad_after_resizes: List[Pad] = []
        self._index: int = 0

        # Scene keeps only owning crop tensors to avoid pinning full video frames in GPU memory.
        for i in range(len(scene)):
            self.frames.append(scene.frames[i])
            self.masks.append(scene.masks[i])
            self.boxes.append(scene.boxes[i])
            self.crop_shapes.append(scene.crop_shapes[i])

        # resize crops to out_size
        max_width, max_height = self.get_max_width_height()
        scale_width, scale_height = size/max_width, size/max_height

        for i, (cropped_img, cropped_mask, cropped_box) in enumerate(zip(self.frames, self.masks, self.boxes)):
            crop_shape = cropped_img.shape

            resize_shape = (int(crop_shape[0] * scale_height), int(crop_shape[1] * scale_width))
            cropped_img = image_utils.resize(cropped_img, resize_shape, interpolation=cv2.INTER_LINEAR)
            cropped_mask = image_utils.resize(cropped_mask, resize_shape, interpolation=cv2.INTER_NEAREST)
            assert cropped_mask.shape[:2] == cropped_img.shape[:2], f"{cropped_mask.shape[:2]}, {cropped_img.shape[:2]}"
            assert cropped_img.shape[0] <= size or cropped_img.shape[1] <= size

            cropped_img, pad_after_resize = image_utils.pad_image(cropped_img, size, size, mode=self.pad_mode)
            cropped_mask, _ = image_utils.pad_image(cropped_mask, size, size, mode='zero')

            self.frames[i] = cropped_img
            self.masks[i] = cropped_mask
            self.boxes[i] = cropped_box
            self.crop_shapes[i] = crop_shape
            self.pad_after_resizes.append(pad_after_resize)

    @classmethod
    def from_clip_range(cls, clip: 'Clip', start_idx: int, end_idx: int, id):
        assert 0 <= start_idx < end_idx <= len(clip.frames)
        segment = cls.__new__(cls)
        segment.id = id
        segment.file_path = clip.file_path
        segment.frame_start = clip.frame_start + start_idx
        segment.frame_end = clip.frame_start + end_idx - 1
        segment.size = clip.size
        segment.pad_mode = clip.pad_mode
        segment.frames = list(clip.frames[start_idx:end_idx])
        segment.masks = list(clip.masks[start_idx:end_idx])
        segment.boxes = list(clip.boxes[start_idx:end_idx])
        segment.crop_shapes = list(clip.crop_shapes[start_idx:end_idx])
        segment.pad_after_resizes = list(clip.pad_after_resizes[start_idx:end_idx])
        segment._index = 0
        return segment

    def get_max_width_height(self):
        max_width = 0
        max_height = 0
        for box in self.boxes:
            t, l, b, r = box
            width, height = r - l + 1, b - t + 1
            if height > max_height:
                max_height = height
            if width > max_width:
                max_width = width
        return max_width, max_height

    def pop(self):
        self.frame_start += 1
        if self.frame_start > self.frame_end:
            self.frame_start = None
            self.frame_end = None

        return self.frames.pop(0), self.masks.pop(0), self.boxes.pop(0), self.crop_shapes.pop(0), self.pad_after_resizes.pop(0)

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self):
            item = self.frames[self._index], self.masks[self._index], self.boxes[self._index], self.crop_shapes[self._index], self.pad_after_resizes[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration

    def __getitem__(self, item):
        return self.frames[item], self.masks[item], self.boxes[item]

class MosaicDetector:
    def __init__(
            self,
            model: Yolo11SegmentationModel,
            video_metadata: VideoMetadata,
            frame_detection_queue: PipelineQueue,
            mosaic_clip_queue: PipelineQueue,
            error_handler: Callable[[ErrorMarker], None],
            max_clip_length=30,
            clip_size=256,
            device: torch.device | None = None,
            pad_mode='reflect',
            batch_size=8,
            frame_input_queue: PipelineQueue | None = None,
            stage_timer: StageTimer | None = None):
        self.model = model
        self.video_meta_data = video_metadata
        self.device = torch.device(device) if device is not None else device
        self.max_clip_length = max_clip_length
        assert max_clip_length > 0
        self.clip_size = clip_size
        self.pad_mode = pad_mode
        self.clip_counter = 0
        self.start_ns = 0
        self.start_frame = 0
        self.frame_detection_queue = frame_detection_queue
        self.mosaic_clip_queue = mosaic_clip_queue
        self.frame_feeder_queue = PipelineQueue(name="frame_feeder_queue", maxsize=8)
        self.inference_queue = PipelineQueue(name="frame_feeder_queue", maxsize=8)
        self.error_handler = error_handler
        self.frame_detector_thread: PipelineThread | None = None
        self.frame_feeder_thread: PipelineThread | None = None
        self.inference_thread: PipelineThread | None = None
        self.stop_requested = False
        self.batch_size = batch_size
        self.frame_input_queue = frame_input_queue
        self.stage_timer = stage_timer
        self.pin_frame_transfers = self._should_pin_frame_transfers()
        self._pin_frame_transfer_warning_logged = False
        self.keep_detection_frames_on_device = self._should_keep_detection_frames_on_device()
        if self.keep_detection_frames_on_device:
            logger.info(
                f"Keeping mosaic detection frames on {self.device} so YOLO preprocessing, masks and clip crops stay on-device"
            )

    def _should_pin_frame_transfers(self) -> bool:
        if self.device is None or self.device.type != "cuda":
            return False
        env_value = os.environ.get("LADA_PIN_FRAME_TRANSFERS", "auto").strip().lower()
        if env_value in ("0", "false", "no", "off", "disable", "disabled"):
            return False
        if env_value in ("", "1", "true", "yes", "on", "enable", "enabled", "auto"):
            return True
        logger.warning("Invalid LADA_PIN_FRAME_TRANSFERS=%s, falling back to auto", env_value)
        return True

    def _should_keep_detection_frames_on_device(self) -> bool:
        if self.device is None or self.device.type not in ("cuda", "xpu"):
            return False

        env_value = os.environ.get("LADA_DETECTION_FRAMES_ON_DEVICE", "auto").strip().lower()
        if env_value in ("0", "false", "no", "off", "disable", "disabled"):
            return False
        if env_value in ("1", "true", "yes", "on", "enable", "enabled"):
            return True
        if env_value not in ("", "auto"):
            logger.warning(
                "Invalid LADA_DETECTION_FRAMES_ON_DEVICE=%s, falling back to auto",
                env_value,
            )

        try:
            max_device_frame_mb = int(os.environ.get(
                "LADA_DETECTION_FRAME_DEVICE_MAX_MB",
                DEFAULT_DETECTION_FRAME_DEVICE_MAX_MB,
            ))
        except ValueError:
            max_device_frame_mb = DEFAULT_DETECTION_FRAME_DEVICE_MAX_MB

        if max_device_frame_mb <= 0:
            return False

        estimated_device_mb = (
            self.batch_size
            * self.video_meta_data.video_width
            * self.video_meta_data.video_height
            * 3
            / (1024 * 1024)
        )
        estimated_device_mb += (
            self.max_clip_length
            * self.clip_size
            * self.clip_size
            * 4
            * 2
            / (1024 * 1024)
        )
        if estimated_device_mb > max_device_frame_mb:
            logger.info(
                "Keeping detection frames on device disabled because active detection tensors may require "
                f"{estimated_device_mb:.0f}MB, above limit {max_device_frame_mb}MB. "
                "Raise LADA_DETECTION_FRAME_DEVICE_MAX_MB or set LADA_DETECTION_FRAMES_ON_DEVICE=1 to force it."
            )
            return False
        return True

    def _move_detection_frames_to_device(self, frames: list[ImageTensor]) -> list[ImageTensor]:
        if not self.keep_detection_frames_on_device:
            return frames
        if len(frames) == 0 or frames[0].device.type == self.device.type:
            return frames
        with self._timed_stage("detection_frames_to_device"):
            device_frames = []
            for frame in frames:
                non_blocking = False
                if self.pin_frame_transfers and frame.device.type == "cpu":
                    try:
                        if not frame.is_pinned():
                            frame = frame.pin_memory()
                        non_blocking = True
                    except Exception as e:
                        if not self._pin_frame_transfer_warning_logged:
                            logger.warning("Unable to pin decoded frames for CUDA transfer: %s", e)
                            self._pin_frame_transfer_warning_logged = True
                device_frames.append(frame.to(device=self.device, non_blocking=non_blocking))
            return device_frames

    def _timed_stage(self, stage: str):
        if self.stage_timer is None:
            return nullcontext()
        return self.stage_timer.measure(stage)

    def start(self, start_ns):
        assert self.frame_feeder_queue.empty()
        assert self.inference_queue.empty()

        self.start_ns = start_ns
        self.start_frame = video_utils.offset_ns_to_frame_num(self.start_ns, self.video_meta_data.video_fps_exact)
        self.stop_requested = False

        self.frame_detector_thread = PipelineThread(name="frame detector worker", target=self._frame_detector_worker, error_handler=self.error_handler)
        self.frame_detector_thread.start()

        self.inference_thread = PipelineThread(name="frame inference worker", target=self._frame_inference_worker, error_handler=self.error_handler)
        self.inference_thread.start()

        self.frame_feeder_thread = PipelineThread(name="frame feeder worker", target=self._frame_feeder_worker, error_handler=self.error_handler)
        self.frame_feeder_thread.start()

    def stop(self):
        logger.debug("MosaicDetector: stopping...")
        start = time.time()
        self.stop_requested = True

        if self.frame_input_queue is not None:
            threading_utils.put_queue_stop_marker(self.frame_input_queue)

        # unblock producer
        threading_utils.empty_out_queue(self.frame_feeder_queue)
        if self.frame_feeder_thread:
            self.frame_feeder_thread.join()
            logger.debug("MosaicDetector: joined frame_feeder_thread")
        self.frame_feeder_thread = None
        
        # unblock consumer
        threading_utils.put_queue_stop_marker(self.frame_feeder_queue)
        # unblock producer
        threading_utils.empty_out_queue(self.inference_queue)
        if self.inference_thread:
            self.inference_thread.join()
            logger.debug("MosaicDetector: joined inference_thread")
        self.inference_thread = None

        # unblock consumer
        threading_utils.put_queue_stop_marker(self.inference_queue)
        # unblock producer
        threading_utils.empty_out_queue(self.mosaic_clip_queue)
        if self.frame_detector_thread:
            self.frame_detector_thread.join()
            logger.debug("MosaicDetector: joined frame_detector_thread")
        self.frame_detector_thread = None

        # garbage collection
        threading_utils.empty_out_queue(self.frame_feeder_queue)
        threading_utils.empty_out_queue(self.inference_queue)

        assert self.frame_feeder_queue.empty()
        assert self.inference_queue.empty()

        logger.debug(f"MosaicDetector: stopped, took: {time.time() - start}")

    def _create_clips_for_completed_scenes(self, scenes, frame_num, eof) -> StopMarker | None:
        completed_scenes = []
        for current_scene in scenes:
            if (current_scene.frame_end < frame_num or len(current_scene) >= self.max_clip_length or eof) and current_scene not in completed_scenes:
                completed_scenes.append(current_scene)
                other_scenes = [other for other in scenes if other != current_scene]
                for other_scene in other_scenes:
                    if other_scene.frame_start < current_scene.frame_start and other_scene not in completed_scenes:
                        completed_scenes.append(other_scene)

        for completed_scene in sorted(completed_scenes, key=lambda s: s.frame_start):
            with self._timed_stage("mosaic_create_clip"):
                clip = Clip(completed_scene, self.clip_size, self.pad_mode, self.clip_counter)
            self.mosaic_clip_queue.put(clip)
            if self.stop_requested:
                logger.debug("frame detector worker: mosaic_clip_queue producer unblocked")
                return STOP_MARKER
            #print(f"frame {frame_num}, yielding clip starting {clip.frame_start}, ending {clip.frame_end}, all scene starts: {[s.frame_start for s in scenes]}, completed scenes: {[s.frame_start for s in completed_scenes]}")
            scenes.remove(completed_scene)
            self.clip_counter += 1
        return None

    def _create_or_append_scenes_based_on_prediction_result(self, results: UltralyticsResults, scenes: list[Scene], frame_num):
        with self._timed_stage("mosaic_update_scenes"):
            for i in range(len(results.boxes)):
                mask = convert_yolo_mask_tensor(results.masks[i], results.orig_shape).to(device=results.orig_img.device)
                box = convert_yolo_box(results.boxes[i], results.orig_shape)

                current_scene = None
                for scene in scenes:
                    if scene.belongs(box):
                        if scene.frame_end == frame_num:
                            current_scene = scene
                            current_scene.merge_mask_box(results.orig_img, mask, box)
                        else:
                            current_scene = scene
                            current_scene.add_frame(frame_num, results.orig_img, mask, box)
                        break
                if current_scene is None:
                    current_scene = Scene(self.video_meta_data.video_file, self.video_meta_data, self.clip_size)
                    scenes.append(current_scene)
                    current_scene.add_frame(frame_num, results.orig_img, mask, box)

    def _frame_feeder_worker(self):
        logger.debug("frame feeder: started")
        eof = False
        if self.frame_input_queue is not None:
            frame_num = self.start_frame
            while not (eof or self.stop_requested):
                frames = []
                for _ in range(self.batch_size):
                    frame_item = self.frame_input_queue.get()
                    if self.stop_requested or frame_item is STOP_MARKER:
                        logger.debug("frame feeder worker: frame_input_queue consumer unblocked")
                        eof = False
                        break
                    if frame_item is EOF_MARKER:
                        eof = True
                        break
                    frame, _frame_pts = frame_item
                    frames.append(frame)
                if len(frames) > 0:
                    frames = self._move_detection_frames_to_device(frames)
                    with self._timed_stage("yolo_preprocess"):
                        frames_batch = self.model.preprocess(frames)
                    data = (frames_batch, frames, frame_num)
                    self.frame_feeder_queue.put(data)
                    if self.stop_requested:
                        logger.debug("frame feeder worker: frame_feeder_queue producer unblocked")
                        break
                frame_num += len(frames)
                if eof:
                    self.frame_feeder_queue.put(EOF_MARKER)
                    if self.stop_requested:
                        logger.debug("frame feeder worker: frame_feeder_queue producer unblocked")
                        break
            if eof:
                logger.debug("frame feeder worker: stopped itself, EOF")
            else:
                logger.debug("frame feeder worker: stopped by request")
            return

        with video_utils.VideoReader(self.video_meta_data.video_file, device=self.device) as video_reader:
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)
            video_frames_generator = video_reader.frames()
            frame_num = self.start_frame
            while not (eof or self.stop_requested):
                try:
                    frames = []
                    for i in range(self.batch_size):
                        frame, _ = next(video_frames_generator)
                        frames.append(frame)
                except StopIteration:
                    eof = True
                if len(frames) > 0:
                    frames = self._move_detection_frames_to_device(frames)
                    with self._timed_stage("yolo_preprocess"):
                        frames_batch = self.model.preprocess(frames)
                    data = (frames_batch, frames, frame_num)
                    self.frame_feeder_queue.put(data)
                    if self.stop_requested:
                        logger.debug("frame feeder worker: frame_feeder_queue producer unblocked")
                        break
                frame_num += len(frames)
                if eof:
                    self.frame_feeder_queue.put(EOF_MARKER)
                    if self.stop_requested:
                        logger.debug("frame feeder worker: frame_feeder_queue producer unblocked")
                        break
        if eof:
            logger.debug("frame feeder worker: stopped itself, EOF")
        else:
            logger.debug("frame feeder worker: stopped by request")

    def _frame_inference_worker(self):
        logger.debug("frame inference worker: started")
        eof = False
        while not (eof or self.stop_requested):
            frames_data = self.frame_feeder_queue.get()
            if self.stop_requested or frames_data is STOP_MARKER:
                logger.debug("inference worker: frame_feeder_queue consumer unblocked")
                break
            if frames_data is EOF_MARKER:
                eof = True
                self.inference_queue.put(EOF_MARKER)
                if self.stop_requested:
                    logger.debug("inference worker: inference_queue producer unblocked")
                    break
                break
            frames_batch, frames, frame_num = frames_data

            with self._timed_stage("yolo_inference_postprocess"):
                batch_prediction_results = self.model.inference_and_postprocess(frames_batch, frames)

            self.inference_queue.put((batch_prediction_results, frames_batch, frame_num))
            if self.stop_requested:
                logger.debug("inference worker: inference_queue producer unblocked")
                break
        if eof:
            logger.debug("inference worker: stopped itself, EOF")
        else:
            logger.debug("inference worker: stopped by request")

    def _frame_detector_worker(self):
        logger.debug("frame detector worker: started")
        scenes: list[Scene] = []
        frame_num = self.start_frame
        eof = False
        while not (eof or self.stop_requested):
            inference_data = self.inference_queue.get()
            if self.stop_requested or inference_data is STOP_MARKER:
                logger.debug("frame detector worker: inference_queue consumer unblocked")
                break
            eof = inference_data is EOF_MARKER
            if eof:
                self._create_clips_for_completed_scenes(scenes, frame_num, eof=True)
                self.frame_detection_queue.put(EOF_MARKER)
                if self.stop_requested:
                    logger.debug("frame detector worker: frame_detection_queue producer unblocked")
                    break
                self.mosaic_clip_queue.put(EOF_MARKER)
                if self.stop_requested:
                    logger.debug("frame detector worker: mosaic_clip_queue producer unblocked")
                    break
            else:
                batch_prediction_results, preprocessed_frames, _frame_num = inference_data
                assert frame_num == _frame_num, "frame detector worker out of sync with frame reader"
                assert len(preprocessed_frames) == len(batch_prediction_results)
                for i, results in enumerate(batch_prediction_results):
                    self._create_or_append_scenes_based_on_prediction_result(results, scenes, frame_num)
                    num_scenes_containing_frame = len([scene for scene in scenes if scene.frame_start <= frame_num <= scene.frame_end])
                    self.frame_detection_queue.put((frame_num, num_scenes_containing_frame))
                    if self.stop_requested:
                        logger.debug("frame detector worker: frame_detection_queue producer unblocked")
                        break
                    queue_marker = self._create_clips_for_completed_scenes(scenes, frame_num, eof=False)
                    if queue_marker is STOP_MARKER:
                        break
                    frame_num += 1
                # Release MPS driver cached memory to prevent unbounded growth
                if self.device is not None and self.device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
        if eof:
            logger.debug("frame detector worker: stopped itself, EOF")
        else:
            logger.debug("frame detector worker: stopped by request")
