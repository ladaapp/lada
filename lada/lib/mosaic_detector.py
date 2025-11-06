# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import queue
import threading
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import torch

from ultralytics.engine.results import Results
from lada.lib import VideoMetadata, threading_utils
from lada.lib import image_utils
from lada.lib.box_utils import box_overlap
from lada.lib.mosaic_detection_model import MosaicDetectionModel
from lada.lib.scene_utils import crop_to_box_v3
from lada.lib import video_utils
from lada import LOG_LEVEL
from lada.lib.ultralytics_utils import convert_yolo_box, convert_yolo_mask_tensor
from lada.lib import Box

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)
Image = torch.Tensor # [H, W, C] fp
Images = List[torch.Tensor] # [H, W, C] fp
Mask = torch.Tensor # [H, W] uint8
Masks = List[torch.Tensor] # [H, W] uint8
Boxes = List[Box] # [4] int64

class Scene:
    def __init__(self, file_path: Path, video_meta_data: VideoMetadata):
        self.file_path = file_path
        self.video_meta_data = video_meta_data
        self.frames: Images = []
        self.masks: Masks = []
        self.boxes: Boxes = []
        self.frame_start: int | None = None
        self.frame_end: int | None = None
        self._index: int = 0

    def __len__(self):
        return len(self.frames)

    def add_frame(self, frame_num: int, img: Image, mask: Mask, box: Box):
        if self.frame_start is None:
            self.frame_start = frame_num
            self.frame_end = frame_num
        else:
            assert frame_num == self.frame_end + 1
            self.frame_end = frame_num

        self.frames.append(img)
        self.masks.append(mask)
        self.boxes.append(box)

    def merge_mask_box(self, mask: Mask, box: Box):
        assert self.belongs(box)
        current_box = self.boxes[-1]
        t = min(current_box[0], box[0])
        l = min(current_box[1], box[1])
        b = max(current_box[2], box[2])
        r = max(current_box[3], box[3])
        new_box = (t, l, b, r)
        self.boxes[-1] = new_box
        self.masks[-1] = torch.maximum(self.masks[-1], mask)

    def belongs(self, box: Box):
        if len(self.boxes) == 0:
            return False
        last_scene_box = self.boxes[-1]
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
        self.frames: Images = []
        self.masks: Masks = []
        self.boxes: Boxes = []
        self.crop_shapes: List[Tuple[int, int]] = []
        self.pad_after_resizes: List[Tuple[int, int, int, int]] = []
        self._index: int = 0

        # crop scene
        for i in range(len(scene)):
            img, mask, box = scene.frames[i], scene.masks[i], scene.boxes[i]
            cropped_img, cropped_mask, cropped_box, _ = crop_to_box_v3(box, img, mask, (size, size), max_box_expansion_factor=1., border_size=0.06)
            self.frames.append(cropped_img)
            self.masks.append(cropped_mask)
            self.boxes.append(cropped_box)
            self.crop_shapes.append(cropped_img.shape)

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
    def __init__(self, model: MosaicDetectionModel, video_file, frame_detection_queue: queue.Queue, mosaic_clip_queue: queue.Queue, max_clip_length=30, clip_size=256, device=None, pad_mode='reflect', batch_size=4):
        self.model = model
        self.video_file = video_file
        self.device = torch.device(device) if device is not None else device
        self.max_clip_length = max_clip_length
        assert max_clip_length > 0
        self.clip_size = clip_size
        self.pad_mode = pad_mode
        self.clip_counter = 0
        self.start_ns = 0
        self.start_frame = 0
        self.video_meta_data = video_utils.get_video_meta_data(self.video_file)
        self.frame_detection_queue = frame_detection_queue
        self.mosaic_clip_queue = mosaic_clip_queue
        self.frame_feeder_queue = queue.Queue(maxsize=8)
        self.inference_queue = queue.Queue(maxsize=8)
        self.frame_detector_thread: threading.Thread | None = None
        self.frame_feeder_thread: threading.Thread | None = None
        self.inference_thread: threading.Thread | None = None
        self.frame_feeder_thread_should_be_running = False
        self.frame_detector_thread_should_be_running = False
        self.inference_worker_thread_should_be_running = False
        self.stop_requested = False
        self.batch_size = batch_size

        self.queue_stats = {}
        self.queue_stats["frame_detection_queue_wait_time_put"] = 0
        self.queue_stats["frame_detection_queue_max_size"] = 0
        self.queue_stats["mosaic_clip_queue_wait_time_put"] = 0
        self.queue_stats["mosaic_clip_queue_max_size"] = 0
        self.queue_stats["frame_feeder_queue_wait_time_put"] = 0
        self.queue_stats["frame_feeder_queue_wait_time_get"] = 0
        self.queue_stats["frame_feeder_queue_max_size"] = 0
        self.queue_stats["inference_queue_wait_time_put"] = 0
        self.queue_stats["inference_queue_wait_time_get"] = 0
        self.queue_stats["inference_queue_max_size"] = 0

    def start(self, start_ns):
        assert self.frame_feeder_queue.empty()
        assert self.inference_queue.empty()

        self.start_ns = start_ns
        self.start_frame = video_utils.offset_ns_to_frame_num(self.start_ns, self.video_meta_data.video_fps_exact)
        self.stop_requested = False
        self.frame_detector_thread_should_be_running = True
        self.frame_feeder_thread_should_be_running = True
        self.inference_worker_thread_should_be_running = True

        self.frame_detector_thread = threading.Thread(target=self._frame_detector_worker)
        self.frame_detector_thread.start()
        
        self.inference_thread = threading.Thread(target=self._frame_inference_worker)
        self.inference_thread.start()

        self.frame_feeder_thread = threading.Thread(target=self._frame_feeder_worker)
        self.frame_feeder_thread.start()

    def stop(self):
        logger.debug("MosaicDetector: stopping...")
        start = time.time()
        self.stop_requested = True
        self.frame_detector_thread_should_be_running = False
        self.frame_feeder_thread_should_be_running = False

        # unblock producer
        threading_utils.empty_out_queue(self.frame_feeder_queue, "frame_feeder_queue")
        if self.frame_feeder_thread:
            self.frame_feeder_thread.join()
            logger.debug("frame feeder worker: stopped")
        self.frame_feeder_thread = None
        
        # unblock consumer
        threading_utils.put_closing_queue_marker(self.frame_feeder_queue, "frame_feeder_queue")
        # unblock producer
        threading_utils.empty_out_queue(self.inference_queue, "inference_queue")
        if self.inference_thread:
            self.inference_thread.join()
            logger.debug("inference worker: stopped")
        self.inference_thread = None

        # unblock consumer
        threading_utils.put_closing_queue_marker(self.inference_queue, "inference_queue")
        # unblock producer
        clean_up_threads = [
            threading_utils.empty_out_queue_until_producer_is_done(self.mosaic_clip_queue, "mosaic_clip_queue", self.frame_detector_thread),
            threading_utils.empty_out_queue_until_producer_is_done(self.mosaic_clip_queue, "frame_detection_queue", self.frame_detector_thread)]
        if self.frame_detector_thread:
            self.frame_detector_thread.join()
            logger.debug("frame detector worker: stopped")
        for clean_up_thread in clean_up_threads:
            clean_up_thread.join()
        self.frame_detector_thread = None

        # garbage collection
        threading_utils.empty_out_queue(self.frame_feeder_queue, "frame_feeder_queue")
        threading_utils.empty_out_queue(self.inference_queue, "inference_queue")

        assert self.frame_feeder_queue.empty()
        assert self.inference_queue.empty()

        logger.debug(f"MosaicDetector: stopped, took: {time.time() - start}")

    def _create_clips_for_completed_scenes(self, scenes, frame_num, eof):
        completed_scenes = []
        for current_scene in scenes:
            if (current_scene.frame_end < frame_num or len(current_scene) >= self.max_clip_length or eof) and current_scene not in completed_scenes:
                completed_scenes.append(current_scene)
                other_scenes = [other for other in scenes if other != current_scene]
                for other_scene in other_scenes:
                    if other_scene.frame_start < current_scene.frame_start and other_scene not in completed_scenes:
                        completed_scenes.append(other_scene)

        for completed_scene in sorted(completed_scenes, key=lambda s: s.frame_start):
            clip = Clip(completed_scene, self.clip_size, self.pad_mode, self.clip_counter)
            self.queue_stats["mosaic_clip_queue_max_size"] = max(self.mosaic_clip_queue.qsize()+1, self.queue_stats["mosaic_clip_queue_max_size"])
            s = time.time()
            self.mosaic_clip_queue.put(clip)
            self.queue_stats["mosaic_clip_queue_wait_time_put"] += time.time() - s
            if self.stop_requested:
                logger.debug("frame detector worker: mosaic_clip_queue producer unblocked")
                return
            #print(f"frame {frame_num}, yielding clip starting {clip.frame_start}, ending {clip.frame_end}, all scene starts: {[s.frame_start for s in scenes]}, completed scenes: {[s.frame_start for s in completed_scenes]}")
            scenes.remove(completed_scene)
            self.clip_counter += 1

    def _create_or_append_scenes_based_on_prediction_result(self, results: Results, scenes: list[Scene], frame_num):
        mosaic_detected = len(results.boxes) > 0
        self.queue_stats["frame_detection_queue_max_size"] = max(self.frame_detection_queue.qsize()+1, self.queue_stats["frame_detection_queue_max_size"])
        s = time.time()
        self.frame_detection_queue.put((frame_num, mosaic_detected))
        self.queue_stats["frame_detection_queue_wait_time_put"] += time.time() - s
        if self.stop_requested:
            logger.debug("frame detector worker: frame_detection_queue producer unblocked")
            return
        for i in range(len(results.boxes)):
            if self.model.is_segmentation_model:
                mask = convert_yolo_mask_tensor(results.masks[i], results.orig_shape)
            else:
                # TODO: we currently don't use mosaic masks in the restoration pipeline, so we could also remove it
                mask = torch.zeros(results.orig_shape, dtype=torch.uint8, device=self.device)
            box = convert_yolo_box(results.boxes[i], results.orig_shape)

            current_scene = None
            for scene in scenes:
                if scene.belongs(box):
                    if scene.frame_end == frame_num:
                        current_scene = scene
                        current_scene.merge_mask_box(mask, box)
                    else:
                        current_scene = scene
                        current_scene.add_frame(frame_num, results.orig_img, mask, box)
                    break
            if current_scene is None:
                current_scene = Scene(self.video_file, self.video_meta_data)
                scenes.append(current_scene)
                current_scene.add_frame(frame_num, results.orig_img, mask, box)

    def _frame_feeder_worker(self):
        logger.debug("frame feeder: started")
        with video_utils.VideoReader(self.video_file, self.device) as video_reader:
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)
            video_frames_generator = video_reader.frames()
            frame_num = self.start_frame
            eof = False
            while self.frame_feeder_thread_should_be_running:
                try:
                    frames = []
                    for i in range(self.batch_size):
                        frame, _ = next(video_frames_generator)
                        frames.append(frame)
                except StopIteration:
                    eof = True
                    self.frame_feeder_thread_should_be_running = False
                if len(frames) > 0:
                    frames_batch = self.model.preprocess(frames)
                    data = (frames_batch, frames, frame_num)
                    self.queue_stats["frame_feeder_queue_max_size"] = max(self.frame_feeder_queue.qsize()+1, self.queue_stats["frame_feeder_queue_max_size"])
                    s = time.time()
                    self.frame_feeder_queue.put(data)
                    self.queue_stats["frame_feeder_queue_wait_time_put"] += time.time() - s
                    if self.stop_requested:
                        logger.debug("frame feeder worker: frame_feeder_queue producer unblocked")
                        break
                frame_num += len(frames)
                if eof:
                    self.queue_stats["frame_feeder_queue_max_size"] = max(self.frame_feeder_queue.qsize()+1, self.queue_stats["frame_feeder_queue_max_size"])
                    s = time.time()
                    self.frame_feeder_queue.put(None)
                    self.queue_stats["frame_feeder_queue_wait_time_put"] += time.time() - s
                    if self.stop_requested:
                        logger.debug("frame feeder worker: frame_feeder_queue producer unblocked")
            if eof and not self.stop_requested:
                logger.debug("frame feeder worker: stopped itself, EOF")

    def _frame_inference_worker(self):
        logger.debug("frame inference worker: started")
        eof = False
        while self.inference_worker_thread_should_be_running:
            s = time.time()
            frames_data = self.frame_feeder_queue.get()
            self.queue_stats["frame_feeder_queue_wait_time_get"] += time.time() - s
            if self.stop_requested:
                logger.debug("inference worker: frame_feeder_queue consumer unblocked")
            if frames_data is None:
                eof = True
                self.inference_worker_thread_should_be_running = False
                self.queue_stats["inference_queue_max_size"] = max(self.inference_queue.qsize()+1, self.queue_stats["inference_queue_max_size"])
                s = time.time()
                self.inference_queue.put(None)
                self.queue_stats["inference_queue_wait_time_put"] += time.time() -s
                if self.stop_requested:
                    logger.debug("inference worker: inference_queue producer unblocked")
                break
            frames_batch, frames, frame_num = frames_data
            inference_results = self.model.inference(frames_batch) if frames_batch is not None else None
            self.queue_stats["inference_queue_max_size"] = max(self.inference_queue.qsize()+1, self.queue_stats["inference_queue_max_size"])
            s = time.time()
            self.inference_queue.put((inference_results, frames_batch, frames, frame_num))
            self.queue_stats["inference_queue_wait_time_put"] += time.time() - s
            if self.stop_requested:
                logger.debug("inference worker: inference_queue producer unblocked")
        if eof:
            logger.debug("inference worker: stopped itself, EOF")

    def _frame_detector_worker(self):
        logger.debug("frame detector worker: started")
        scenes: list[Scene] = []
        frame_num = self.start_frame
        eof = False
        while self.frame_detector_thread_should_be_running:
            s = time.time()
            inference_data = self.inference_queue.get()
            self.queue_stats["inference_queue_wait_time_get"] += time.time() - s
            if self.stop_requested:
                logger.debug("frame detector worker: inference_queue consumer unblocked")
            if inference_data is None:
                eof = True
            if eof:
                self._create_clips_for_completed_scenes(scenes, frame_num, eof=True)
                self.queue_stats["frame_detection_queue_max_size"] = max(self.frame_detection_queue.qsize()+1, self.queue_stats["frame_detection_queue_max_size"])
                s = time.time()
                self.frame_detection_queue.put(None)
                self.queue_stats["frame_detection_queue_wait_time_put"] += time.time() - s
                if self.stop_requested:
                    logger.debug("frame detector worker: frame_detection_queue producer unblocked")
                self.queue_stats["mosaic_clip_queue_max_size"] = max(self.mosaic_clip_queue.qsize()+1, self.queue_stats["mosaic_clip_queue_max_size"])
                s = time.time()
                self.mosaic_clip_queue.put(None)
                self.queue_stats["mosaic_clip_queue_wait_time_put"] += time.time() - s
                if self.stop_requested:
                    logger.debug("frame detector worker: mosaic_clip_queue producer unblocked")
                self.frame_detector_thread_should_be_running = False
            else:
                inference_results, preprocessed_frames, orig_frames, _frame_num = inference_data
                assert frame_num == _frame_num, "frame detector worker out of sync with frame reader"
                batch_prediction_results = self.model.postprocess(inference_results, preprocessed_frames, orig_frames)
                assert preprocessed_frames.shape[0] == len(batch_prediction_results)
                for i, results in enumerate(batch_prediction_results):
                    self._create_or_append_scenes_based_on_prediction_result(results, scenes, frame_num)
                    self._create_clips_for_completed_scenes(scenes, frame_num, eof=False)
                    frame_num += 1
        if eof:
            logger.debug("frame detector worker: stopped itself, EOF")
