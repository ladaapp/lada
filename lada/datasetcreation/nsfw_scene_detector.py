# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import math
import os.path
import pathlib
import queue
import concurrent.futures as concurrent_futures
from dataclasses import dataclass
from typing import Generator, Optional, Dict

import numpy as np
import torch
import ultralytics.models

from lada import LOG_LEVEL
from lada.utils import Mask, Image, Box, VideoMetadata, threading_utils, video_utils
from lada.utils import mask_utils
from lada.utils.scene_utils import crop_to_box_v3
from lada.utils.threading_utils import wait_until_completed
from lada.utils.ultralytics_utils import choose_biggest_detection, convert_yolo_mask, convert_yolo_box
from lada.models.yolo.yolo import Yolo

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@dataclass
class FileProcessingOptions:
    input_dir: str
    output_dir: pathlib.Path
    start_index: int
    stride_length: int
    scene_min_length: int
    scene_max_length: int
    scene_max_memory: int
    random_extend_masks: bool
    skip4k: bool

@dataclass
class NsfwFrame:
    video_metadata: VideoMetadata
    frame_number: int
    last_frame: bool
    frame: Image
    _box: ultralytics.engine.results.Boxes
    _mask: ultralytics.engine.results.Masks
    object_detected: bool = False
    object_id: int = None

    @property
    def mask(self) -> Mask:
        mask = convert_yolo_mask(self._mask, self.frame.shape)
        # TODO: use mask_utils.clean_mask()
        return mask

    @property
    def box(self) -> Box:
        return convert_yolo_box(self._box, self.frame.shape)


class Scene:

    def __init__(self, video_meta_data, id, scene_min_length, scene_max_length):
        self.video_meta_data: VideoMetadata = video_meta_data
        self.id: int = id
        self.data: Optional[list] = None # will be set when complete() is called
        self._tmp_data: list[NsfwFrame] = []
        self.realized = False
        self.frame_start: int | None = None
        self.frame_end: int | None = None
        self._index: int = 0
        self.scene_max_length: int = scene_max_length
        self.scene_min_length: int = scene_min_length

    def __len__(self):
        return len(self.data) if self.data else len(self._tmp_data)

    def min_length_reached(self):
        return len(self) >= self.scene_min_length

    def max_length_reached(self):
        return len(self) >= self.scene_max_length

    def add_frame(self, nsfw_frame: NsfwFrame):
        if self.frame_start is None:
            self.frame_start = nsfw_frame.frame_number
            self.frame_end = nsfw_frame.frame_number
            self._tmp_data.append(nsfw_frame)
        else:
            assert nsfw_frame.frame_number == self.frame_end + 1
            self.frame_end = nsfw_frame.frame_number
            if not self.max_length_reached():
                self._tmp_data.append(nsfw_frame)

    def complete(self):
        worker_count = 6
        def _convert_data_from_yolo(chunk, chunk_idx_start, chunk_idx_exclusive_end):
            for i, nsfw_frame in enumerate(self._tmp_data[chunk_idx_start:chunk_idx_exclusive_end], start=chunk_idx_start):
                chunk.append((nsfw_frame.frame, nsfw_frame.mask, nsfw_frame.box))

        with concurrent_futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            chunk_indices = list(np.linspace(0, len(self), num=worker_count, dtype=int, endpoint=False))
            futures = []
            chunks = []
            for j, chunk_idx_start in enumerate(chunk_indices):
                chunk_idx_exclusive_end = chunk_indices[j+1] if chunk_idx_start != chunk_indices[-1] else len(self)
                chunk = []
                chunks.append(chunk)
                futures.append(executor.submit(_convert_data_from_yolo, chunk, chunk_idx_start, chunk_idx_exclusive_end))
            wait_until_completed(futures)
            self.data = []
            for chunk in chunks:
                self.data.extend(chunk)
            assert len(self.data) == len(self._tmp_data)
            self._tmp_data = None

    def get_images(self) -> list[Image]:
        return [img for img, _, _ in self.data]

    def get_masks(self) -> list[Mask]:
        return [mask for _, mask, _ in self.data]

    def get_boxes(self) -> list[Box]:
        return [box for _, _, box in self.data]

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Image, Mask, Box]:
        if self._index < len(self):
            item = self.data[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration

    def __getitem__(self, item) -> tuple[Image, Mask, Box]:
        return self.data[item]


class CroppedScene:

    def __init__(self, scene: Scene, window_in_seconds=1.0, target_size=(400,400), smoothing=True, border_size=0):
        self.video_meta_data: VideoMetadata = scene.video_meta_data
        self.id: int = scene.id
        self.data: list = []
        self._index: int = 0

        if smoothing:
            smoothed_boxes = SmoothSceneBoxes.smooth_boxes(scene, window_in_seconds, smooth_function='median')
        else:
            smoothed_boxes = scene.get_boxes()
        scene_images = scene.get_images()
        scene_mask_images = scene.get_masks()

        for i, smoothed_box in enumerate(smoothed_boxes):
            cropped_image, cropped_mask_image, cropped_box, scale_factor = crop_to_box_v3(smoothed_box, scene_images[i],
                                                                         scene_mask_images[i], target_size, border_size=border_size)
            self.data.append((cropped_image, cropped_mask_image, cropped_box))

    def __len__(self):
        return len(self.data)

    def get_images(self) -> list[Image]:
        return [img for img, _, _ in self.data]

    def get_masks(self) -> list[Mask]:
        return [mask for _, mask, _ in self.data]

    def get_boxes(self) -> list[Box]:
        """
        Location of cropped area in original image
        """
        return [box for _, _, box in self.data]

    def get_max_width_height(self):
        max_width = 0
        max_height = 0
        for _, _, box in self.data:
            t, l, b, r = box
            width, height = r - l + 1, b - t + 1
            if height > max_height:
                max_height = height
            if width > max_width:
                max_width = width
        return max_width, max_height

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Image, Mask, Box]:
        if self._index < len(self):
            item = self.data[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration

    def __getitem__(self, item) -> tuple[Image, Mask, Box]:
        return self.data[item]


class SmoothSceneBoxes:

    @staticmethod
    def median_filter(data, window=11):
        assert window % 2 != 0
        pad_size = int((window - 1) / 2)
        data_size = len(data)
        data_type = data[0]

        padded_data = np.pad(data, pad_size, 'edge')
        filtered_data = np.zeros(data_size, dtype=data_type)
        for i in range(data_size):
            filtered_data[i] = np.median(padded_data[i:i + window])
        return filtered_data

    @staticmethod
    def mean_filter(data, window=11):
        assert window % 2 != 0
        pad_size = int((window - 1) / 2)
        data_size = len(data)
        data_type = data[0]
        padded_data = np.pad(data, pad_size, 'edge')
        filtered_data = np.zeros(data_size, dtype=data_type)
        for i in range(data_size):
            filtered_data[i] = np.mean(padded_data[i:i + window])
        return filtered_data

    @staticmethod
    def min_max_filter(data, window, mode):
        assert window % 2 != 0
        func = np.max if mode == 'max' else np.min
        pad_size = int((window - 1) / 2)
        data_size = len(data)
        padded_data = np.pad(data, pad_size, 'edge')
        filtered_data = np.zeros_like(data)
        for i in range(data_size):
            filtered_data[i] = func(padded_data[i:i + window])
        return filtered_data

    @staticmethod
    def smooth_boxes(scene: Scene, window_in_seconds: float, smooth_function='median'):
        _scene_boxes = np.array(scene.get_boxes())
        window_in_frames = min(math.ceil(window_in_seconds * scene.video_meta_data.video_fps), len(scene))
        if window_in_frames % 2 == 0:
            window_in_frames -= 1
        if window_in_frames < 1:
            return _scene_boxes.tolist()

        for i, position in zip(range(4),('t','l','b','r')):
            if smooth_function == 'median':
                _scene_boxes[:, i] = SmoothSceneBoxes.median_filter(_scene_boxes[:, i], window_in_frames)
            elif smooth_function == 'min_max':
                _scene_boxes[:, i] = SmoothSceneBoxes.min_max_filter(_scene_boxes[:, i], window_in_frames, 'max' if position in ('b', 'r') else 'min')
            elif smooth_function == 'mean':
                _scene_boxes[:, i] = SmoothSceneBoxes.mean_filter(_scene_boxes[:, i], window_in_frames)
            else:
                raise NotImplementedError()

        return _scene_boxes.tolist()

    @staticmethod
    def smooth_boxes_center_point(scene: Scene, window_in_seconds: float, smooth_function='median'):
        _scene_boxes = np.array(scene.get_boxes())

        window_in_frames = min(math.ceil(window_in_seconds * scene.video_meta_data.video_fps), len(scene))
        if window_in_frames % 2 == 0:
            window_in_frames -= 1

        heights = _scene_boxes[:, 2] - _scene_boxes[:, 0]
        widths = _scene_boxes[:, 3] - _scene_boxes[:, 1]

        half_widths = widths / 2
        half_heights = heights / 2

        if smooth_function == 'median':
            smoothed_heights = SmoothSceneBoxes.median_filter(heights, window_in_frames)
            smoothed_widths = SmoothSceneBoxes.median_filter(widths, window_in_frames)
            center_x = SmoothSceneBoxes.median_filter(_scene_boxes[:, 1] + half_widths, window_in_frames)
            center_y = SmoothSceneBoxes.median_filter(_scene_boxes[:, 0] + half_heights, window_in_frames)
        else:
            raise NotImplementedError()

        half_smoothed_widths = smoothed_widths / 2
        half_smoothed_heights = smoothed_heights / 2

        new_t = np.clip((center_y - half_smoothed_heights).round().astype(np.int64), 0, scene.video_meta_data.video_height)
        new_b = np.clip((center_y + half_smoothed_heights).round().astype(np.int64), 0, scene.video_meta_data.video_height)
        new_l = np.clip((center_x - half_smoothed_widths).round().astype(np.int64), 0, scene.video_meta_data.video_width)
        new_r = np.clip((center_x + half_smoothed_widths).round().astype(np.int64), 0, scene.video_meta_data.video_width)

        smoothed_boxes = np.stack((new_t, new_l, new_b, new_r), axis=-1)
        return smoothed_boxes.tolist()

def determine_max_scene_length(video_metadata: VideoMetadata, limit_seconds: int | None, limit_memory: int | None):
    scene_max_length = None
    if limit_seconds:
        scene_max_length = limit_seconds
    if limit_memory:
        scene_max_length_memory = video_utils.approx_max_length_by_memory_limit(video_metadata, limit_memory)
        scene_max_length = min(scene_max_length, scene_max_length_memory) if scene_max_length else scene_max_length_memory
    return scene_max_length

def apply_random_mask_extensions(scene: Scene):
    value = np.random.choice([0, 0, 1, 1, 2])
    worker_count = 6
    def _apply_random_mask_extensions(chunk_idx_start, chunk_idx_exclusive_end):
        for i, (img, mask, _) in enumerate(scene.data[chunk_idx_start:chunk_idx_exclusive_end], start=chunk_idx_start):
            mask_extended = mask_utils.extend_mask(mask, value)
            box_extended = mask_utils.get_box(mask_extended)
            scene.data[i] = img, mask_extended, box_extended

    with concurrent_futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        chunk_indices = list(np.linspace(0, len(scene), num=worker_count, dtype=int, endpoint=False))
        futures = []
        for j, chunk_idx_start in enumerate(chunk_indices):
            chunk_idx_exclusive_end = chunk_indices[j+1] if chunk_idx_start != chunk_indices[-1] else len(scene)
            futures.append(executor.submit(_apply_random_mask_extensions, chunk_idx_start, chunk_idx_exclusive_end))
        wait_until_completed(futures)


class NsfwDetector:
    def __init__(self, nsfw_detection_model: Yolo, device: str, file_queue: queue.Queue, frame_queue: queue.Queue, scene_queue: queue.Queue, file_processing_options: FileProcessingOptions, random_extend_masks=True):
        self.nsfw_detection_model: Yolo = nsfw_detection_model
        self.device = torch.device(device) if device is not None else device
        self.file_queue: queue.Queue = file_queue
        self.frame_queue: queue.Queue = frame_queue
        self.scene_queue: queue.Queue = scene_queue
        self.file_processing_options = file_processing_options

        self.metadata: Dict[str, VideoMetadata] = {}
        self.previous_completed_scene_frame_end: Dict[str, Optional[int]] = {}
        self.scenes_counter: Dict[str, int] = {}
        self.random_extend_masks = random_extend_masks

        self.stop_requested = False
        self.thread_pool = concurrent_futures.ThreadPoolExecutor()
        self.frame_detector_thread_futures: list[concurrent_futures.Future] = []
        self.scene_detector_thread_futures: list[concurrent_futures.Future] = []
        # todo: frame thread is faster than scene thread so ideally we could scale it up to multiple consumers. Needs some refactoring first to preserve order of frames
        #  also, more frame threads (processing more than a single file) could be an improvement when NSFW detection becomes a bottleneck when running dataset creation script.
        self.frame_detector_thread_count = 1
        self.scene_detector_thread_count = 1
        self.frame_detector_thread_should_be_running = False
        self.scene_detector_thread_should_be_running = False

        self.no_nsfw_scenes_found_file: pathlib.Path = file_processing_options.output_dir.joinpath("no_nsfw_scenes.txt")
        self.done_processing_file: pathlib.Path = file_processing_options.output_dir.joinpath("done_processing.txt")
        self.files_already_processed: set[str] = set([])
        for file in (self.no_nsfw_scenes_found_file, self.done_processing_file):
            if file.exists():
                with open(file, 'r', encoding='utf-8') as f:
                    for file_path in f:
                        self.files_already_processed.add(file_path.strip())

    def _mark_file_as_processed(self, text_file: pathlib.Path, path_to_save: str):
        if not text_file.exists():
            text_file.parent.mkdir(parents=True, exist_ok=True)
            text_file.touch()
        with open(text_file, 'a', encoding='utf-8') as f:
            f.write(f"{path_to_save}\n")

    def _file_already_processed(self, path_to_check: str):
        return str(path_to_check) in self.files_already_processed and os.path.exists(path_to_check)

    def _process_completed_scene(self, completed_scene: Scene) -> Optional[Scene]:
        """returns Scene if it fits the criteria for a valid completed scene like min/max length"""
        # todo: implementing video strides / discarding frames here is inefficient as we still pass every frame through YOLO / nsfw frames generator.
        #  We'd need to read and pass only the frames we want to be processed to YOLO instead of passing the video file.
        video_file = completed_scene.video_meta_data.video_file
        skip_scene = not (completed_scene.min_length_reached() and (self.previous_completed_scene_frame_end[video_file] is None or (completed_scene.frame_start - self.previous_completed_scene_frame_end[video_file]) > self.stride_length_frames))
        if skip_scene:
            return None
        completed_scene.complete()
        self.scenes_counter[video_file] += 1
        completed_scene.id = self.scenes_counter[video_file]
        self.previous_completed_scene_frame_end[video_file] = completed_scene.frame_end
        if self.random_extend_masks:
            apply_random_mask_extensions(completed_scene)
        return completed_scene

    def _init_new_file(self, metadata: VideoMetadata):
        file_path = metadata.video_file
        self.metadata[file_path] = metadata
        self.scene_min_length =  math.ceil(self.file_processing_options.scene_min_length * metadata.video_fps)
        scene_max_length = determine_max_scene_length (metadata, self.file_processing_options.scene_max_length, self.file_processing_options.scene_max_memory)
        self.scene_max_length = math.ceil(scene_max_length * metadata.video_fps)
        self.stride_length_frames = math.ceil(self.file_processing_options.stride_length * metadata.video_fps)
        self.previous_completed_scene_frame_end[file_path] = None
        self.scenes_counter[file_path] = 0

    def _check_file(self, file_index: 0, file_path: str) -> Optional[VideoMetadata]:
        file_name = pathlib.Path(file_path)
        if file_index < self.file_processing_options.start_index or self._file_already_processed(file_path):
            print(f"{file_index}, Skipping {file_name}: Already processed")
            return None
        if not video_utils.is_video_file(file_path):
            print(f"{file_index}, Skipping {file_name}: Unsupported file format")
            return None
        video_metadata = video_utils.get_video_meta_data(file_path)
        if max(video_metadata.video_width, video_metadata.video_height) > 2_000:
            print(f"{file_index}, Skipping {file_name}: 4K")
            return None
        scene_max_length = determine_max_scene_length (video_metadata, self.file_processing_options.scene_max_length, self.file_processing_options.scene_max_memory)
        if scene_max_length < self.file_processing_options.scene_min_length:
            print(f"{file_index}, Skipping {file_name}: Scene maximum length is less than minimum length")
            return None
        return video_metadata

    def add_files(self, video_files):
        for file_index, file_path in enumerate(video_files):
            self.file_queue.put((file_index, file_path))
        self.file_queue.put(None)

    def _frame_detector_worker(self):
        logger.debug("NsfwDetector: frame detector worker: started")
        while self.frame_detector_thread_should_be_running:
            item: tuple[int, str] | None = self.file_queue.get()
            if self.stop_requested:
                logger.debug("NsfwDetector: frame detector worker: file_queue consumer unblocked")
            if self.stop_requested:
                break
            if not item:
                self.frame_queue.put(None)
                break
            video_file_index, video_file_path = item
            video_metadata = self._check_file(video_file_index, video_file_path)
            if not video_metadata:
                continue
            if video_file_path not in self.metadata:
                self._init_new_file(video_metadata)
            nsfw_frame = None
            print(f"{video_file_index}, Processing {pathlib.Path(video_file_path).name}")
            for frame_num, results in enumerate(self.nsfw_detection_model.track(source=video_metadata.video_file, stream=True, verbose=False, tracker="bytetrack.yaml", device=self.device)):
                if nsfw_frame:
                    self.frame_queue.put(nsfw_frame)
                    if self.stop_requested:
                        logger.debug("NsfwDetector: frame detector worker: frame_queue producer unblocked")
                    if self.stop_requested:
                        break
                yolo_box, yolo_mask = choose_biggest_detection(results, tracking_mode=True)
                object_detected = yolo_box is not None
                nsfw_frame = NsfwFrame(video_metadata, frame_num, False, results.orig_img, yolo_box, yolo_mask, object_detected, int(yolo_box.id.item()) if object_detected else None)
            if nsfw_frame and not self.stop_requested:
                nsfw_frame.last_frame = True
                self.frame_queue.put(nsfw_frame)
                if self.stop_requested:
                    logger.debug("NsfwDetector: frame detector worker: frame_queue producer unblocked")

    def _scene_detector_worker(self):
        logger.debug("NsfwDetector: scene detector worker: started")

        scene: Scene | None = None
        nsfw_frame: NsfwFrame
        previous_file: str = None
        previous_file_no_completed_scenes = True

        while self.scene_detector_thread_should_be_running:
            nsfw_frame: NsfwFrame | None = self.frame_queue.get()
            if self.stop_requested:
                logger.debug("NsfwDetector: scene detector worker: frame_queue consumer unblocked")
            if self.stop_requested:
                break
            if not nsfw_frame:
                self.scene_queue.put(None)
                if self.stop_requested:
                    logger.debug("NsfwDetector: frame detector worker: scene_queue producer unblocked")
                break

            if not previous_file:
                previous_file = nsfw_frame.video_metadata.video_file
            new_file = previous_file != nsfw_frame.video_metadata.video_file
            if new_file:
                if previous_file_no_completed_scenes:
                    self._mark_file_as_processed(self.no_nsfw_scenes_found_file, previous_file)
                previous_file = nsfw_frame.video_metadata.video_file
                previous_file_no_completed_scenes = True

            if nsfw_frame.object_detected:
                if scene is None:
                    scene = Scene(nsfw_frame.video_metadata, nsfw_frame.object_id, self.scene_min_length, self.scene_max_length)
                    scene.add_frame(nsfw_frame)
                else:
                    if scene.id == nsfw_frame.object_id and scene.frame_end + 1 == nsfw_frame.frame_number:
                        scene.add_frame(nsfw_frame)
                    else:
                        completed_scene = self._process_completed_scene(scene)
                        if completed_scene:
                            previous_file_no_completed_scenes = False
                            self.scene_queue.put(completed_scene)
                            if self.stop_requested:
                                logger.debug("NsfwDetector: frame detector worker: scene_queue producer unblocked")
                        scene = Scene(nsfw_frame.video_metadata, nsfw_frame.object_id, self.scene_min_length, self.scene_max_length)
                        scene.add_frame(nsfw_frame)

            if scene is not None and (nsfw_frame.last_frame or not nsfw_frame.object_detected):
                completed_scene = self._process_completed_scene(scene)
                if completed_scene and not self.stop_requested:
                    previous_file_no_completed_scenes = False
                    self.scene_queue.put(completed_scene)
                    if self.stop_requested:
                        logger.debug("NsfwDetector: frame detector worker: scene_queue producer unblocked")
                scene = None

            if nsfw_frame.last_frame:
                self._mark_file_as_processed(self.done_processing_file, nsfw_frame.video_metadata.video_file)

    def __call__(self) -> Generator[Scene, None, None]:
        while not self.stop_requested:
            elem = self.scene_queue.get()
            if self.stop_requested:
                logger.debug("scene_queue consumer unblocked")
            if elem is None and not self.stop_requested:
                self.stop()
                break
            yield elem

    def start(self):
        self.stop_requested = False
        self.frame_detector_thread_should_be_running = True
        self.scene_detector_thread_should_be_running = True

        for i in range(self.frame_detector_thread_count):
            self.frame_detector_thread_futures.append(self.thread_pool.submit(self._frame_detector_worker))
        for i in range(self.scene_detector_thread_count):
            self.scene_detector_thread_futures.append(self.thread_pool.submit(self._scene_detector_worker))

    def stop(self):
        logger.debug("NsfwDetector: stopping...")
        self.stop_requested = True
        self.frame_detector_thread_should_be_running = False
        self.scene_detector_thread_should_be_running = False

        # unblock consumer
        for i in range(self.frame_detector_thread_count): threading_utils.put_closing_queue_marker(self.frame_queue, "file_queue")
        # unblock producer
        threading_utils.empty_out_queue_until_futures_are_done(self.scene_queue, "frame_queue", self.frame_detector_thread_futures)
        concurrent_futures.wait(self.frame_detector_thread_futures, return_when=concurrent_futures.ALL_COMPLETED)
        logger.debug("NsfwDetector: frame detector worker: stopped")
        self.frame_detector_thread_futures = []

        # unblock consumer
        threading_utils.put_closing_queue_marker(self.scene_queue, "scene_queue")
        for i in range(self.scene_detector_thread_count): threading_utils.put_closing_queue_marker(self.frame_queue, "frame_queue")
        # unblock producer
        threading_utils.empty_out_queue_until_futures_are_done(self.scene_queue, "scene_queue", self.scene_detector_thread_futures)
        wait_until_completed(self.scene_detector_thread_futures)
        concurrent_futures.wait(self.scene_detector_thread_futures, return_when=concurrent_futures.ALL_COMPLETED)
        logger.debug("NsfwDetector: scene detector worker: stopped")
        self.scene_detector_thread_futures = []

        # garbage collection
        threading_utils.empty_out_queue(self.file_queue, "file_queue")
        threading_utils.empty_out_queue(self.file_queue, "frame_queue")
        threading_utils.empty_out_queue(self.scene_queue, "scene_queue")

        logger.debug(f"NsfwDetector: stopped")