# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import queue
import textwrap
import threading
import time
from typing import Optional

import cv2
import torch
import numpy as np

from lada import LOG_LEVEL
from lada.utils import image_utils, video_utils, threading_utils, mask_utils
from lada.utils import visualization_utils
from lada.restorationpipeline.mosaic_detector import MosaicDetector
from lada.restorationpipeline.mosaic_detector import Clip

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class FrameRestorer:
    def __init__(self, device, video_file, max_clip_length, mosaic_restoration_model_name,
                 mosaic_detection_model, mosaic_restoration_model, preferred_pad_mode,
                 mosaic_detection=False):
        self.device = torch.device(device)
        self.mosaic_restoration_model_name = mosaic_restoration_model_name
        self.max_clip_length = max_clip_length
        self.video_meta_data = video_utils.get_video_meta_data(video_file)
        self.mosaic_detection_model = mosaic_detection_model
        self.mosaic_restoration_model = mosaic_restoration_model
        self.preferred_pad_mode = preferred_pad_mode
        self.start_ns = 0
        self.start_frame = 0
        self.mosaic_detection = mosaic_detection
        self.eof = False
        self.stop_requested = False

        # limit queue size to approx 512MB
        self.frame_restoration_queue = queue.Queue()
        max_frames_in_frame_restoration_queue = (512 * 1024 * 1024) // (self.video_meta_data.video_width * self.video_meta_data.video_height * 3)
        self.frame_restoration_queue = queue.Queue(maxsize=max_frames_in_frame_restoration_queue)

        # limit queue size to approx 512MB
        max_clips_in_mosaic_clips_queue = max(1, (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4)) # 4 = 3 color channels + mask
        logger.debug(f"Set queue size of queue mosaic_clip_queue to {max_clips_in_mosaic_clips_queue}")
        self.mosaic_clip_queue = queue.Queue(maxsize=max_clips_in_mosaic_clips_queue)

        # limit queue size to approx 512MB
        max_clips_in_restored_clips_queue = max(1, (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4)) # 4 = 3 color channels + mask
        logger.debug(f"Set queue size of queue restored_clip_queue to {max_clips_in_restored_clips_queue}")
        self.restored_clip_queue = queue.Queue(maxsize=max_clips_in_restored_clips_queue)

        # no queue size limit needed, elements are tiny
        self.frame_detection_queue = queue.Queue()

        self.mosaic_detector = MosaicDetector(self.mosaic_detection_model, self.video_meta_data.video_file,
                                              frame_detection_queue=self.frame_detection_queue,
                                              mosaic_clip_queue=self.mosaic_clip_queue,
                                              device=self.device,
                                              max_clip_length=self.max_clip_length,
                                              pad_mode=self.preferred_pad_mode)

        self.clip_restoration_thread: threading.Thread | None = None
        self.frame_restoration_thread: threading.Thread | None = None
        self.clip_restoration_thread_should_be_running = False
        self.frame_restoration_thread_should_be_running = False
        self.stop_requested = False

        self.queue_stats = {}
        self.queue_stats["restored_clip_queue_max_size"] = 0
        self.queue_stats["restored_clip_queue_wait_time_put"] = 0
        self.queue_stats["restored_clip_queue_wait_time_get"] = 0
        self.queue_stats["mosaic_clip_queue_wait_time_get"] = 0
        self.queue_stats["frame_restoration_queue_max_size"] = 0
        self.queue_stats["frame_restoration_queue_wait_time_get"] = 0
        self.queue_stats["frame_restoration_queue_wait_time_put"] = 0
        self.queue_stats["frame_detection_queue_wait_time_get"] = 0

    def start(self, start_ns=0):
        assert self.frame_restoration_thread is None and self.clip_restoration_thread is None, "Illegal State: Tried to start FrameRestorer when it's already running. You need to stop it first"
        assert self.mosaic_clip_queue.empty()
        assert self.restored_clip_queue.empty()
        assert self.frame_detection_queue.empty()
        assert self.frame_restoration_queue.empty()

        self.start_ns = start_ns
        self.start_frame = video_utils.offset_ns_to_frame_num(self.start_ns, self.video_meta_data.video_fps_exact)
        self.stop_requested = False
        self.frame_restoration_thread_should_be_running = True
        self.clip_restoration_thread_should_be_running = True

        self.frame_restoration_thread = threading.Thread(target=self._frame_restoration_worker, daemon=True)
        self.clip_restoration_thread = threading.Thread(target=self._clip_restoration_worker, daemon=True)

        self.mosaic_detector.start(start_ns=start_ns)
        self.clip_restoration_thread.start()
        self.frame_restoration_thread.start()

    def stop(self):
        logger.debug("FrameRestorer: stopping...")
        start = time.time()
        self.stop_requested = True
        self.clip_restoration_thread_should_be_running = False
        self.frame_restoration_thread_should_be_running = False

        self.mosaic_detector.stop()

        # unblock consumer
        threading_utils.put_closing_queue_marker(self.mosaic_clip_queue, "mosaic_clip_queue")
        # unblock producer
        threading_utils.empty_out_queue(self.restored_clip_queue, "restored_clip_queue")
        # wait until thread stopped
        if self.clip_restoration_thread:
            self.clip_restoration_thread.join()
            logger.debug("clip restoration worker: stopped")
        self.clip_restoration_thread = None

        # unblock consumer
        threading_utils.put_closing_queue_marker(self.frame_detection_queue, "frame_detection_queue")
        threading_utils.put_closing_queue_marker(self.restored_clip_queue, "restored_clip_queue")
        # unblock producer
        threading_utils.empty_out_queue(self.frame_restoration_queue, "frame_restoration_queue")
        # wait until thread stopped
        if self.frame_restoration_thread:
            self.frame_restoration_thread.join()
            logger.debug("frame restoration worker: stopped")
        self.frame_restoration_thread = None

        # garbage collection
        threading_utils.empty_out_queue(self.mosaic_clip_queue, "mosaic_clip_queue")
        threading_utils.empty_out_queue(self.restored_clip_queue, "restored_clip_queue")
        threading_utils.empty_out_queue(self.frame_detection_queue, "frame_detection_queue")
        threading_utils.empty_out_queue(self.frame_restoration_queue, "frame_restoration_queue")

        assert self.mosaic_clip_queue.empty()
        assert self.restored_clip_queue.empty()
        assert self.frame_detection_queue.empty()
        assert self.frame_restoration_queue.empty()

        logger.debug(f"FrameRestorer: stopped, took {time.time() - start}")

        logger.debug(textwrap.dedent(f"""\
            FrameRestorer: Queue stats:
                frame_restoration_queue/wait-time-get: {self.queue_stats["frame_restoration_queue_wait_time_get"]:.0f}
                frame_restoration_queue/wait-time-put: {self.queue_stats["frame_restoration_queue_wait_time_put"]:.0f}
                frame_restoration_queue/max-qsize: {self.queue_stats["frame_restoration_queue_max_size"]}/{self.frame_restoration_queue.maxsize}
                ---
                mosaic_clip_queue/wait-time-get: {self.queue_stats["mosaic_clip_queue_wait_time_get"]:.0f}
                mosaic_clip_queue/wait-time-put: {self.mosaic_detector.queue_stats["mosaic_clip_queue_wait_time_put"]:.0f}
                mosaic_clip_queue/max-qsize: {self.mosaic_detector.queue_stats["mosaic_clip_queue_max_size"]}/{self.mosaic_clip_queue.maxsize}
                ---
                frame_detection_queue/wait-time-get: {self.queue_stats["frame_detection_queue_wait_time_get"]:.0f}
                frame_detection_queue/wait-time-put: {self.mosaic_detector.queue_stats["frame_detection_queue_wait_time_put"]:.0f}
                frame_detection_queue/max-qsize: {self.mosaic_detector.queue_stats["frame_detection_queue_max_size"]}/{self.frame_detection_queue.maxsize}
                ---
                restored_clip_queue/wait-time-get: {self.queue_stats["restored_clip_queue_wait_time_get"]:.0f}
                restored_clip_queue/wait-time-put: {self.queue_stats["restored_clip_queue_wait_time_put"]:.0f}
                restored_clip_queue/max-qsize: {self.queue_stats["restored_clip_queue_max_size"]}/{self.restored_clip_queue.maxsize}
                ---
                frame_feeder_queue/wait-time-get: {self.mosaic_detector.queue_stats["frame_feeder_queue_wait_time_get"]:.0f}
                frame_feeder_queue/wait-time-put: {self.mosaic_detector.queue_stats["frame_feeder_queue_wait_time_put"]:.0f}
                frame_feeder_queue/max-qsize: {self.mosaic_detector.queue_stats["frame_feeder_queue_max_size"]}/{self.mosaic_detector.frame_feeder_queue.maxsize}"""))


    def _restore_clip_frames(self, images):
        if self.mosaic_restoration_model_name.startswith("deepmosaics"):
            from lada.restorationpipeline.deepmosaics_mosaic_restorer import DeepmosaicsMosaicRestorer
            assert isinstance(self.mosaic_restoration_model, DeepmosaicsMosaicRestorer)
            restored_clip_images = self.mosaic_restoration_model.restore(images)
        elif self.mosaic_restoration_model_name.startswith("basicvsrpp"):
            from lada.restorationpipeline.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
            assert isinstance(self.mosaic_restoration_model, BasicvsrppMosaicRestorer)
            restored_clip_images = self.mosaic_restoration_model.restore(images)
        else:
            raise NotImplementedError()
        return restored_clip_images

    def _restore_frame(self, frame, frame_num, restored_clips):
        """
        Takes mosaic frame and restored clips and replaces mosaic regions in frame with restored content from the clips starting at the same frame number as mosaic frame.
        Pops starting frame from each restored clip in the process if they actually start at the same frame number as frame.
        """
        for buffered_clip in [c for c in restored_clips if c.frame_start == frame_num]:
            clip_img, clip_mask, orig_clip_box, orig_crop_shape, pad_after_resize = buffered_clip.pop()
            clip_img = image_utils.unpad_image(clip_img, pad_after_resize)
            clip_mask = image_utils.unpad_image(clip_mask, pad_after_resize)
            clip_img = image_utils.resize(clip_img, orig_crop_shape[:2])
            clip_mask = image_utils.resize(clip_mask, orig_crop_shape[:2],interpolation=cv2.INTER_NEAREST)
            t, l, b, r = orig_clip_box
            blend_mask = mask_utils.create_blend_mask(clip_mask).to(device=frame.device)

            frame_roi = frame[t:b + 1, l:r + 1, :]
            roi_f = frame_roi.to(dtype=self.mosaic_restoration_model.dtype)
            temp = clip_img.to(dtype=self.mosaic_restoration_model.dtype, device=frame_roi.device)
            temp.sub_(roi_f)
            temp.mul_(blend_mask.unsqueeze(-1))
            temp.add_(roi_f)
            temp.round_().clamp_(0, 255)
            frame_roi[:] = temp

    def _restore_clip(self, clip: Clip):
        """
        Restores each contained from of the mosaic clip. If self.mosaic_detection is True will instead draw mosaic detection
        boundaries on each frame.
        """
        if self.mosaic_detection:
            restored_clip_images = visualization_utils.draw_mosaic_detections(clip)
        else:
            restored_clip_images = self._restore_clip_frames(clip.frames)
        assert len(restored_clip_images) == len(clip.frames)

        for i in range(len(restored_clip_images)):
            assert clip.frames[i].shape == restored_clip_images[i].shape
            clip.frames[i] = restored_clip_images[i]

    def _collect_garbage(self, clip_buffer):
        processed_clips = list(filter(lambda _clip: len(_clip) == 0, clip_buffer))
        has_processed_clips = len(processed_clips) > 0
        for processed_clip in processed_clips:
            clip_buffer.remove(processed_clip)

        if has_processed_clips and self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _contains_at_least_one_clip_starting_after_frame_num(self, frame_num, clip_buffer):
        return len(clip_buffer) > 0 and frame_num < max(clip_buffer, key=lambda c: c.frame_start).frame_start

    def _clip_restoration_worker(self):
        logger.debug("clip restoration worker: started")
        eof = False
        while self.clip_restoration_thread_should_be_running:
            s = time.time()
            clip = self.mosaic_clip_queue.get()
            self.queue_stats["mosaic_clip_queue_wait_time_get"] += time.time() - s
            if self.stop_requested:
                logger.debug("clip restoration worker: mosaic_clip_queue consumer unblocked")
            if clip is None:
                if not self.stop_requested:
                    eof = True
                    self.clip_restoration_thread_should_be_running = False
                    self.queue_stats["restored_clip_queue_max_size"] = max(self.restored_clip_queue.qsize()+1, self.queue_stats["restored_clip_queue_max_size"])
                    s = time.time()
                    self.restored_clip_queue.put(None)
                    self.queue_stats["restored_clip_queue_wait_time_put"] += time.time() -s
                    logger.debug("clip restoration worker: restored_clip_queue producer unblocked")
            else:
                self._restore_clip(clip)
                self.queue_stats["restored_clip_queue_max_size"] = max(self.restored_clip_queue.qsize()+1, self.queue_stats["restored_clip_queue_max_size"])
                s = time.time()
                self.restored_clip_queue.put(clip)
                self.queue_stats["restored_clip_queue_wait_time_put"] += time.time() - s
                if self.stop_requested:
                    logger.debug("clip restoration worker: restored_clip_queue producer unblocked")
        if eof:
            logger.debug("clip restoration worker: stopped itself, EOF")

    def _read_next_frame(self, video_frames_generator, expected_frame_num) -> Optional[tuple[bool, np.ndarray, int]]:
        try:
            frame, frame_pts = next(video_frames_generator)
        except StopIteration:
            s = time.time()
            elem = self.frame_detection_queue.get()
            self.queue_stats["frame_detection_queue_wait_time_get"] += time.time() - s
            if self.stop_requested:
                logger.debug("frame restoration worker: frame_detection_queue consumer unblocked")
            assert elem is None, f"Illegal state: Expected to read None (EOF marker) from detection queue but received f{elem}"
            return None
        s = time.time()
        elem = self.frame_detection_queue.get()
        self.queue_stats["frame_detection_queue_wait_time_get"] += time.time() - s
        if self.stop_requested:
            logger.debug("frame restoration worker: frame_detection_queue consumer unblocked")
            return None
        assert elem is not None, "Illegal state: Expected to read detection result from detection queue but received None (EOF marker)"
        detection_frame_num, mosaic_detected = elem
        assert self.stop_requested or detection_frame_num == expected_frame_num, f"frame detection queue out of sync: received {detection_frame_num} expected {expected_frame_num}"
        return mosaic_detected, frame, frame_pts

    def _read_next_clip(self, current_frame_num, clip_buffer) -> bool:
        s = time.time()
        clip = self.restored_clip_queue.get()
        self.queue_stats["restored_clip_queue_wait_time_get"] += time.time() - s
        if self.stop_requested:
            logger.debug("frame restoration worker: restored_clip_queue consumer unblocked")
        if clip is None:
            return False
        assert self.stop_requested or clip.frame_start >= current_frame_num, "clip queue out of sync!"
        clip_buffer.append(clip)
        return True

    def _frame_restoration_worker(self):
        logger.debug("frame restoration worker: started")
        with video_utils.VideoReader(self.video_meta_data.video_file) as video_reader:
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)

            video_frames_generator = video_reader.frames()

            frame_num = self.start_frame
            clips_remaining = True
            clip_buffer = []

            while self.frame_restoration_thread_should_be_running:
                _frame_result = self._read_next_frame(video_frames_generator, frame_num)
                if _frame_result is None:
                    if not self.stop_requested:
                        self.eof = True
                        self.frame_restoration_thread_should_be_running = False
                        self.frame_restoration_queue.put(None)
                    break
                else:
                    mosaic_detected, frame, frame_pts = _frame_result
                if mosaic_detected:
                    # As we don't know how many clips starting with the current frame we'll read and buffer restored clips until we receive a clip
                    # that starts after the current frame. This makes sure that we've gather all restored clips necessary to restore the current frame.
                    while clips_remaining and not self._contains_at_least_one_clip_starting_after_frame_num(frame_num, clip_buffer):
                        clips_remaining = self._read_next_clip(frame_num, clip_buffer)

                    self._restore_frame(frame, frame_num, clip_buffer)
                    self.queue_stats["frame_restoration_queue_max_size"] = max(self.frame_restoration_queue.qsize()+1, self.queue_stats["frame_restoration_queue_max_size"])
                    s = time.time()
                    self.frame_restoration_queue.put((frame, frame_pts))
                    self.queue_stats["frame_restoration_queue_wait_time_put"] += time.time() -s
                    if self.stop_requested:
                        logger.debug("frame restoration worker: frame_restoration_queue producer unblocked")
                    self._collect_garbage(clip_buffer)
                else:
                    self.queue_stats["frame_restoration_queue_max_size"] = max(self.frame_restoration_queue.qsize()+1, self.queue_stats["frame_restoration_queue_max_size"])
                    s = time.time()
                    self.frame_restoration_queue.put((frame, frame_pts))
                    self.queue_stats["frame_restoration_queue_wait_time_put"] += time.time() - s
                    if self.stop_requested:
                        logger.debug("frame restoration worker: frame_restoration_queue producer unblocked")
                frame_num += 1
            if self.eof:
                logger.debug("frame restoration worker: stopped itself, EOF")

    def __iter__(self):
        return self

    def __next__(self) -> tuple[np.ndarray, int] | None:
        """
        returns None if being called while FrameRestorer is being stopped
        """
        if self.eof and self.frame_restoration_queue.empty():
            raise StopIteration
        else:
            while not self.stop_requested:
                s = time.time()
                elem = self.frame_restoration_queue.get()
                self.queue_stats["frame_restoration_queue_wait_time_get"] += time.time() -s
                if self.stop_requested:
                    logger.debug("frame_restoration_queue consumer unblocked")
                if elem is None and not self.stop_requested:
                    raise StopIteration
                return elem

    def get_frame_restoration_queue(self):
        return self.frame_restoration_queue