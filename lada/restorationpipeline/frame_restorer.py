# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import textwrap
import threading
import time

import cv2
import torch
import numpy as np

from lada import LOG_LEVEL
from lada.utils.threading_utils import EOF_MARKER, STOP_MARKER, StopMarker, EofMarker, PipelineQueue
from lada.utils import image_utils, video_utils, threading_utils, mask_utils, ImageTensor, Image
from lada.utils import visualization_utils
from lada.restorationpipeline.mosaic_detector import MosaicDetector
from lada.restorationpipeline.mosaic_detector import Clip
from lada.models.yolo.yolo11_segmentation_model import Yolo11SegmentationModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class FrameRestorer:
    def __init__(self, device, video_file, max_clip_length, mosaic_restoration_model_name,
                 mosaic_detection_model: Yolo11SegmentationModel, mosaic_restoration_model, preferred_pad_mode,
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
        self.detection_for_next_frame: bool | EofMarker | None = None

        # limit queue size to approx 512MB
        max_frames_in_frame_restoration_queue = (512 * 1024 * 1024) // (self.video_meta_data.video_width * self.video_meta_data.video_height * 3)
        self.frame_restoration_queue = PipelineQueue(name="frame_restoration_queue", maxsize=max_frames_in_frame_restoration_queue)

        # limit queue size to approx 512MB
        max_clips_in_mosaic_clips_queue = max(1, (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4)) # 4 = 3 color channels + mask
        self.mosaic_clip_queue = PipelineQueue(name="mosaic_clip_queue", maxsize=max_clips_in_mosaic_clips_queue)

        # limit queue size to approx 512MB
        max_clips_in_restored_clips_queue = max(1, (512 * 1024 * 1024) // (self.max_clip_length * 256 * 256 * 4)) # 4 = 3 color channels + mask
        self.restored_clip_queue = PipelineQueue(name="restored_clip_queue", maxsize=max_clips_in_restored_clips_queue)

        # no queue size limit needed, elements are tiny
        self.frame_detection_queue = PipelineQueue(name="frame_detection_queue")

        self.mosaic_detector = MosaicDetector(self.mosaic_detection_model, self.video_meta_data,
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
        threading_utils.put_queue_stop_marker(self.mosaic_clip_queue)
        # unblock producer
        threading_utils.empty_out_queue(self.restored_clip_queue)
        # wait until thread stopped
        if self.clip_restoration_thread:
            self.clip_restoration_thread.join()
            logger.debug("FrameRestorer: joined clip_restoration_thread")
        self.clip_restoration_thread = None

        # unblock consumer
        threading_utils.put_queue_stop_marker(self.frame_detection_queue)
        threading_utils.put_queue_stop_marker(self.restored_clip_queue)
        # unblock producer
        threading_utils.empty_out_queue(self.frame_restoration_queue)
        # wait until thread stopped
        if self.frame_restoration_thread:
            self.frame_restoration_thread.join()
            logger.debug("FrameRestorer: joined frame_restoration_thread")
        self.frame_restoration_thread = None

        # garbage collection
        threading_utils.empty_out_queue(self.mosaic_clip_queue)
        threading_utils.empty_out_queue(self.restored_clip_queue)
        threading_utils.empty_out_queue(self.frame_detection_queue)
        threading_utils.empty_out_queue(self.frame_restoration_queue)

        assert self.mosaic_clip_queue.empty()
        assert self.restored_clip_queue.empty()
        assert self.frame_detection_queue.empty()
        assert self.frame_restoration_queue.empty()

        logger.debug(f"FrameRestorer: stopped, took {time.time() - start}")

        logger.debug(textwrap.dedent(f"""\
            FrameRestorer: Queue stats:
                frame_restoration_queue/wait-time-get: {self.frame_restoration_queue.stats[f"{self.frame_restoration_queue.name}_wait_time_get"]:.0f}
                frame_restoration_queue/wait-time-put: {self.frame_restoration_queue.stats[f"{self.frame_restoration_queue.name}_wait_time_put"]:.0f}
                frame_restoration_queue/max-qsize: {self.frame_restoration_queue.stats[f"{self.frame_restoration_queue.name}_max_size"]}/{self.frame_restoration_queue.maxsize}
                ---
                mosaic_clip_queue/wait-time-get: {self.mosaic_clip_queue.stats[f"{self.mosaic_clip_queue.name}_wait_time_get"]:.0f}
                mosaic_clip_queue/wait-time-put: {self.mosaic_clip_queue.stats[f"{self.mosaic_clip_queue.name}_wait_time_put"]:.0f}
                mosaic_clip_queue/max-qsize: {self.mosaic_clip_queue.stats[f"{self.mosaic_clip_queue.name}_max_size"]}/{self.mosaic_clip_queue.maxsize}
                ---
                frame_detection_queue/wait-time-get: {self.frame_detection_queue.stats[f"{self.frame_detection_queue.name}_wait_time_get"]:.0f}
                frame_detection_queue/wait-time-put: {self.frame_detection_queue.stats[f"{self.frame_detection_queue.name}_wait_time_put"]:.0f}
                frame_detection_queue/max-qsize: {self.frame_detection_queue.stats[f"{self.frame_detection_queue.name}_max_size"]}/{self.frame_detection_queue.maxsize}
                ---
                restored_clip_queue/wait-time-get: {self.restored_clip_queue.stats[f"{self.restored_clip_queue.name}_wait_time_get"]:.0f}
                restored_clip_queue/wait-time-put: {self.restored_clip_queue.stats[f"{self.restored_clip_queue.name}_wait_time_put"]:.0f}
                restored_clip_queue/max-qsize: {self.restored_clip_queue.stats[f"{self.restored_clip_queue.name}_max_size"]}/{self.restored_clip_queue.maxsize}
                ---
                frame_feeder_queue/wait-time-get: {self.mosaic_detector.frame_feeder_queue.stats[f"{self.mosaic_detector.frame_feeder_queue.name}_wait_time_get"]:.0f}
                frame_feeder_queue/wait-time-put: {self.mosaic_detector.frame_feeder_queue.stats[f"{self.mosaic_detector.frame_feeder_queue.name}_wait_time_put"]:.0f}
                frame_feeder_queue/max-qsize: {self.mosaic_detector.frame_feeder_queue.stats[f"{self.mosaic_detector.frame_feeder_queue.name}_max_size"]}/{self.mosaic_detector.frame_feeder_queue.maxsize}"""))


    def _restore_clip_frames(self, images: list[ImageTensor]):
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

    def _restore_frame(self, frame: ImageTensor, frame_num: int, restored_clips: list[Clip]):
        """
        Takes mosaic frame and restored clips and replaces mosaic regions in frame with restored content from the clips starting at the same frame number as mosaic frame.
        Pops starting frame from each restored clip in the process if they actually start at the same frame number as frame.
        """
        is_cpu_input = frame.device.type == 'cpu'
        target_dtype = torch.float32 if is_cpu_input else self.mosaic_restoration_model.dtype
        def _blend_gpu(blend_mask: torch.Tensor, clip_img: torch.Tensor, orig_clip_box: tuple[int, int, int, int]):
            t, l, b, r = orig_clip_box
            frame_roi = frame[t:b + 1, l:r + 1, :]
            roi_f = frame_roi.to(dtype=self.mosaic_restoration_model.dtype)
            temp = clip_img.to(dtype=self.mosaic_restoration_model.dtype, device=frame_roi.device)
            temp.sub_(roi_f)
            temp.mul_(blend_mask.unsqueeze(-1))
            temp.add_(roi_f)
            temp.round_().clamp_(0, 255)
            frame_roi[:] = temp

        def _blend_cpu(blend_mask: torch.Tensor, clip_img: torch.Tensor, orig_clip_box: tuple[int, int, int, int]):
            blend_mask = blend_mask.cpu().numpy()
            clip_img = clip_img.cpu().numpy()
            t, l, b, r = orig_clip_box
            frame_roi = frame[t:b + 1, l:r + 1, :].numpy()
            temp_buffer = np.empty_like(frame_roi, dtype=np.float32)
            np.subtract(clip_img, frame_roi, out=temp_buffer, dtype=np.float32)
            np.multiply(temp_buffer, blend_mask[..., None], out=temp_buffer)
            np.add(temp_buffer, frame_roi, out=temp_buffer)
            frame_roi[:] = temp_buffer.astype(np.uint8)
            
        blend = _blend_cpu if is_cpu_input else _blend_gpu

        for buffered_clip in [c for c in restored_clips if c.frame_start == frame_num]:
            clip_img, clip_mask, orig_clip_box, orig_crop_shape, pad_after_resize = buffered_clip.pop()
            clip_img = image_utils.unpad_image(clip_img, pad_after_resize)
            clip_mask = image_utils.unpad_image(clip_mask, pad_after_resize)
            clip_img = image_utils.resize(clip_img, orig_crop_shape[:2])
            clip_mask = image_utils.resize(clip_mask, orig_crop_shape[:2],interpolation=cv2.INTER_NEAREST)
            blend_mask = mask_utils.create_blend_mask(clip_mask.to(device=self.device).to(dtype=self.mosaic_restoration_model.dtype)).to(device=clip_img.device, dtype=target_dtype)

            blend(blend_mask, clip_img, orig_clip_box)

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
            clip = self.mosaic_clip_queue.get()
            if self.stop_requested or clip is STOP_MARKER:
                logger.debug("clip restoration worker: mosaic_clip_queue consumer unblocked")
                break
            if clip is EOF_MARKER:
                eof = True
                self.clip_restoration_thread_should_be_running = False
                self.restored_clip_queue.put(EOF_MARKER)
                if self.stop_requested:
                    logger.debug("clip restoration worker: restored_clip_queue producer unblocked")
                    break
            else:
                assert isinstance(clip, Clip)
                self._restore_clip(clip)
                self.restored_clip_queue.put(clip)
                if self.stop_requested:
                    logger.debug("clip restoration worker: restored_clip_queue producer unblocked")
                    break
        if eof:
            logger.debug("clip restoration worker: stopped itself, EOF")
        else:
            logger.debug("clip restoration worker: stopped by request")

    def _read_next_frame(self, video_frames_generator, expected_frame_num) -> tuple[bool, bool | EofMarker, np.ndarray, int] | StopMarker | EofMarker:
        # Get current frame
        try:
            frame, frame_pts = next(video_frames_generator)
        except StopIteration:
            assert self.detection_for_next_frame is EOF_MARKER, f"Illegal state: We should have received EOF_MARKER in previous call but is {self.detection_for_next_frame}"
            return EOF_MARKER
        # Get detection for current frame (expected_frame_num)
        if self.detection_for_next_frame is None:
            # first frame so we didn't read ahead yet
            elem = self.frame_detection_queue.get()
            if self.stop_requested or elem is STOP_MARKER:
                logger.debug("frame restoration worker: frame_detection_queue consumer unblocked")
                return STOP_MARKER
            detection_frame_num, mosaic_detected = elem
            assert detection_frame_num == expected_frame_num, f"frame detection queue out of sync: received {detection_frame_num} expected {expected_frame_num}"
        else:
            # We already got detection for this frame in the previous call
            detection_frame_num, mosaic_detected = expected_frame_num, self.detection_for_next_frame

        # Get detection for next frame (expected_frame_num + 1)
        elem = self.frame_detection_queue.get()
        if self.stop_requested or elem is STOP_MARKER:
            logger.debug("frame restoration worker: frame_detection_queue consumer unblocked")
            return STOP_MARKER
        if elem is EOF_MARKER:
            self.detection_for_next_frame = EOF_MARKER
        else:
            detection_frame_num_next, mosaic_detected_next = elem
            assert detection_frame_num_next == expected_frame_num + 1, f"frame detection queue out of sync: received {detection_frame_num} expected {expected_frame_num + 1}"
            self.detection_for_next_frame = mosaic_detected_next

        return mosaic_detected, self.detection_for_next_frame, frame, frame_pts

    def _read_next_clip(self, current_frame_num, clip_buffer) -> StopMarker | EofMarker | None:
        clip = self.restored_clip_queue.get()
        if self.stop_requested or clip is STOP_MARKER:
            logger.debug("frame restoration worker: restored_clip_queue consumer unblocked")
            return STOP_MARKER
        if clip is EOF_MARKER:
            return EOF_MARKER
        assert isinstance(clip, Clip)
        assert clip.frame_start >= current_frame_num, "clip queue out of sync!"
        clip_buffer.append(clip)
        return None

    def _frame_restoration_worker(self):
        logger.debug("frame restoration worker: started")
        with video_utils.VideoReader(self.video_meta_data.video_file) as video_reader:
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)

            video_frames_generator = video_reader.frames()

            frame_num = self.start_frame
            queue_marker = None
            clip_buffer = []

            while self.frame_restoration_thread_should_be_running:
                _frame_result = self._read_next_frame(video_frames_generator, frame_num)
                if self.stop_requested or _frame_result is STOP_MARKER:
                    break
                if _frame_result is EOF_MARKER:
                    self.eof = True
                    self.frame_restoration_thread_should_be_running = False
                    self.frame_restoration_queue.put(EOF_MARKER)
                    break
                mosaic_detected, mosaic_detected_in_next_frame, frame, frame_pts = _frame_result
                if mosaic_detected:
                    # Unless we receive an EofMarker or the next frame doesn't contain mosaics we don't know if other clips exist that start with the current frame
                    # so we'll read and buffer restored clips until we receive a clip that starts after the current frame.
                    # This makes sure that we've gather all restored clips necessary to restore the current frame.
                    while queue_marker is not EOF_MARKER and mosaic_detected_in_next_frame is True and not self._contains_at_least_one_clip_starting_after_frame_num(frame_num, clip_buffer):
                        queue_marker = self._read_next_clip(frame_num, clip_buffer)
                    if queue_marker is STOP_MARKER:
                        break

                    self._restore_frame(frame, frame_num, clip_buffer)
                    self.frame_restoration_queue.put((frame, frame_pts))
                    if self.stop_requested:
                        logger.debug("frame restoration worker: frame_restoration_queue producer unblocked")
                        break
                    self._collect_garbage(clip_buffer)
                else:
                    # Nothing to restore, pass through
                    self.frame_restoration_queue.put((frame, frame_pts))
                    if self.stop_requested:
                        logger.debug("frame restoration worker: frame_restoration_queue producer unblocked")
                        break
                frame_num += 1
        if self.eof:
            logger.debug("frame restoration worker: stopped itself, EOF")
        else:
            logger.debug("frame restoration worker: stopped by request")

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Image, int] | None:
        """
        returns None if being called while FrameRestorer is being stopped
        """
        if self.eof and self.frame_restoration_queue.empty():
            raise StopIteration
        else:
            while True:
                elem = self.frame_restoration_queue.get()
                if self.stop_requested or elem is STOP_MARKER:
                    logger.debug("frame_restoration_queue consumer unblocked")
                    break
                if elem is EOF_MARKER:
                    raise StopIteration
                return elem
            return None

    def get_frame_restoration_queue(self) -> PipelineQueue:
        return self.frame_restoration_queue