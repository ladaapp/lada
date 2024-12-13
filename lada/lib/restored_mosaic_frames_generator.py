import logging
import queue
import threading
import time
from typing import Optional

import numpy as np
from ultralytics import YOLO

from lada import LOG_LEVEL
from lada.lib import image_utils, video_utils
from lada.lib import visualization
from lada.lib.mosaic_frames_generator import MosaicFramesWorker
from lada.lib.clean_mosaic_utils import clean_cropped_mosaic
from lada.pidinet import pidinet_inference

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

def load_models(device, mosaic_restoration_model_name, mosaic_restoration_model_path, mosaic_restoration_config_path,
                mosaic_detection_model_path, mosaic_cleaning_edge_detection_model_path=None):
    mosaic_edge_detection_model = None
    if mosaic_cleaning_edge_detection_model_path:
        mosaic_edge_detection_model = pidinet_inference.load_model(mosaic_cleaning_edge_detection_model_path, model_type="tiny")

    if mosaic_restoration_model_name.startswith("rvrt"):
        from lada.rvrt import rvrt_inferencer
        mosaic_restoration_model = rvrt_inferencer.get_model(model_path=mosaic_restoration_model_path, device=device)
        pad_mode = 'zero'
    elif mosaic_restoration_model_name.startswith("deepmosaics"):
        from lada.deepmosaics.models import loadmodel, model_util
        mosaic_restoration_model = loadmodel.video(model_util.device_to_gpu_id(device), mosaic_restoration_model_path)
        pad_mode = 'reflect'
    elif mosaic_restoration_model_name.startswith("basicvsrpp"):
        from lada.basicvsrpp.inference import load_model, get_default_gan_inference_config
        if mosaic_restoration_config_path:
            config = mosaic_restoration_config_path
        else:
            config = get_default_gan_inference_config()
        mosaic_restoration_model = load_model(config, mosaic_restoration_model_path, device)
        pad_mode = 'reflect'
    elif mosaic_restoration_model_name.startswith("tecogan"):
        from lada.tecogan.tecogan_inferencer import load_model
        mosaic_restoration_model = load_model(mosaic_restoration_config_path)
        pad_mode = 'reflect'
    else:
        raise NotImplementedError()

    mosaic_detection_model = YOLO(mosaic_detection_model_path)
    return mosaic_detection_model, mosaic_restoration_model, mosaic_edge_detection_model, pad_mode


class FrameRestorer:
    def __init__(self, device, video_file, preserve_relative_scale, max_clip_length, mosaic_restoration_model_name,
                 mosaic_detection_model, mosaic_restoration_model, mosaic_edge_detection_model, preferred_pad_mode,
                 mosaic_detection=False, mosaic_cleaning=False):
        self.device = device
        self.mosaic_restoration_model_name = mosaic_restoration_model_name
        self.max_clip_length = max_clip_length
        self.preserve_relative_scale = preserve_relative_scale
        self.video_meta_data = video_utils.get_video_meta_data(video_file)
        self.mosaic_detection_model = mosaic_detection_model
        self.mosaic_restoration_model = mosaic_restoration_model
        self.mosaic_edge_detection_model = mosaic_edge_detection_model
        self.preferred_pad_mode = preferred_pad_mode
        self.start_ns = 0
        self.start_frame = 0
        self.mosaic_cleaning = mosaic_cleaning
        self.mosaic_detection = mosaic_detection
        self.eof = False
        self.stop_requested = False

        # limit queue size to approx 512MB
        self.frame_restoration_queue = queue.Queue()
        max_frames_in_frame_restoration_queue = (512 * 1024 * 1024) // (self.video_meta_data.video_width * self.video_meta_data.video_height * 3)
        self.frame_restoration_queue = queue.Queue(maxsize=max_frames_in_frame_restoration_queue)

        # limit queue size to approx 256MB
        max_frames_in_frame_restoration_queue = (256 * 1024 * 1024) // (256 * 256 * 4) # 4 = 3 color channels + mask
        self.mosaic_clip_queue = queue.Queue(maxsize=max_frames_in_frame_restoration_queue)

        # limit queue size to approx 256MB
        max_frames_in_frame_restoration_queue = (256 * 1024 * 1024) // (256 * 256 * 4) # 4 = 3 color channels + mask
        self.mosaic_clip_queue = queue.Queue(maxsize=max_frames_in_frame_restoration_queue)
        self.restored_clip_queue = queue.Queue(maxsize=max_frames_in_frame_restoration_queue)

        # no queue size limit needed, elements are tiny
        self.frame_detection_queue = queue.Queue()

        self.mosaic_frames_generator = MosaicFramesWorker(self.mosaic_detection_model, self.video_meta_data.video_file,
                                                          frame_queue=self.frame_detection_queue,
                                                          clip_queue=self.mosaic_clip_queue,
                                                          device=self.device,
                                                         max_clip_length=self.max_clip_length,
                                                         pad_mode=self.preferred_pad_mode,
                                                         preserve_relative_scale=self.preserve_relative_scale,
                                                         dont_preserve_relative_scale=(not self.preserve_relative_scale))

        self.restoration_thread: threading.Thread | None = None
        self.reassembly_thread: threading.Thread | None = None
        self.restoration_thread_should_be_running = False
        self.reassembly_thread_should_be_running = False
        self.stop_requested = False

    def restore_clip_frames(self, images):
        if self.mosaic_restoration_model_name.startswith("rvrt"):
            from lada.rvrt import rvrt_inferencer
            restored_clip_images = rvrt_inferencer.inference(images, self.mosaic_restoration_model)
        elif self.mosaic_restoration_model_name.startswith("deepmosaics"):
            from lada.deepmosaics.inference import restore_video_frames
            from lada.deepmosaics.models import model_util
            restored_clip_images = restore_video_frames(model_util.device_to_gpu_id(self.device), self.mosaic_restoration_model, images)
        elif self.mosaic_restoration_model_name.startswith("basicvsrpp"):
            from lada.basicvsrpp.inference import inference
            restored_clip_images = inference(self.mosaic_restoration_model, images, self.device)
        elif self.mosaic_restoration_model_name.startswith("tecogan"):
            from lada.tecogan.tecogan_inferencer import inference
            restored_clip_images = inference(images, self.mosaic_restoration_model)
        else:
            raise NotImplementedError()
        return restored_clip_images

    def restore_frame(self, frame, frame_num, restored_clips):
        """
        Takes mosaic frame and restored clips and replaces mosaic regions in frame with restored content from the clips starting at the same frame number as mosaic frame.
        Pops starting frame from each restored clip in the process if they actually start at the same frame number as frame.
        """
        for buffered_clip in [c for c in restored_clips if c.frame_start == frame_num]:
            clip_img, clip_mask, orig_clip_box, orig_crop_shape, pad_after_resize, pad_before_resize = buffered_clip.pop()
            clip_img = image_utils.unpad_image(clip_img, pad_after_resize)
            clip_img = image_utils.resize(clip_img, orig_crop_shape[:2])
            clip_img = image_utils.unpad_image(clip_img, pad_before_resize)
            t, l, b, r = orig_clip_box
            frame[t:b + 1, l:r + 1, :] = clip_img

    def restore_clip(self, clip):
        """
        Restores each contained from of the mosaic clip. If self.mosaic_detection is True will instead draw mosaic detection
        boundaries on each frame.
        """
        if self.mosaic_detection:
            restored_clip_images = visualization.draw_mosaic_detections(clip)
        else:
            if self.mosaic_cleaning:
                images = []
                for (
                cropped_img, cropped_mask, cropped_box, orig_crop_shape, pad_after_resize, pad_before_resize) in clip:
                    images.append(clean_cropped_mosaic(cropped_img, cropped_mask, pad_after_resize,
                                                       pidinet_model=self.mosaic_edge_detection_model))
            else:
                images = clip.get_clip_images()

            restored_clip_images = self.restore_clip_frames(images)
        assert len(restored_clip_images) == len(clip.get_clip_images())

        for i in range(len(restored_clip_images)):
            assert clip.data[i][0].shape == restored_clip_images[i].shape
            clip.data[i] = restored_clip_images[i], clip.data[i][1], clip.data[i][2], clip.data[i][3], clip.data[i][4], \
            clip.data[i][5]

    def collect_garbage(self, clip_buffer):
        processed_clips = list(filter(lambda _clip: len(_clip) == 0, clip_buffer))
        for processed_clip in processed_clips:
            clip_buffer.remove(processed_clip)

    def contains_at_least_one_clip_starting_after_frame_num(self, frame_num, clip_buffer):
        return len(clip_buffer) > 0 and frame_num < max(clip_buffer, key=lambda c: c.frame_start).frame_start

    def empty_out_queues(self):
        for queue in (self.mosaic_clip_queue, self.restored_clip_queue, self.frame_restoration_queue, self.frame_detection_queue):
            while not queue.empty():
                queue.get()
                queue.task_done()

    def start(self, start_ns=0):
        self.start_ns = start_ns
        self.start_frame = video_utils.offset_ns_to_frame_num(self.start_ns, self.video_meta_data.video_fps_exact)
        self.reassembly_thread_should_be_running = True
        self.restoration_thread_should_be_running = True
        self.stop_requested = False

        self.reassembly_thread = threading.Thread(target=self._reassembly_worker)
        self.restoration_thread = threading.Thread(target=self._restoration_worker)

        self.mosaic_frames_generator.start(start_ns=start_ns)
        self.restoration_thread.start()
        self.reassembly_thread.start()

    def stop(self):
        start = time.time()
        logger.debug("frame restorer: stopping...")
        self.restoration_thread_should_be_running = False
        self.reassembly_thread_should_be_running = False
        self.stop_requested = True

        self.mosaic_frames_generator.stop()

        if self.restoration_thread:
            self.restoration_thread.join()
            logger.debug("restoration worker: stopped")
        self.restoration_thread = None

        self.restored_clip_queue.put(None)
        self.frame_restoration_queue.put(None)

        # todo: for some reason just putting a single none message isn't enough
        self.restored_clip_queue.put(None)
        self.frame_restoration_queue.put(None)

        if self.reassembly_thread:
            self.reassembly_thread.join()
            logger.debug("reassembly worker: stopped")
        self.reassembly_thread.join()
        self.reassembly_thread = None

        self.empty_out_queues()
        logger.debug(f"frame restorer: stopped. took {time.time() - start}s")

    def _restoration_worker(self):
        logger.debug("restoration_worker: started")
        while self.restoration_thread_should_be_running:
            start = time.time()
            clip = self.mosaic_clip_queue.get()
            waited_for_mosaic_clip = time.time() - start
            if clip is None:
                self.restoration_thread_should_be_running = False
                self.restored_clip_queue.put(None)
            else:
                logger.debug(f"restoration_worker: read mosaic clip, waited for {waited_for_mosaic_clip}s", )
                self.restore_clip(clip)
                self.restored_clip_queue.put(clip)

    def _read_next_frame(self, video_frames_generator, expected_frame_num) -> Optional[tuple[bool, np.ndarray, int]]:
        try:
            frame, frame_pts = next(video_frames_generator)
        except StopIteration:
            elem = self.frame_detection_queue.get()
            assert elem is None, f"Illegal state: Expected to read None (EOF marker) from detection queue but received f{elem}"
            return None
        elem = self.frame_detection_queue.get()
        if self.stop_requested and elem is None:
            return None
        assert elem is not None, "Illegal state: Expected to read detection result from detection queue but received None (EOF marker)"
        detection_frame_num, mosaic_detected = elem
        assert detection_frame_num == expected_frame_num, "frame queue out of sync!"
        return mosaic_detected, frame, frame_pts

    def _read_next_clip(self, current_frame_num, clip_buffer) -> bool:
        start = time.time()
        clip = self.restored_clip_queue.get()
        waited_for_restored_clip = time.time() - start
        if clip is None:
            return False
        assert clip.frame_start >= current_frame_num, "clip queue out of sync!"
        clip_buffer.append(clip)
        logger.debug(f"reassembly worker: read restored clip, waited for {waited_for_restored_clip}s")
        return True

    def _reassembly_worker(self):
        logger.debug("reassembly worker: started")
        with video_utils.VideoReader(self.video_meta_data.video_file) as video_reader:
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)

            video_frames_generator = video_reader.frames()

            frame_num = self.start_frame
            clips_remaining = True
            clip_buffer = []

            while self.reassembly_thread_should_be_running:
                _frame_result = self._read_next_frame(video_frames_generator, frame_num)
                if _frame_result is None:
                    self.eof = True
                    self.reassembly_thread_should_be_running = False
                    break
                else:
                    mosaic_detected, frame, frame_pts = _frame_result
                if mosaic_detected:
                    # As we don't know how many clips starting with the current frame we'll read and buffer restored clips until we receive a clip
                    # that starts after the current frame. This makes sure that we've gather all restored clips necessary to restore the current frame.
                    while clips_remaining and not self.contains_at_least_one_clip_starting_after_frame_num(frame_num, clip_buffer):
                        clips_remaining = self._read_next_clip(frame_num, clip_buffer)

                    self.restore_frame(frame, frame_num, clip_buffer)
                    self.frame_restoration_queue.put((frame, frame_pts))
                    self.collect_garbage(clip_buffer)
                else:
                    self.frame_restoration_queue.put((frame, frame_pts))
                frame_num += 1

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
                try:
                    return self.frame_restoration_queue.get(timeout=0.1)
                except queue.Empty:
                    pass
