import logging
import threading
import time

import numpy as np
from gi.repository import Gst, GstApp

from lada import LOG_LEVEL
from lada.gui.frame_restorer_provider import FrameRestorerProvider
from lada.lib import video_utils, VideoMetadata, threading_utils
from lada.lib.frame_restorer import FrameRestorer

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class FrameRestorerAppSrc:
    def __init__(self, video_metadata: VideoMetadata, frame_restorer_provider: FrameRestorerProvider, eos_callback):
        self.video_metadata: VideoMetadata = video_metadata
        self.eos_callback = eos_callback

        self.frame_restorer: FrameRestorer | None = None
        self.frame_restorer_provider: FrameRestorerProvider = frame_restorer_provider
        self.frame_restorer_lock: threading.Lock = threading.Lock()

        self.frame_duration_ns = (1 / self.video_metadata.video_fps) * Gst.SECOND
        self.file_duration_ns = int((self.video_metadata.frames_count * self.frame_duration_ns))

        self.appsource_thread: threading.Thread | None = None
        self.appsource_thread_should_be_running: bool = False
        self.appsource_thread_stop_requested = False

        self._appsrc: GstApp = self._create_appsrc()
        self.appsrc_lock: threading.Lock = threading.Lock()

        self.current_timestamp_ns = 0

    @property
    def appsrc(self):
        return self._appsrc

    def reinit(self, video_metadata: VideoMetadata):
        self.appsrc_lock.acquire()
        self._stop_appsource_worker()
        self.video_metadata = video_metadata
        self.current_timestamp_ns = 0
        self.frame_duration_ns = (1 / self.video_metadata.video_fps) * Gst.SECOND
        self.file_duration_ns = int((self.video_metadata.frames_count * self.frame_duration_ns))
        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={GstPaddingHelpers.get_padded_width(self.video_metadata.video_width)},height={self.video_metadata.video_height},framerate={self.video_metadata.video_fps_exact.numerator}/{self.video_metadata.video_fps_exact.denominator}")
        self._appsrc.set_property('caps', caps)
        self._appsrc.set_property('duration', self.file_duration_ns)
        self.appsrc_lock.release()

    def stop(self):
        self._stop_appsource_worker()

    def _create_appsrc(self) -> GstApp:
        appsrc = Gst.ElementFactory.make('appsrc', "numpy-source")

        caps = Gst.Caps.from_string(
            f"video/x-raw,format=BGR,width={GstPaddingHelpers.get_padded_width(self.video_metadata.video_width)},height={self.video_metadata.video_height},framerate={self.video_metadata.video_fps_exact.numerator}/{self.video_metadata.video_fps_exact.denominator}")

        appsrc.set_property('caps', caps)
        appsrc.set_property('is-live', False)
        appsrc.set_property('emit-signals', True)
        appsrc.set_property('stream-type', GstApp.AppStreamType.SEEKABLE)
        appsrc.set_property('format', Gst.Format.TIME)
        appsrc.set_property('duration', self.file_duration_ns)
        appsrc.set_property('max-buffers', 5) # doesn't need to be much as we're using this AppSrc with a Queue
        appsrc.set_property('max-bytes', 0)
        appsrc.set_property('block', False)


        appsrc.connect('need-data', self._on_need_data)
        appsrc.connect('enough-data', self._on_enough_data)
        appsrc.connect('seek-data', self._on_seek_data)
        appsrc.connect("end-of-stream", self._on_eos)

        return appsrc

    def _on_eos(self, appsrc):
        logger.debug("appsource end-of-stream")
        self.eos_callback()
        return True

    def _on_need_data(self, src, length):
        logger.debug("appsource need-data")
        self.appsrc_lock.acquire()
        self._start_appsource_worker()
        self.appsrc_lock.release()
        return True

    def _on_enough_data(self, src):
        logger.debug("appsource enough-data")
        self.appsrc_lock.acquire()
        self._request_stop_appsource_worker()
        self.appsrc_lock.release()
        return True

    def _on_seek_data(self, appsrc, offset_ns):
        logger.debug(f"appsource seek: offset (sec): {offset_ns / Gst.SECOND}, current position (sec): {self.current_timestamp_ns / Gst.SECOND}")
        if offset_ns == self.current_timestamp_ns:
            # nothing to do, we're already at the desired position in the file or already received this seek request
            logger.debug("appsource seek: skipped seek as we're already at the seek position")
            return True
        self.appsrc_lock.acquire()
        self._stop_appsource_worker()
        self._start_appsource_worker(seek_position=offset_ns)
        self.appsrc_lock.release()
        return True

    def _start_appsource_worker(self, seek_position=None):
        self.frame_restorer_lock.acquire()
        self.appsource_thread_stop_requested = False
        self.appsource_thread_should_be_running = True

        if self.appsource_thread and self.appsource_thread.is_alive():
            logger.debug(f"appsource worker: requested to start but already started")
            self.frame_restorer_lock.release()
            return

        if seek_position:
            logger.debug(f"appsource worker: applying pending seek timestamp")
            assert self.appsource_thread is None, "starting appsource worker with pending timestamp but worker is still running -> you need to stop the worker before setting a pending timestamp"
            assert self.frame_restorer is None, "starting appsource worker with pending timestamp but frame restorer is still running -> you need to stop the frame restorer before setting a pending timestamp"

        if not self.frame_restorer:
            logger.debug(f"appsource worker: setting up frame restorer")
            self.frame_restorer = self.frame_restorer_provider.get()
            if seek_position is not None:
                self.frame_restorer.start(start_ns=int(seek_position))
                self.current_timestamp_ns = seek_position
            else:
                self.frame_restorer.start(start_ns=int(self.current_timestamp_ns))

        self.appsource_thread = threading.Thread(target=self._appsource_worker)
        self.appsource_thread.start()
        self.frame_restorer_lock.release()

    def _request_stop_appsource_worker(self):
        self.frame_restorer_lock.acquire()
        self.appsource_thread_stop_requested = True
        self.appsource_thread_should_be_running = False
        self.frame_restorer_lock.release()

    def _stop_appsource_worker(self):
        self.frame_restorer_lock.acquire()
        start = time.time()
        self.appsource_thread_stop_requested = True
        self.appsource_thread_should_be_running = False

        frame_restorer_thread_queue = None
        if self.frame_restorer:
            logger.debug(f"appsource worker: stopping frame restorer")
            self.frame_restorer.stop()
            frame_restorer_thread_queue = self.frame_restorer.get_frame_restoration_queue()
            # unblock consumer
            threading_utils.put_closing_queue_marker(frame_restorer_thread_queue, "frame_restorer_thread_queue")

        if self.appsource_thread:
            self.appsource_thread.join()
            self.appsource_thread = None

        if self.frame_restorer:
            # garbage collection
            threading_utils.empty_out_queue(frame_restorer_thread_queue, "frame_restorer_thread_queue")
            self.frame_restorer = None

        logger.debug(f"appsource worker: stopped, took {time.time() - start}")
        self.frame_restorer_lock.release()

    def _appsource_worker(self):
        logger.debug("appsource worker: started")
        eof = False
        while self.appsource_thread_should_be_running:
            eof = self._push_next_frame()
        if eof:
            logger.debug("appsource worker: stopped itself, EOF")

    def _push_next_frame(self) -> bool:
        result = self.frame_restorer.get_frame_restoration_queue().get()
        if self.appsource_thread_stop_requested:
            logger.debug("appsource worker: frame_restoration_queue consumer unblocked")
        if result is None:
            self.appsource_thread_should_be_running = False
            if not self.appsource_thread_stop_requested:
                self._appsrc.emit("end-of-stream")
                return True
            return False
        else:
            frame, frame_pts, _ = result

        frame_timestamp_ns = int((frame_pts * self.video_metadata.time_base) * Gst.SECOND)
        frame = GstPaddingHelpers.pad_frame(frame)

        data = frame.tobytes()

        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = round(self.frame_duration_ns)
        buf.pts = frame_timestamp_ns
        buf.offset = video_utils.offset_ns_to_frame_num(frame_timestamp_ns, self.video_metadata.video_fps_exact)
        self._appsrc.emit('push-buffer', buf)
        self.current_timestamp_ns = frame_timestamp_ns

        return False


class GstPaddingHelpers:
    # TODO: As we're using BGR format GStreamer expects to receive a 'buffer size = rstride (image) * height' where 'rstride = RU4 (width * 3)'
    # RU4 here means that it will round up to nearest number divisible by 4. (https://gstreamer.freedesktop.org/documentation/additional/design/mediatype-video-raw.html)
    # Most common video widths like 1920, 1280, 640 are divisible by 4 so no problem we can allocate a buffer according to numpy shape H*W*C
    # But if we receive a video with a width which isn't divisible by 4 (I saw this on a file with dimensions 854 x 480) then the pipeline would break as H*W*C is less then expected buffer size given above calculation.
    # For now let's just add some zero padding. Maybe there are ways to explicitly set the stride size or tell GStreamer about our zero padding but couldn't find anything...

    @staticmethod
    def pad_frame(frame: np.ndarray):
        width = frame.shape[1]
        # TODO: see reasoning for this zero padding in TODO where we specify appsrc Caps
        if width % 4 != 0:
            return np.pad(frame, ((0, 0), (0, width % 4), (0, 0)), mode='constant', constant_values=0)
        return frame

    @staticmethod
    def get_padded_width(width):
        return width + width % 4