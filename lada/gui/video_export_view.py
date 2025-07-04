import logging
import os
import pathlib
import tempfile
import threading

from gi.repository import Gtk, GObject, Gio

from lada.gui.config import Config
from lada.lib import audio_utils, video_utils
from lada import LOG_LEVEL
from lada.gui.frame_restorer_provider import FrameRestorerOptions, FRAME_RESTORER_PROVIDER

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(filename=here / 'video_export_view.ui')
class VideoExportView(Gtk.Widget):
    __gtype_name__ = 'VideoExportView'

    status_page = Gtk.Template.Child()
    progress_bar_file_export = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._window_title: str | None = None
        self._config: Config | None = None
        self.export_in_progress = False

        self.connect("video-export-finished", self.show_video_export_success)
        self.connect("video-export-progress", self.on_video_export_progress)
        
    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @GObject.Property(type=str)
    def window_title(self):
        return self._window_title

    @window_title.setter
    def window_title(self, value):
        self._window_title = value

    @GObject.Signal(name="video-export-finished")
    def video_export_finished_signal(self):
        pass

    @GObject.Signal(name="video-export-progress")
    def video_export_progress_signal(self, status: float):
        pass

    def show_video_export_success(self, obj):
        self.status_page.set_title("Finished video restoration!")
        self.status_page.set_icon_name("check-round-outline2-symbolic")
        self.progress_bar_file_export.set_fraction(1.0)
        self.export_in_progress = False

    def show_video_export_started(self):
        self.status_page.set_title("Exporting restored video...")
        self.status_page.set_icon_name("cafe-symbolic")
        self.progress_bar_file_export.set_fraction(0.)
        self.export_in_progress = True

    def on_video_export_progress(self, obj, progress):
        self.progress_bar_file_export.set_fraction(progress)

    def start_export(self, source_file: Gio.File, save_file: Gio.File):
        self.show_video_export_started()

        def run_export():
            frame_restorer_options = FrameRestorerOptions(self._config.mosaic_restoration_model, self._config.mosaic_detection_model, video_utils.get_video_meta_data(source_file.get_path()), self._config.device, self._config.max_clip_duration, False, False)
            video_metadata = frame_restorer_options.video_metadata
            frame_restorer_provider = FRAME_RESTORER_PROVIDER
            frame_restorer_provider.init(frame_restorer_options)
            frame_restorer = frame_restorer_provider.get()

            progress_update_step_size = 100
            success = True
            video_tmp_file_output_path = os.path.join(tempfile.gettempdir(),f"{os.path.basename(os.path.splitext(video_metadata.video_file)[0])}.tmp{os.path.splitext(video_metadata.video_file)[1]}")
            try:
                frame_restorer.start(start_ns=0)

                with video_utils.VideoWriter(video_tmp_file_output_path, video_metadata.video_width,
                                             video_metadata.video_height, video_metadata.video_fps_exact,
                                             self._config.export_codec, time_base=video_metadata.time_base,
                                             crf=self._config.export_crf) as video_writer:
                    for frame_num, elem in enumerate(frame_restorer):
                        if elem is None:
                            success = False
                            logger.error("Error on export: frame restorer stopped prematurely")
                            break
                        (restored_frame, restored_frame_pts) = elem
                        video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)
                        if frame_num % progress_update_step_size == 0:
                            self.emit('video-export-progress', frame_num / video_metadata.frames_count)

            except Exception as e:
                success = False
                logger.error("Error on export", exc_info=e)
            finally:
                frame_restorer.stop()

            if success:
                audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, save_file.get_path())
                self.emit('video-export-progress', 1.0)
                self.emit('video-export-finished')
            else:
                if os.path.exists(video_tmp_file_output_path):
                    os.remove(video_tmp_file_output_path)

        exporter_thread = threading.Thread(target=run_export)
        exporter_thread.start()
