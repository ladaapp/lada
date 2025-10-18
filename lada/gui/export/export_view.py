import logging
import os
import pathlib
import tempfile
import threading
import time
import traceback

from gi.repository import Gtk, GObject, Gio, Adw, GLib

from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.config.config import Config
from lada.gui.config.no_gpu_banner import NoGpuBanner
from lada.gui.export import export_utils
from lada.gui.export.export_item_data import ExportItemData
from lada.gui.export.export_item_row import ExportItemRow
from lada.gui.export.export_single_file_status_page import ExportSingleFileStatusPage
from lada.gui.export.export_utils import ExportItemState
from lada.gui.frame_restorer_provider import FrameRestorerOptions, FRAME_RESTORER_PROVIDER
from lada.lib import audio_utils, video_utils

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'export_view.ui'))
class ExportView(Gtk.Widget):
    __gtype_name__ = 'ExportView'

    status_page: ExportSingleFileStatusPage = Gtk.Template.Child()
    list_box: Gtk.ListBox = Gtk.Template.Child()
    button_start_export: Gtk.Button = Gtk.Template.Child()
    stack: Gtk.Stack = Gtk.Template.Child()
    view_switcher: Adw.ViewSwitcher = Gtk.Template.Child()
    config_sidebar = Gtk.Template.Child()
    button_add_files: Gtk.Button = Gtk.Template.Child()
    banner_no_gpu: NoGpuBanner = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._view_stack: Adw.ViewStack | None = None
        self._config: Config | None = None
        self.in_progress_idx: int | None = None
        self.single_file = True
        self.close_requested = False

        self.connect("video-export-finished", self.on_video_export_finished)
        self.connect("video-export-failed", self.on_video_export_failed)
        self.connect("video-export-progress", self.on_video_export_progress)

        self.model =  Gio.ListStore(item_type=ExportItemData)
        self.list_box.bind_model(self.model, self.create_item_for_list_box_fun())

        def on_files_added(obj, files):
            self.button_add_files.set_sensitive(True)
            self.add_files(files)
        self.connect("files-added", on_files_added)

        self.status_page.connect("start-export-requested", self.button_start_export_callback)

        drop_target = utils.create_video_files_drop_target(lambda files: self.emit("files-added", files))
        self.add_controller(drop_target)

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
        self._config.connect("notify::export-directory", self.on_config_changed)
        self._config.connect("notify::file-name-pattern", self.on_config_changed)
        self.set_restore_button_label()

    @GObject.Property(type=Adw.ViewStack)
    def view_stack(self):
        return self._view_stack

    @view_stack.setter
    def view_stack(self, value: Adw.ViewStack):
        self._view_stack = value

    def add_files(self, added_files: list[Gio.File]):
        assert len(added_files) > 0

        for orig_file in added_files:
            if any([orig_file.get_path() == item.orig_file.get_path() for item in self.model]):
                # duplicate
                continue
            if self._config.export_directory:
                restored_file = self.get_restored_file_path(orig_file, self._config.export_directory)
            else:
                # We don't know the output directory yet. This guess needs to be updated after the user set one via FilePicker
                restored_file = self.get_restored_file_path(orig_file, added_files[0].get_parent().get_path())
            export_item = ExportItemData(orig_file, restored_file)
            self.model.append(export_item)

        self.single_file = len(self.model) == 1

        if self.single_file:
            self.stack.set_visible_child_name("single-file")
            self.status_page.on_add_file(self.model[0])
        else:
            self.stack.set_visible_child_name("multiple-files")
            self.button_start_export.set_visible(self.is_should_show_start_button())

    def is_should_show_start_button(self) -> bool:
        count_queued_items = sum([item.state == ExportItemState.QUEUED for item in self.model])
        is_in_progress = self.in_progress_idx is None
        return is_in_progress and count_queued_items > 0

    @GObject.Signal(name="video-export-finished")
    def video_export_finished_signal(self):
        pass

    @GObject.Signal(name="video-export-failed", arg_types=(GObject.TYPE_STRING,))
    def video_export_failed_signal(self, error_message: str):
        pass

    @GObject.Signal(name="video-export-progress")
    def video_export_progress_signal(self, status: float, time_remaining: str):
        pass

    @GObject.Signal(name="video-export-requested")
    def video_export_requested_signal(self, save_file: Gio.File):
        pass

    @GObject.Signal(name="files-added", arg_types=(GObject.TYPE_PYOBJECT,))
    def files_opened_signal(self, files: list[Gio.File]):
        pass

    @Gtk.Template.Callback()
    def button_start_export_callback(self, *args):
        if self._config.export_directory:
            item = self.model[self.get_next_queued_item_idx()]
            self.emit("video-export-requested", item.restored_file)
        else:
            self.show_export_dialog()

    @Gtk.Template.Callback()
    def button_add_files_callback(self, button_clicked):
        self.button_add_files.set_sensitive(False)
        callback = lambda files: self.emit("files-added", files)
        dismissed_callback = lambda *args: self.button_add_files.set_sensitive(True)
        utils.show_open_files_dialog(callback, dismissed_callback)

    def on_config_changed(self, *args):
        if self._config.export_directory:
            for model_item in self.model:
                restored_file = self.get_restored_file_path(model_item.orig_file, self._config.export_directory)
                model_item.restored_file = restored_file
        self.set_restore_button_label()

    def set_restore_button_label(self):
        label = _("Restore") if self._config.export_directory else _("Restore…")
        self.status_page.set_button_start_restore_label(label)
        self.button_start_export.set_label(label)

    def get_next_queued_item_idx(self) -> int | None:
        for idx, item in enumerate(self.model):
            if item.state == ExportItemState.QUEUED:
                return idx
        return None

    def on_video_export_finished(self, obj):
        assert self.in_progress_idx is not None
        current_idx = self.in_progress_idx

        view_item = self.list_box.get_row_at_index(current_idx)
        model_item = self.model[current_idx]

        model_item.progress = 1.0
        model_item.state = ExportItemState.FINISHED

        view_item.progress = 1.0
        view_item.state = ExportItemState.FINISHED

        if self.single_file:
            self.status_page.on_video_export_finished()

        self.continue_next_file()

    def continue_next_file(self):
        next_idx = self.get_next_queued_item_idx()
        if next_idx is None:
            # done, all queued items processed
            self.view_switcher.set_sensitive(True)
            self.config_sidebar.set_property("disabled", False)
            self.in_progress_idx = None
        else:
            # continue, queued items remaining
            self._start_export(self.model[next_idx].orig_file, self.model[next_idx].restored_file)

    def show_video_export_started(self, save_file: Gio.File):
        self.view_switcher.set_sensitive(False)
        self.config_sidebar.set_property("disabled", True)
        self.button_start_export.set_visible(False)

        idx = self.get_next_queued_item_idx()
        if idx is None:
            return

        self.in_progress_idx = idx

        view_item = self.list_box.get_row_at_index(idx)
        model_item = self.model[idx]

        model_item.progress = 0.
        model_item.state = ExportItemState.PROCESSING

        view_item.progress = 0.
        view_item.state = ExportItemState.PROCESSING

        if self.single_file:
            self.status_page.show_video_export_started(save_file)

    def on_video_export_progress(self, obj, progress:float, time_remaining: str):
        if self.in_progress_idx is None:
            return
        idx = self.in_progress_idx

        view_item = self.list_box.get_row_at_index(idx)
        model_item = self.model[idx]

        model_item.progress = progress
        view_item.progress = progress
        view_item.time_remaining = time_remaining

        if self.single_file:
            self.status_page.on_video_export_progress(progress, time_remaining)

    def on_video_export_failed(self, obj, error_message):
        assert self.in_progress_idx is not None
        current_idx = self.in_progress_idx

        view_item = self.list_box.get_row_at_index(current_idx)
        model_item = self.model[current_idx]

        model_item.state = ExportItemState.FAILED
        model_item.error_details = error_message
        view_item.state = ExportItemState.FAILED

        if self.single_file:
            self.status_page.on_video_export_failed()

        export_utils.open_error_dialog(self, model_item.orig_file.get_basename(), error_message)

        self.continue_next_file()

    def start_export(self, restore_directory_or_file: Gio.File):
        # Update initial guessed output restore directory/file now that the user has provided it via file/dir picker dialog
        if not self._config.export_directory:
            if self.single_file:
                assert len(self.model) == 1
                restored_file = restore_directory_or_file
                model_item = self.model[0]
                view_item = self.list_box.get_row_at_index(0)
                model_item.restored_file = restored_file
                view_item.restored_file = restored_file
            else:
                assert os.path.isdir(restore_directory_or_file.get_path())
                restore_directory = restore_directory_or_file
                for idx, model_item in enumerate(self.model):
                    view_item = self.list_box.get_row_at_index(idx)
                    restored_file = self.get_restored_file_path(model_item.orig_file, restore_directory.get_path())
                    model_item.restored_file = restored_file
                    view_item.restored_file = restored_file

        item = self.model[self.get_next_queued_item_idx()]
        self._start_export(item.orig_file, item.restored_file)

    def _start_export(self, source_file: Gio.File, restore_file: Gio.File):
        assert os.path.isfile(source_file.get_path())
        self.show_video_export_started(restore_file)

        def run_export():
            frame_restorer_options = FrameRestorerOptions(self._config.mosaic_restoration_model, self._config.mosaic_detection_model, video_utils.get_video_meta_data(source_file.get_path()), self._config.device, self._config.max_clip_duration, False, False)
            video_metadata = frame_restorer_options.video_metadata
            frame_restorer_provider = FRAME_RESTORER_PROVIDER
            frame_restorer_provider.init(frame_restorer_options)
            frame_restorer = frame_restorer_provider.get()
            restore_file_path = restore_file.get_path()

            progress_update_step_size = 100
            success = True
            video_tmp_file_output_path = os.path.join(tempfile.gettempdir(),f"{os.path.basename(os.path.splitext(restore_file_path)[0])}.tmp{os.path.splitext(restore_file_path)[1]}")
            remaining_processing_time_estimator = export_utils.RemainingProcessingTimeEstimator(video_metadata)
            try:
                frame_restorer.start(start_ns=0)

                with video_utils.VideoWriter(video_tmp_file_output_path, video_metadata.video_width,
                                             video_metadata.video_height, video_metadata.video_fps_exact,
                                             video_metadata.is_vfr,
                                             self._config.export_codec, time_base=video_metadata.time_base,
                                             crf=self._config.export_crf, custom_encoder_options=self._config.custom_ffmpeg_encoder_options) as video_writer:
                    start = time.time()
                    for frame_num, elem in enumerate(frame_restorer):
                        if self.close_requested:
                            success = False
                            logger.warning("Close requested: frame restorer stopped prematurely")
                            break
                        if elem is None:
                            success = False
                            logger.error("Error on export: frame restorer stopped prematurely")
                            break

                        (restored_frame, restored_frame_pts) = elem
                        video_writer.write(restored_frame, restored_frame_pts, bgr2rgb=True)

                        stop = time.time()
                        duration = stop - start
                        start = stop
                        remaining_processing_time_estimator.add_processing_duration(duration)
                        if frame_num % progress_update_step_size == 0:
                            GLib.idle_add(lambda: self.emit('video-export-progress', frame_num / video_metadata.frames_count, remaining_processing_time_estimator.get_time_remaining(frame_num)))

            except Exception as e:
                success = False
                err_msg = "".join(traceback.format_exception_only(e))
                GLib.idle_add(lambda: self.emit('video-export-failed', err_msg))
            finally:
                frame_restorer.stop()

            if success:
                audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, restore_file_path)
                GLib.idle_add(lambda: self.emit('video-export-progress', 1.0, None))
                GLib.idle_add(lambda: self.emit('video-export-finished'))
            else:
                if os.path.exists(video_tmp_file_output_path):
                    os.remove(video_tmp_file_output_path)

        exporter_thread = threading.Thread(target=run_export)
        exporter_thread.start()

    def create_item_for_list_box_fun(self):
        def fun(obj: ExportItemData):
            list_row = ExportItemRow(
                original_file=obj.orig_file,
                restored_file=obj.restored_file,
            )
            list_row.connect("remove-requested", lambda *args: self.on_export_item_remove_requested(list_row))
            list_row.connect("show-error-requested", lambda *args: self.on_show_error_requested(list_row))
            return list_row
        return fun

    def on_export_item_remove_requested(self, view_item: ExportItemRow):
        for idx, model_item in enumerate(self.model):
            if model_item.orig_file == view_item.original_file:
                self.model.remove(idx)
                break

    def on_show_error_requested(self, view_item: ExportItemRow):
        for idx, model_item in enumerate(self.model):
            if model_item.state == ExportItemState.FAILED and model_item.orig_file == view_item.original_file:
                export_utils.open_error_dialog(self, model_item.orig_file.get_basename(), model_item.error_details)
                break

    def show_export_dialog(self):
        def on_dialog_result(dialog, result):
            try:
                if self.single_file:
                    selected = dialog.save_finish(result)
                else:
                    selected = dialog.select_folder_finish(result)
                if selected is not None:
                    self.emit("video-export-requested",selected)
            except GLib.Error as error:
                if error.message == "Dismissed by user":
                    logger.debug("FileDialog cancelled: Dismissed by user")
                else:
                    logger.error(f"Error opening file: {error.message}")
                    raise error

        if self.single_file:
            file_dialog = Gtk.FileDialog()
            video_file_filter = Gtk.FileFilter()
            video_file_filter.add_mime_type("video/*")
            file_dialog.set_default_filter(video_file_filter)
            file_dialog.set_title(_("Save restored video file"))
            initial_restored_file = self.model[0].restored_file
            file_dialog.set_initial_folder(initial_restored_file.get_parent())
            file_dialog.set_initial_name(initial_restored_file.get_basename())
            file_dialog.save(callback=on_dialog_result)
        else:
            file_dialog = Gtk.FileDialog()
            file_dialog.set_title(_("Save restored video files"))
            first_orig_file = self.model[0].orig_file
            file_dialog.set_initial_folder(first_orig_file.get_parent())
            file_dialog.select_folder(callback=on_dialog_result)

    def get_restored_file_path(self, original_file: Gio.File, output_dir: str) -> Gio.File:
        orig_file_name = os.path.splitext(original_file.get_basename())[0]
        restored_file_name = self._config.file_name_pattern.replace("{orig_file_name}", orig_file_name)
        return Gio.File.new_build_filenamev([output_dir, restored_file_name])

    def close(self):
        self.close_requested = True
