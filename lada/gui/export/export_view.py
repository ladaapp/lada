import logging
import os
import pathlib
import tempfile
import threading

from gi.repository import Gtk, GObject, Gio, Adw, GLib

from gettext import gettext as _
from lada.gui.config.config import Config
from lada.gui.export.export_item_data import ExportItemData
from lada.gui.export.export_item_row import ExportItemRow, ExportItemState, get_video_metadata_string
from lada.lib import audio_utils, video_utils
from lada import LOG_LEVEL
from lada.gui.frame_restorer_provider import FrameRestorerOptions, FRAME_RESTORER_PROVIDER

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(filename=here / 'export_view.ui')
class ExportView(Gtk.Widget):
    __gtype_name__ = 'ExportView'

    status_page = Gtk.Template.Child()
    list_box: Gtk.ListBox = Gtk.Template.Child()
    button_start_export: Gtk.Button = Gtk.Template.Child()
    button_start_export_status_page: Gtk.Button = Gtk.Template.Child()
    progress_bar_file_export_status_page: Gtk.ProgressBar = Gtk.Template.Child()
    label_meta_data_status_page: Gtk.Label = Gtk.Template.Child()
    label_file_name_status_page: Gtk.Label = Gtk.Template.Child()
    stack: Gtk.Stack = Gtk.Template.Child()
    button_open_status_page: Gtk.Button = Gtk.Template.Child()
    view_switcher: Adw.ViewSwitcher = Gtk.Template.Child()
    config_sidebar = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._view_stack: Adw.ViewStack | None = None
        self._config: Config | None = None
        self.export_in_progress = False
        self.in_progress_idx: int | None = None
        self._files: list[Gio.File] = []
        self.single_file = True
        self.close_requested = False

        self.connect("video-export-finished", self.show_video_export_success)
        self.connect("video-export-progress", self.on_video_export_progress)

        self.model =  Gio.ListStore(item_type=ExportItemData)
        self.list_box.bind_model(self.model, self.create_item_for_list_box)

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

    def open_files(self, value: list[Gio.File]):
        self._files = value
        self.single_file = len(self._files) == 1
        if self.single_file:
            file = self._files[0]
            self.stack.set_visible_child_name("single-file")
            self.button_start_export_status_page.set_visible(True)
            self.label_meta_data_status_page.set_visible(True)
            self.label_file_name_status_page.set_label(file.get_basename())
            def update_label_with_video_metadata():
                label = get_video_metadata_string(file)
                GLib.idle_add(lambda: self.label_meta_data_status_page.set_label(label))
            threading.Thread(target=update_label_with_video_metadata).start()
        else:
            self.stack.set_visible_child_name("multiple-files")
            self.button_start_export.set_visible(True)

            for orig_file in self._files:
                if self._config.export_directory:
                    restored_file = self.get_restored_file_path(orig_file, self._config.export_directory)
                else:
                    # We don't know the output directory yet. This guess needs to be updated after the user set one via FilePicker
                    restored_file = self.get_restored_file_path(orig_file, self._files[0].get_parent().get_path())
                export_item = ExportItemData(orig_file, restored_file)
                self.model.append(export_item)

    @GObject.Signal(name="video-export-finished")
    def video_export_finished_signal(self):
        pass

    @GObject.Signal(name="video-export-progress")
    def video_export_progress_signal(self, status: float):
        pass

    @GObject.Signal(name="video-export-requested")
    def video_export_requested_signal(self, source_file: Gio.File, save_file: Gio.File):
        pass

    @Gtk.Template.Callback()
    def button_start_export_callback(self, button_clicked):
        self.export_in_progress = True
        if self._config.export_directory:
            if self.single_file:
                orig_file = self._files[0]
                restored_file = self.get_restored_file_path(self._files[0], self._config.export_directory)
                self.emit("video-export-requested", orig_file, restored_file)
            else:
                self.emit("video-export-requested", self.model[0].orig_file, self.model[0].restored_file)
        else:
            self.show_export_dialog()

    def on_config_changed(self, *args):
        if not self.single_file and self._config.export_directory:
            for model_item, orig_file in zip(self.model, self._files):
                restored_file = self.get_restored_file_path(orig_file, self._config.export_directory)
                model_item.restored_file = restored_file
        self.set_restore_button_label()

    def set_restore_button_label(self):
        label = "Restore" if self._config.export_directory else "Restore…"
        self.button_start_export_status_page.set_label(label)
        self.button_start_export.set_label(label)

    def get_next_queued_item_idx(self) -> int | None:
        for idx, item in enumerate(self.model):
            if item.state == ExportItemState.QUEUED:
                return idx
        return None

    def show_video_export_success(self, obj):
        self.view_switcher.set_sensitive(True)
        if self.single_file:
            self.status_page.set_title(_("Finished video restoration!"))
            self.status_page.set_icon_name("check-round-outline2-symbolic")
            self.progress_bar_file_export_status_page.set_visible(False)
            self.button_open_status_page.set_visible(True)
            self.view_switcher.set_sensitive(True)
            self.config_sidebar.set_property("disabled", False)
        else:
            assert self.in_progress_idx is not None
            idx = self.in_progress_idx

            view_item = self.list_box.get_row_at_index(idx)
            model_item = self.model[idx]

            model_item.progress = 1.0
            model_item.state = ExportItemState.FINISHED

            view_item.progress = 1.0
            view_item.state = ExportItemState.FINISHED

            self.export_in_progress = False

            idx = self.get_next_queued_item_idx()
            if idx is None:
                # done, all queued items processed
                self.view_switcher.set_sensitive(True)
                self.config_sidebar.set_property("disabled", False)
            else:
                # continue, queued items remaining
                self._start_export(self.model[idx].orig_file, self.model[idx].restored_file)

    def show_video_export_started(self, save_file: Gio.File):
        self.view_switcher.set_sensitive(False)
        self.config_sidebar.set_property("disabled", True)
        self.button_start_export.set_visible(False)
        if self.single_file:
            self.status_page.set_title(_("Restoring video…"))
            self.status_page.set_icon_name("cafe-symbolic")
            self.progress_bar_file_export_status_page.set_visible(True)
            self.button_start_export_status_page.set_visible(False)
            file_launcher = Gtk.FileLauncher(
                always_ask=False,
                file=save_file
            )
            self.button_open_status_page.connect("clicked", lambda _: file_launcher.launch())
        else:
            idx = self.get_next_queued_item_idx()
            if idx is None:
                return

            self.in_progress_idx = idx
            self.export_in_progress = True

            view_item = self.list_box.get_row_at_index(idx)
            model_item = self.model[idx]

            model_item.progress = 0.
            model_item.state = ExportItemState.PROCESSING

            view_item.progress = 0.
            view_item.state = ExportItemState.PROCESSING


    def on_video_export_progress(self, obj, progress):
        if self.single_file:
            self.progress_bar_file_export_status_page.set_fraction(progress)
        else:
            if self.in_progress_idx is None:
                return
            idx = self.in_progress_idx

            view_item = self.list_box.get_row_at_index(idx)
            model_item = self.model[idx]

            model_item.progress = progress
            view_item.progress = progress

    def start_export(self, source_file: Gio.File, restore_directory_or_file: Gio.File):
        assert os.path.isfile(source_file.get_path())

        if os.path.isdir(restore_directory_or_file.get_path()):
            restore_directory = restore_directory_or_file
            if not self._config.export_directory:
                # Update initial guessed output directory for all items now that the user has provided one via FilePicker
                for idx, model_item in enumerate(self.model):
                    view_item = self.list_box.get_row_at_index(idx)
                    restored_file = self.get_restored_file_path(model_item.orig_file, restore_directory.get_path())
                    model_item.restored_file = restored_file
                    view_item.restored_file = restored_file
            self._start_export(self.model[0].orig_file, self.model[0].restored_file)
        else:
            restore_file = restore_directory_or_file
            self._start_export(source_file, restore_file)

    def _start_export(self, source_file: Gio.File, restore_file: Gio.File):
        assert os.path.isfile(source_file.get_path())
        assert os.path.isfile(source_file.get_path())
        self.show_video_export_started(restore_file)

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
                                             video_metadata.is_vfr,
                                             self._config.export_codec, time_base=video_metadata.time_base,
                                             crf=self._config.export_crf, custom_encoder_options=self._config.custom_ffmpeg_encoder_options) as video_writer:
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
                        if frame_num % progress_update_step_size == 0:
                            GLib.idle_add(lambda: self.emit('video-export-progress', frame_num / video_metadata.frames_count))

            except Exception as e:
                success = False
                logger.error("Error on export", exc_info=e)
            finally:
                frame_restorer.stop()

            if success:
                audio_utils.combine_audio_video_files(video_metadata, video_tmp_file_output_path, restore_file.get_path())
                GLib.idle_add(lambda: self.emit('video-export-progress', 1.0))
                GLib.idle_add(lambda: self.emit('video-export-finished'))
            else:
                if os.path.exists(video_tmp_file_output_path):
                    os.remove(video_tmp_file_output_path)

        exporter_thread = threading.Thread(target=run_export)
        exporter_thread.start()

    def create_item_for_list_box(_view, obj: ExportItemData):
        list_row = ExportItemRow(
            original_file=obj.orig_file,
            restored_file=obj.restored_file,
        )
        return list_row

    def on_remove_clicked(self, _button):
        selected_row = self.list_box.get_selected_row()
        index = selected_row.get_index()
        self.model.remove(index)

    def show_export_dialog(self):
        def on_dialog_result(dialog, result):
            try:
                if self.single_file:
                    selected = dialog.save_finish(result)
                else:
                    selected = dialog.select_folder_finish(result)
                if selected is not None:
                    self.emit("video-export-requested", first_orig_file, selected)
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
            first_orig_file = self._files[0]
            initial_restored_file = self.get_restored_file_path(first_orig_file, first_orig_file.get_parent().get_path())
            file_dialog.set_initial_folder(initial_restored_file.get_parent())
            file_dialog.set_initial_name(initial_restored_file.get_basename())
            file_dialog.save(callback=on_dialog_result)
        else:
            file_dialog = Gtk.FileDialog()
            file_dialog.set_title(_("Save restored video files"))
            first_orig_file = self._files[0]
            file_dialog.set_initial_folder(first_orig_file.get_parent())
            file_dialog.select_folder(callback=on_dialog_result)

    def get_restored_file_path(self, original_file: Gio.File, output_dir: str) -> Gio.File:
        orig_file_name = os.path.splitext(original_file.get_basename())[0]
        restored_file_name = self._config.file_name_pattern.replace("{orig_file_name}", orig_file_name)
        return Gio.File.new_build_filenamev([output_dir, restored_file_name])

    def close(self):
        self.close_requested = True