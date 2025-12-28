# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pathlib
import threading

from gi.repository import Adw, Gtk, Gio, GObject, GLib

from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.export import export_utils
from lada.gui.export.export_item_data import ExportItemDataProgress, ExportItemState
from lada.gui.export.export_utils import MIN_VISIBLE_PROGRESS_FRACTION, get_video_metadata_string
from lada.gui.export.spinner_button import SpinnerButton

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'export_multiple_files_row.ui'))
class ExportMultipleFilesRow(Adw.PreferencesRow):
    __gtype_name__ = "ExportMultipleFilesRow"

    progressbar: Gtk.ProgressBar = Gtk.Template.Child()
    button_open: Gtk.Button = Gtk.Template.Child()
    button_preview: SpinnerButton = Gtk.Template.Child()
    button_remove: Gtk.Button = Gtk.Template.Child()
    button_show_error: Gtk.Button = Gtk.Template.Child()

    def __init__(self, original_file: Gio.File, restored_file: Gio.File, **kwargs) -> None:
        super().__init__(**kwargs)
        self._restored_file = restored_file
        self.original_file = original_file
        self._progress: ExportItemDataProgress = ExportItemDataProgress()
        self._state: ExportItemState = ExportItemState.QUEUED
        self._subtitle = ""
        self._temp_file_path: str | None = None
        self._temp_file_path_check_ready_timeout_id: int | None = None

        self.set_title(original_file.get_basename())
        self._handler_id_button_open_clicked = None
        self._attach_file_launcher_to_open_button()

        def update_title_with_video_metadata():
            subtitle = get_video_metadata_string(original_file)
            GLib.idle_add(lambda: self.set_property("subtitle", subtitle))
        threading.Thread(target=update_title_with_video_metadata, daemon=True).start()

    @GObject.Property(type=ExportItemDataProgress)
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, value: ExportItemDataProgress):
        self._progress = value
        fraction = max(MIN_VISIBLE_PROGRESS_FRACTION, self._progress.fraction) if self._state != ExportItemState.QUEUED else self._progress.fraction
        self.progressbar.set_fraction(fraction)
        self.progressbar.set_text(export_utils.get_progressbar_text(self._state, self._progress))

    @GObject.Property(type=ExportItemState, default=ExportItemState.QUEUED)
    def state(self):
        return self._state

    @state.setter
    def state(self, value: ExportItemState):
        self._state = value
        if value == ExportItemState.FINISHED:
            self.progressbar.set_fraction(1.0)
            self.progressbar.add_css_class("finished")
            self.button_open.set_visible(True)
            self.button_preview.set_visible(False)
            self.button_remove.set_visible(True)
            self.button_show_error.set_visible(False)
            self.progressbar.set_text(export_utils.get_progressbar_text(self._state, self._progress))
            self.progressbar.set_show_text(True)
            if self._temp_file_path_check_ready_timeout_id:
                GLib.source_remove(self._temp_file_path_check_ready_timeout_id)
                self._temp_file_path_check_ready_timeout_id = None
        elif value == ExportItemState.QUEUED:
            self.progressbar.remove_css_class("finished")
            self.progressbar.remove_css_class("failed")
            self.button_open.set_visible(False)
            self.button_preview.set_visible(False)
            self.button_remove.set_visible(True)
            self.button_show_error.set_visible(False)
            self.progressbar.set_show_text(False)
            if self._temp_file_path_check_ready_timeout_id:
                GLib.source_remove(self._temp_file_path_check_ready_timeout_id)
                self._temp_file_path_check_ready_timeout_id = None
        elif value == ExportItemState.PROCESSING:
            self.progressbar.remove_css_class("finished")
            self.progressbar.remove_css_class("failed")
            self.button_open.set_visible(False)
            self.button_remove.set_visible(False)
            self.button_show_error.set_visible(False)
            self.progressbar.set_fraction(max(MIN_VISIBLE_PROGRESS_FRACTION, self._progress.fraction))
            self.progressbar.set_text(export_utils.get_progressbar_text(self._state, self._progress))
            self.progressbar.set_show_text(True)
            if self._temp_file_path is not None:
                self.button_preview.set_sensitive(False)
                self.button_preview.set_spinner_visible(True)
                self.button_preview.set_visible(True)
                # Check every second if file exists and has content
                def check_file_ready():
                    if export_utils.preview_file_available(self._temp_file_path):
                        self.button_preview.set_sensitive(True)
                        self.button_preview.set_spinner_visible(False)
                        return GLib.SOURCE_REMOVE
                    return GLib.SOURCE_CONTINUE
                self._temp_file_path_check_ready_timeout_id = GLib.timeout_add_seconds(1, check_file_ready)
            else:
                self.button_preview.set_sensitive(True)
                self.button_preview.set_spinner_visible(False)
                self.button_preview.set_visible(False)
                if self._temp_file_path_check_ready_timeout_id:
                    GLib.source_remove(self._temp_file_path_check_ready_timeout_id)
                    self._temp_file_path_check_ready_timeout_id = None
        elif value == ExportItemState.FAILED:
            self.progressbar.remove_css_class("finished")
            self.progressbar.add_css_class("failed")
            self.button_open.set_visible(False)
            self.button_preview.set_visible(False)
            self.button_remove.set_visible(True)
            self.button_show_error.set_visible(True)
            self.progressbar.set_text(export_utils.get_progressbar_text(self._state, self._progress))
            self.progressbar.set_show_text(True)
            if self._temp_file_path_check_ready_timeout_id:
                GLib.source_remove(self._temp_file_path_check_ready_timeout_id)
                self._temp_file_path_check_ready_timeout_id = None
        elif value == ExportItemState.PAUSED:
            self.progressbar.remove_css_class("finished")
            self.progressbar.remove_css_class("failed")
            self.button_open.set_visible(False)
            self.button_remove.set_visible(False)
            self.button_show_error.set_visible(False)
            self.progressbar.set_text(export_utils.get_progressbar_text(self._state, self._progress))
            self.progressbar.set_show_text(True)
        else:
            logger.error("Unhandled enum state", value)

    @GObject.Property(type=str)
    def subtitle(self):
        return self._subtitle

    @subtitle.setter
    def subtitle(self, value: str):
        self._subtitle = value

    @GObject.Property(type=Gio.File)
    def restored_file(self):
        return self._restored_file

    @restored_file.setter
    def restored_file(self, value: Gio.File):
        if self._restored_file.get_path() != value.get_path():
            self._restored_file = value
            self._attach_file_launcher_to_open_button()

    @GObject.Property(type=str)
    def temp_file_path(self):
        return self._temp_file_path

    @temp_file_path.setter
    def temp_file_path(self, value: str | None):
        self._temp_file_path = value

    def _attach_file_launcher_to_open_button(self):
        file_launcher = Gtk.FileLauncher(
            always_ask=False,
            file=self._restored_file,
        )
        if self._handler_id_button_open_clicked is not None:
            self.button_open.disconnect(self._handler_id_button_open_clicked)
        self._handler_id_button_open_clicked = self.button_open.connect("clicked", lambda _: file_launcher.launch())

    @Gtk.Template.Callback()
    def button_remove_callback(self, button):
        self.emit("remove-requested")

    @Gtk.Template.Callback()
    def on_button_show_error_clicked(self, button):
        self.emit("show-error-requested")

    @Gtk.Template.Callback()
    def on_button_preview_clicked(self, button):
        assert export_utils.preview_file_available(self._temp_file_path)
        temp_file = Gio.File.new_for_path(self._temp_file_path)
        preview_launcher = Gtk.FileLauncher(
            always_ask=False,
            file=temp_file
        )
        preview_launcher.launch()

    @GObject.Signal(name="remove-requested")
    def video_export_requested_signal(self):
        pass

    @GObject.Signal(name="show-error-requested")
    def show_error_requested_signal(self):
        pass