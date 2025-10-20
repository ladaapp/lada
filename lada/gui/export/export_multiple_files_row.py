import logging
import pathlib
import threading

from gi.repository import Adw, Gtk, Gio, GObject, GLib

from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.export import export_utils
from lada.gui.export.export_item_data import ExportItemDataProgress, ExportItemState
from lada.gui.export.export_utils import MIN_VISIBLE_PROGRESS_FRACTION, get_video_metadata_string

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'export_multiple_files_row.ui'))
class ExportMultipleFilesRow(Adw.PreferencesRow):
    __gtype_name__ = "ExportMultipleFilesRow"

    progressbar: Gtk.ProgressBar = Gtk.Template.Child()
    button_open: Gtk.Button = Gtk.Template.Child()
    button_remove: Gtk.Button = Gtk.Template.Child()
    button_show_error: Gtk.Button = Gtk.Template.Child()

    def __init__(self, original_file: Gio.File, restored_file: Gio.File, **kwargs) -> None:
        super().__init__(**kwargs)
        self._restored_file = restored_file
        self.original_file = original_file
        self._progress: ExportItemDataProgress = ExportItemDataProgress()
        self._state: ExportItemState = ExportItemState.QUEUED
        self._subtitle = ""

        self.set_title(original_file.get_basename())
        self._attach_file_launcher_to_open_button()

        def update_title_with_video_metadata():
            subtitle = get_video_metadata_string(original_file)
            GLib.idle_add(lambda: self.set_property("subtitle", subtitle))
        threading.Thread(target=update_title_with_video_metadata).start()

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
            self.progressbar.add_css_class("finished")
            self.button_open.set_visible(True)
            self.button_remove.set_visible(True)
            self.button_show_error.set_visible(False)
            self.progressbar.set_text(export_utils.get_progressbar_text(self._state, self._progress))
            self.progressbar.set_show_text(True)
        elif value == ExportItemState.QUEUED:
            self.button_open.set_visible(False)
            self.button_remove.set_visible(True)
            self.button_show_error.set_visible(False)
            self.progressbar.set_show_text(False)
        elif value == ExportItemState.PROCESSING:
            self.button_open.set_visible(False)
            self.button_remove.set_visible(False)
            self.button_show_error.set_visible(False)
            self.progressbar.set_fraction(max(MIN_VISIBLE_PROGRESS_FRACTION, self._progress.fraction))
            self.progressbar.set_text(export_utils.get_progressbar_text(self._state, self._progress))
            self.progressbar.set_show_text(True)
        elif value == ExportItemState.FAILED:
            self.button_open.set_visible(False)
            self.button_remove.set_visible(True)
            self.button_show_error.set_visible(True)
            self.progressbar.add_css_class("failed")
            self.progressbar.set_text(export_utils.get_progressbar_text(self._state, self._progress))
            self.progressbar.set_show_text(True)
        elif value == ExportItemState.PAUSED:
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

    def _attach_file_launcher_to_open_button(self):
        file_launcher = Gtk.FileLauncher(
            always_ask=False,
            file=self._restored_file,
        )

        self.button_open.connect("clicked", lambda _: file_launcher.launch())

    @Gtk.Template.Callback()
    def button_remove_callback(self, button):
        self.emit("remove-requested")

    @Gtk.Template.Callback()
    def on_button_show_error_clicked(self, button):
        self.emit("show-error-requested")

    @GObject.Signal(name="remove-requested")
    def video_export_requested_signal(self):
        pass

    @GObject.Signal(name="show-error-requested")
    def show_error_requested_signal(self):
        pass