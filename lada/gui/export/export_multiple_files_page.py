import logging
import pathlib
import threading

from gi.repository import Gtk, Gio, GObject, GLib
from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.export import export_utils
from lada.gui.export.export_item_data import ExportItemData, ExportItemDataProgress, ExportItemState
from lada.gui.export.export_multiple_files_row import ExportMultipleFilesRow
from lada.gui.export.export_utils import MIN_VISIBLE_PROGRESS_FRACTION

here = pathlib.Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'export_multiple_files_page.ui'))
class ExportMultipleFilesPage(Gtk.Widget):
    __gtype_name__ = 'ExportMultipleFilesPage'

    list_box: Gtk.ListBox = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @GObject.Signal(name="remove-item-requested", arg_types=(GObject.TYPE_INT64,))
    def stop_export_requested_signal(self, idx: int):
        pass

    @GObject.Signal(name="show-error-requested", arg_types=(GObject.TYPE_INT64,))
    def pause_export_requested_signal(self, idx: int):
        pass

    def bind(self, model):
        self.list_box.bind_model(model, self.create_item_for_list_box_fun())

    def create_item_for_list_box_fun(self):
        def fun(obj: ExportItemData):
            list_row = ExportMultipleFilesRow(
                original_file=obj.original_file,
                restored_file=obj.restored_file,
            )
            list_row.connect("remove-requested", lambda *args: self.on_export_item_remove_requested(list_row))
            list_row.connect("show-error-requested", lambda *args: self.on_show_error_requested(list_row))
            return list_row
        return fun

    def on_export_item_remove_requested(self, view_item: ExportMultipleFilesRow):
        for idx, list_item in enumerate(self.list_box):
            if list_item.original_file == view_item.original_file:
                self.emit("remove-item-requested", idx)
                break

    def on_show_error_requested(self, view_item: ExportMultipleFilesRow):
        for idx, list_item in enumerate(self.list_box):
            if list_item.state == ExportItemState.FAILED and list_item.original_file == view_item.original_file:
                self.emit("show-error-requested", idx)
                break

    def on_video_export_finished(self, idx: int):
        view_item = self.list_box.get_row_at_index(idx)
        view_item.progress.complete()
        view_item.state = ExportItemState.FINISHED

    def on_video_export_progress(self, idx: int, progress: ExportItemDataProgress):
        view_item = self.list_box.get_row_at_index(idx)
        view_item.progress = progress

    def show_video_export_started(self, idx: int):
        view_item = self.list_box.get_row_at_index(idx)
        view_item.state = ExportItemState.PROCESSING

    def on_video_export_stopped(self, idx: int):
        view_item = self.list_box.get_row_at_index(idx)
        view_item.state = ExportItemState.QUEUED
        view_item.progress = ExportItemDataProgress()

    def on_video_export_paused(self, idx: int):
        view_item = self.list_box.get_row_at_index(idx)
        view_item.state = ExportItemState.PAUSED

    def on_video_export_resumed(self, idx: int):
        view_item = self.list_box.get_row_at_index(idx)
        view_item.state = ExportItemState.PROCESSING

    def on_video_export_failed(self, idx: int):
        view_item = self.list_box.get_row_at_index(idx)
        view_item.state = ExportItemState.FAILED

    def on_video_export_started(self, restored_files: list[Gio.File]):
        for idx, restored_file in enumerate(restored_files):
            view_item = self.list_box.get_row_at_index(idx)
            view_item.restored_file = restored_file
