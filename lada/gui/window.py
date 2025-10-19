import pathlib
import threading

from gi.repository import Adw, Gtk, Gio, GLib, GObject

from lada.gui import utils
from lada.gui.config.config import Config
from lada.gui.export.export_view import ExportView
from lada.gui.fileselection.file_selection_view import FileSelectionView
from lada.gui.preview.preview_view import PreviewView
from lada.gui.shortcuts import ShortcutsManager

here = pathlib.Path(__file__).parent.resolve()

@Gtk.Template(string=utils.translate_ui_xml(here / 'window.ui'))
class MainWindow(Adw.ApplicationWindow):
    __gtype_name__ = 'MainWindow'

    file_selection_view: FileSelectionView = Gtk.Template.Child()
    export_view: ExportView = Gtk.Template.Child()
    preview_view: PreviewView = Gtk.Template.Child()
    view_stack: Adw.ViewStack = Gtk.Template.Child()
    stack: Gtk.Stack = Gtk.Template.Child()
    shortcut_controller = Gtk.Template.Child()

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @GObject.Property(type=ShortcutsManager)
    def shortcuts_manager(self):
        return self._shortcuts_manager

    @shortcuts_manager.setter
    def shortcuts_manager(self, value):
        self._shortcuts_manager = value
        self._setup_shortcuts()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._config: Config | None
        self._shortcuts_manager: ShortcutsManager | None = None

        self.set_title("Lada")

        self.connect("close-request", self.close)
        self.file_selection_view.connect("files-selected", lambda obj, files: self.on_files_selected(files))
        self.preview_view.connect("toggle-fullscreen-requested", lambda *args: self.on_toggle_fullscreen())
        self.connect("notify::fullscreened", lambda object, spec: self.on_fullscreened(object.get_property(spec.name)))

        self.export_view.props.view_stack = self.view_stack
        self.export_view.connect("video-export-requested", lambda obj, restore_directory_or_file: self.on_video_export_requested(restore_directory_or_file))
        self.preview_view.props.view_stack = self.view_stack

    def on_video_export_requested(self, restore_directory_or_file: Gio.File):
        self.stack.props.visible_child_name = "main"
        self.view_stack.props.visible_child_name = "export"
        def run():
            self.preview_view.close(block=True)
            GLib.idle_add(lambda: self.export_view.start_export(restore_directory_or_file))
        threading.Thread(target=run).start()

    def on_files_selected(self, files: list[Gio.File]):
        self.stack.props.visible_child_name = "main"
        self.view_stack.props.visible_child_name = "preview" if self._config.initial_view == "preview" else "export"
        self.preview_view.add_files(files)
        if self.view_stack.props.visible_child_name == "preview":
            self.preview_view.play_file(0)
        self.export_view.add_files(files)

    def on_fullscreened(self, fullscreened: bool):
        if self.stack.props.visible_child_name == "main" and self.view_stack.props.visible_child_name == "preview":
            self.preview_view.on_fullscreened(fullscreened)

    def on_toggle_fullscreen(self):
        if self.is_fullscreen():
            self.unfullscreen()
        else:
            self.fullscreen()

    def _setup_shortcuts(self):
        self._shortcuts_manager.register_group("ui", "UI")
        def switch_views(child_name):
            if self.stack.props.visible_child_name == "main":
                self.view_stack.set_visible_child_name(child_name)
        self._shortcuts_manager.add("ui", "show-export-view", "e", lambda *args: switch_views('export'), _("Switch to Export View"))
        self._shortcuts_manager.add("ui", "show-preview-view", "p", lambda *args: switch_views('preview'), _("Switch to Preview View"))

    def close(self, *args):
        self.preview_view.close()
        self.export_view.close()