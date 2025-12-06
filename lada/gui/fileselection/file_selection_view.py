# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pathlib
from gi.repository import Adw, Gtk, Gio, GObject
from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.shortcuts import ShortcutsManager

here = pathlib.Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'file_selection_view.ui'))
class FileSelectionView(Gtk.Widget):
    __gtype_name__ = 'FileSelectionView'

    button_open_file: Gtk.Button = Gtk.Template.Child()
    status_page: Adw.StatusPage = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._shortcuts_manager: ShortcutsManager | None = None
        self._window_title: str | None = None

        drop_target = utils.create_video_files_drop_target(lambda files: self.emit("files-selected", files))
        self.add_controller(drop_target)

        logo_image = Gtk.Image.new_from_resource("/io/github/ladaapp/lada/icons/128x128/lada-logo-gray.png")
        self.status_page.set_paintable(logo_image.get_paintable())

    @GObject.Property(type=str)
    def window_title(self):
        return self._window_title

    @window_title.setter
    def window_title(self, value):
        self._window_title = value

    @Gtk.Template.Callback()
    def button_open_file_callback(self, button_clicked):
        self.button_open_file.set_sensitive(False)
        callback = lambda files: self.emit("files-selected", files)
        dismissed_callback = lambda *args: self.button_open_file.set_sensitive(True)
        utils.show_open_files_dialog(callback, dismissed_callback)

    @GObject.Signal(name="files-selected", arg_types=(GObject.TYPE_PYOBJECT,))
    def files_opened_signal(self, files: list[Gio.File]):
        pass
