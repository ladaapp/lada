# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pathlib

from gi.repository import Adw, Gtk, GObject

from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.config.config import Config

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'no_gpu_banner.ui'))
class NoGpuBanner(Gtk.Box):
    __gtype_name__ = "NoGpuBanner"

    banner: Adw.Banner = Gtk.Template.Child()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config: Config | None = None

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
        if self._config.get_property('device') == 'cpu':
            self.banner.set_revealed(True)

    @Gtk.Template.Callback()
    def banner_no_gpu_button_clicked(self, button_clicked):
        self.banner.set_revealed(False)

    def set_revealed(self, value: bool):
        self.banner.set_revealed(value)