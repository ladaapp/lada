import logging
import pathlib

from gi.repository import Gtk, Adw, GObject

from lada import LOG_LEVEL
from lada.gui import utils

here = pathlib.Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'spinner_button.ui'))
class SpinnerButton(Gtk.Button):
    __gtype_name__ = 'SpinnerButton'

    spinner: Adw.Spinner = Gtk.Template.Child()
    _label: Gtk.Label = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @GObject.Property(type=str)
    def label(self):
        return self._label.get_label()

    @label.setter
    def label(self, value):
        self._label.set_label(value)

    @GObject.Property(type=bool, default=True)
    def spinner_visible(self):
        return self.spinner.get_visible()

    @spinner_visible.setter
    def spinner_visible(self, value):
        self.spinner.set_visible(value)

    @label.setter
    def label(self, value):
        self._label.set_label(value)

    def set_label(self, label: str):
        self._label.set_label(label)

    def get_label(self) -> str:
        return self._label.get_label()

    def set_spinner_visible(self, value: bool):
        self.spinner.set_visible(value)