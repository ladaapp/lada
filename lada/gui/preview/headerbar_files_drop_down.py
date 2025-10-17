import logging
import pathlib

from gi.repository import Gtk, Gio, GObject, Pango

from lada import LOG_LEVEL

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)


@Gtk.Template(filename=here / 'headerbar_files_drop_down.ui')
class HeaderbarFilesDropDown(Gtk.DropDown):
    __gtype_name__ = "HeaderbarFilesDropDown"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Custom drop down factory so we can limit the dropdown button width and ellipse the selected item in the header bar if they are too long
        button_factory = Gtk.SignalListItemFactory()
        button_factory.connect("setup", self.on_button_item_setup)
        button_factory.connect("bind", self.on_item_bind)
        button_factory.connect("unbind", self.on_item_unbind)

        # Custom drop down factory for popup items shown when dropdown is clicked without width restrictions
        popup_factory = Gtk.SignalListItemFactory()
        popup_factory.connect("setup", self.on_popup_item_setup)
        popup_factory.connect("bind", self.on_item_bind)
        popup_factory.connect("unbind", self.on_item_unbind)

        expression = Gtk.ClosureExpression.new(
            GObject.TYPE_STRING,
            lambda obj: obj.get_string(),
            None,
        )

        model = Gtk.StringList()
        self.props.model = model
        self.props.expression = expression
        self.props.factory = button_factory
        self.props.list_factory = popup_factory

    def on_popup_item_setup(self, factory, list_item):
        label = Gtk.Label()
        list_item.set_child(label)

    def on_button_item_setup(self, factory, list_item):
        label = Gtk.Label()
        label.set_ellipsize(Pango.EllipsizeMode.END)
        label.set_max_width_chars(20)
        list_item.set_child(label)

    def on_item_bind(self, factory, list_item):
        label = list_item.get_child()
        idx = list_item.get_position()

        label.set_text(self.props.model[idx].get_string())

    def on_item_unbind(self, factory, list_item):
        label = list_item.get_child()
        if label:
            label.set_text("")

    def add_files(self, files: list[Gio.File]):
        for file in files:
            self.props.model.append(file.get_basename())

        is_multiple_files = len(self.props.model) > 1
        self.set_enable_search(is_multiple_files)
        self.set_sensitive(is_multiple_files)
        self.set_show_arrow(is_multiple_files)