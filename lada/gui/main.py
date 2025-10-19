import logging
import pathlib
import sys

import gi

from lada import VERSION, LOG_LEVEL

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')

from gi.repository import Gtk, Gio, Adw, Gdk, Gst, GObject

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

from lada.gui.window import MainWindow
from lada.gui.shortcuts import ShortcutsWindow, ShortcutsManager
from lada.gui.config.config import Config

class LadaApplication(Adw.Application):

    def __init__(self):
        super().__init__(application_id='io.github.ladaapp.lada',
                         flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
        self.create_action('quit', self.on_close, ['<primary>q'])
        self.create_action('about', self.on_about_action)
        self.create_action('shortcuts', self.on_shortcut_action)

        css_provider = Gtk.CssProvider()
        css_provider.load_from_path(str(here.joinpath('style.css')))
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        resource = Gio.resource_load(str(here.joinpath('resources.gresource')))
        Gio.resources_register(resource)

        self._shortcuts_manager: ShortcutsManager = ShortcutsManager()
        self._config: Config = Config(self.get_style_manager())
        self._config.load_config()
        self.window: MainWindow | None = None

        Gst.init(None)

    @GObject.Property(type=ShortcutsManager)
    def shortcuts_manager(self):
        return self._shortcuts_manager

    @shortcuts_manager.setter
    def shortcuts_manager(self, value):
        self._shortcuts_manager = value

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    def do_activate(self):
        win = self.props.active_window
        if not win:
            win = MainWindow(application=self)
            self.bind_property("style-manager", win.preview_view.widget_timeline, "style-manager", flags=GObject.BindingFlags.SYNC_CREATE)
            self.bind_property("shortcuts-manager", win.preview_view, "shortcuts-manager", flags=GObject.BindingFlags.SYNC_CREATE)
            self.bind_property("config", win.preview_view, "config", flags=GObject.BindingFlags.SYNC_CREATE)
            self.bind_property("config", win.export_view, "config", flags=GObject.BindingFlags.SYNC_CREATE)
            self.bind_property("config", win, "config", flags=GObject.BindingFlags.SYNC_CREATE)
            self.bind_property("shortcuts-manager", win.preview_view, "shortcuts-manager", flags=GObject.BindingFlags.SYNC_CREATE)
            self.bind_property("shortcuts-manager", win, "shortcuts-manager", flags=GObject.BindingFlags.SYNC_CREATE)
            self.window = win

            self._shortcuts_manager.init(win.shortcut_controller)
        win.present()

    def on_close(self, *args):
        if self.window:
            self.window.close()
            self.quit()

    def on_about_action(self, *args):
        about = Adw.AboutDialog(application_name='lada',
                                application_icon='io.github.ladaapp.lada',
                                license_type=Gtk.License.AGPL_3_0,
                                website='https://github.com/ladaapp',
                                issue_url='https://github.com/ladaapp/issues',
                                version=VERSION)
        about.present(self.props.active_window)

    def on_shortcut_action(self, *args):
        shortcuts_window = ShortcutsWindow(self._shortcuts_manager)
        shortcuts_window.show()

    def create_action(self, name, callback, shortcuts=None):
        action = Gio.SimpleAction.new(name, None)
        action.connect("activate", callback)
        self.add_action(action)
        if shortcuts:
            self.set_accels_for_action(f"app.{name}", shortcuts)


def main():
    app = LadaApplication()
    try:
        return app.run(sys.argv)
    except KeyboardInterrupt:
        logger.info("Received Ctrl-C, quitting")
        app.on_close()

if __name__ == "__main__":
    main()
