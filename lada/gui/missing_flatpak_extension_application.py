# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pathlib

import gi

gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

from gi.repository import Gtk, Gio, Adw, Gdk

here = pathlib.Path(__file__).parent.resolve()

from lada import VERSION, LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class MissingFlatpakExtensionApplication(Adw.Application):

    def __init__(self):
        super().__init__(application_id='io.github.ladaapp.lada',
                         flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
        self.create_action("quit", lambda *_: self.quit(), ("<primary>q",))
        self.create_action('about', self.on_about_action)

        css_provider = Gtk.CssProvider()
        css_provider.load_from_path(str(here.joinpath('style.css')))
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

        resource = Gio.resource_load(str(here.joinpath('resources.gresource')))
        Gio.resources_register(resource)

    def on_about_action(self, *args):
        about = Adw.AboutDialog(application_name='Lada',
                                application_icon='io.github.ladaapp.lada',
                                license_type=Gtk.License.AGPL_3_0,
                                website='https://codeberg.org/ladaapp/lada',
                                issue_url='https://codeberg.org/ladaapp/lada/issues',
                                version=VERSION)
        about.present(self.props.active_window)

    def create_action(self, name, callback, shortcuts=None):
        action = Gio.SimpleAction.new(name, None)
        action.connect("activate", callback)
        self.add_action(action)
        if shortcuts:
            self.set_accels_for_action(f"app.{name}", shortcuts)

    def do_activate(self):
        win = self.props.active_window
        if not win:
            win = Adw.ApplicationWindow(application=self, title="Hello World")
            win.set_title("Lada")
            win.set_default_size(900, 550)

            status_page = Adw.StatusPage()
            status_page.props.title = _("No GPU Add-On installed")
            status_page.props.description = _("In order to use the application you need to install one of Lada's Flatpak Add-Ons that is compatible with your hardware")
            status_page.set_icon_name("exclamation-mark-symbolic")

            header_bar = Adw.HeaderBar()
            menu_button = Gtk.MenuButton()
            menu_button.set_icon_name("open-menu-symbolic")
            menu = Gio.Menu()
            menu.append(_("About Lada"), "app.about")
            menu_button.set_menu_model(menu)
            header_bar.pack_end(menu_button)

            toolbar_view = Adw.ToolbarView()
            toolbar_view.add_top_bar(header_bar)
            toolbar_view.set_content(status_page)

            win.set_content(toolbar_view)
        win.present()

    def on_shutdown(self, *_args) -> None:
        for win in self.get_windows():
            if isinstance(win, Gtk.Window):
                win.close()
