# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import pathlib

import cv2
from gi.repository import Gtk, GObject, Gdk, Graphene, Gsk, Adw, GLib, GdkPixbuf

from lada.gui import utils
from lada.utils import Image

here = pathlib.Path(__file__).parent.resolve()

@Gtk.Template(string=utils.translate_ui_xml(here / 'seek_preview_popover.ui'))
class SeekPreviewPopover(Gtk.Popover):
    __gtype_name__ = 'SeekPreviewPopover'

    label: Gtk.Label = Gtk.Template.Child()
    spinner: Gtk.Spinner = Gtk.Template.Child()
    picture: Gtk.Picture = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_text(self, text: str):
        self.label.set_text(text)

    def show_spinner(self):
        self.spinner.set_visible(True)
        self.spinner.start()

    def hide_spinner(self):
        self.spinner.stop()
        self.spinner.set_visible(False)

    def set_thumbnail(self, thumbnail: Image):
        # Convert BGR to RGB for GdkPixbuf
        rgb_thumbnail = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)

        # Create pixbuf from bytes in memory
        height, width, channels = rgb_thumbnail.shape
        pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
            GLib.Bytes.new(rgb_thumbnail.tobytes()),
            GdkPixbuf.Colorspace.RGB,
            False,  # has_alpha
            8,  # bits_per_sample
            width,
            height,
            width * channels
        )

        def update_ui():
            self.picture.set_pixbuf(pixbuf)
            self.hide_spinner()
            return False

        GLib.idle_add(update_ui)

    def clear_thumbnail(self):
        GLib.idle_add(lambda: self.picture.set_pixbuf(None))

