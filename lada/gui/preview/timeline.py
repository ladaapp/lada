import pathlib
from dataclasses import dataclass

from gi.repository import Gtk, GObject, Gdk, Graphene, Gsk, Adw

from lada.gui import utils

here = pathlib.Path(__file__).parent.resolve()

@dataclass
class TimelineColors:
    timeline_color: Gdk.RGBA()
    playhead_color: Gdk.RGBA()
    cursor_color: Gdk.RGBA()

@Gtk.Template(string=utils.translate_ui_xml(here / 'timeline.ui'))
class Timeline(Gtk.Widget):
    __gtype_name__ = 'Timeline'

    @GObject.Property(type=Adw.StyleManager)
    def style_manager(self):
        return self._style_manager

    @style_manager.setter
    def style_manager(self, value):
        self._style_manager = value

    @GObject.Property()
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        self.update_duration(value)
        self.update_playhead_position(0)

    @GObject.Property()
    def playhead_position(self):
        return self._playhead_position

    @playhead_position.setter
    def playhead_position(self, value):
        self.update_playhead_position(value)

    @GObject.Signal(name="seek_requested", arg_types=(GObject.TYPE_INT64,))
    def seek_requested_signal(self, position: int):
        pass

    @GObject.Signal(name="cursor_position_changed", arg_types=(GObject.TYPE_INT64,))
    def cursor_position(self, position: int | None):
        pass

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._playhead_position = 0
        self.cursor_position_x: int | None = None
        self._duration = 0
        self.set_hexpand(True)

        self.gesture_drag = Gtk.GestureDrag.new()
        self.drag_start = 0
        def on_drag_begin(gesture_drag, x, y):
            gesture_drag.set_state( Gtk.EventSequenceState.CLAIMED)
            self.on_drag_begin(x)
        self.gesture_drag.connect("drag-begin", on_drag_begin)
        self.gesture_drag.connect("drag-end", lambda _, offset_x, offset_y: self.on_drag_end(offset_x))
        self.add_controller(self.gesture_drag)

        event_controller_motion = Gtk.EventControllerMotion.new()
        event_controller_motion.connect("leave", lambda _: self.update_cursor_position(None))
        event_controller_motion.connect("motion", lambda _, x, y: self.update_cursor_position(x))
        self.add_controller(event_controller_motion)

        self._style_manager = None

    def update_duration(self, value):
        self._duration = value
        self._playhead_position = 0
        self.queue_draw()

    def update_playhead_position(self, value):
        self._playhead_position = value
        self.queue_draw()

    def on_drag_end(self, offset_x):
        x = self.drag_start + offset_x
        x = max(0, x)
        allocation = self.get_allocation()
        width = allocation.width
        new_position = int((x / width) * self._duration)
        self.update_playhead_position(new_position)
        self.emit('seek_requested', new_position)

    def on_drag_begin(self, x):
        self.drag_start = x

    def update_cursor_position(self, x):
        self.cursor_position_x = x
        if x:
            allocation = self.get_allocation()
            width = allocation.width
            cursor_position = int((x / width) * self._duration)
        else:
            cursor_position = -1
        self.queue_draw()
        self.emit('cursor_position_changed', cursor_position)

    def do_snapshot(self, s: Gtk.Snapshot):
        """
        AFAIK, it's currently only possible to get the accent color programmatically. Other colors need apparently be to be
        hardcoded and pray that it somewhat matches the documented Adwaita colors (unless another theme is used)
        https://gnome.pages.gitlab.gnome.org/libadwaita/doc/main/css-variables.html
        """
        allocation = self.get_allocation()
        width = allocation.width
        height = allocation.height

        playhead_position_x = min(int((self._playhead_position / self._duration) * width), width - 1) if self._duration > 0 else 0

        colors = self.get_timeline_colors()

        cursor_width = 2
        playhead_width = 4
        border_radius = 10

        clip_rect = Graphene.Rect().init(0, 0, width, height)
        rounded_clip_rect = Gsk.RoundedRect()
        rounded_clip_rect.init_from_rect(clip_rect, border_radius)
        s.push_rounded_clip(rounded_clip_rect)

        background_rect = Graphene.Rect().init(0, 0, width, height)
        background_rounded_rect = Gsk.RoundedRect()
        background_rounded_rect.init_from_rect(background_rect, border_radius)
        s.push_rounded_clip(background_rounded_rect)
        s.append_color(colors.timeline_color, background_rect)
        s.pop()

        playhead_rect_x = playhead_position_x - (playhead_width // 2)
        if playhead_rect_x < 0:
            playhead_rect_x = 0
        elif playhead_rect_x + playhead_width > width:
            playhead_rect_x = width - cursor_width
        playhead_rect = Graphene.Rect().init(playhead_rect_x, 0, playhead_width, height)
        s.append_color(colors.playhead_color, playhead_rect)

        if self.cursor_position_x:
            cursor_rect_x = self.cursor_position_x - (cursor_width // 2)
            if cursor_rect_x < 0:
                cursor_rect_x = 0
            elif cursor_rect_x + cursor_width > width:
                cursor_rect_x = width - cursor_width
            cursor_rect = Graphene.Rect().init(cursor_rect_x, 0, cursor_width, height)
            s.append_color(colors.cursor_color, cursor_rect)

        s.pop()

    def get_timeline_colors(self) -> TimelineColors:
        if self._style_manager:
            playhead_color = self._style_manager.get_accent_color()
            uses_dark_scheme = bool(self._style_manager.get_dark())
        else:
            playhead_color = Adw.AccentColor.BLUE
            uses_dark_scheme = False

        # On current libadwaita==1.7.0 / PyGObject==3.52.3 Adw.AccentColor.to_rgba() takes no additional argument,
        # previously one had to pass itself
        try:
            playhead_color = playhead_color.to_rgba()
        except TypeError:
            playhead_color = playhead_color.to_rgba(playhead_color)

        timeline_color = Gdk.RGBA()
        cursor_color = Gdk.RGBA()
        if uses_dark_scheme:
            timeline_color.parse("#ffffff1a")
            cursor_color.parse("#ffffffff")
        else:
            timeline_color.parse("#0000001a")
            cursor_color.parse("#000000ff")

        return TimelineColors(timeline_color, playhead_color, cursor_color)
