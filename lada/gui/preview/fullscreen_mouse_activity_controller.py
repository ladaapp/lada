import sys
import threading
import time

from gi.repository import Gtk, GObject, Gdk


class FullscreenMouseActivityController(GObject.Object):
    def __init__(self, fullscreen_widget: Gtk.Widget, video_widget: Gtk.Widget):
        GObject.Object.__init__(self)
        self.fullscreen_widget: Gtk.Widget = fullscreen_widget
        self.video_widget: Gtk.Widget = video_widget
        self.fullscreen_motion_controller: Gtk.EventControllerMotion | None = None
        self.video_motion_controller: Gtk.EventControllerMotion | None = None
        self.activity_timer = None
        self.motion_started_time = None
        self.last_motion_x: float = 0.
        self.last_motion_y: float = 0.
        self._fullscreen_activity = False
        self.idle_time_seconds = 1.5
        self.fullscreen_activated = None
        self.last_allocation: Gdk.Rectangle = video_widget.get_allocation()

    @GObject.Property()
    def fullscreen_activity(self):
        return self._fullscreen_activity

    @fullscreen_activity.setter
    def fullscreen_activity(self, value):
        self._fullscreen_activity = value

    def on_fullscreened(self, fullscreened: bool):
        self.fullscreen_activated  = time.time()
        if fullscreened:
            self.fullscreen_motion_controller = Gtk.EventControllerMotion.new()
            self.fullscreen_motion_controller.connect("motion", self._on_motion)
            self.fullscreen_widget.add_controller(self.fullscreen_motion_controller)

            self.video_motion_controller = Gtk.EventControllerMotion.new()
            self.video_motion_controller.connect("enter", self._on_enter)
            self.video_motion_controller.connect("leave", self._on_leave)
            self.video_widget.add_controller(self.video_motion_controller)

            self.last_motion_y = sys.maxsize
            self.last_motion_y = sys.maxsize
            self._start_timer()
        else:
            self.fullscreen_widget.remove_controller(self.fullscreen_motion_controller)
            self.video_widget.remove_controller(self.video_motion_controller)
            if self.activity_timer:
                self.activity_timer.cancel()
                self.activity_timer = None

    def on_activity_timer_run(self, *args):
        self.motion_started_time = None
        self.activity_timer = None
        self.fullscreen_activity = False

    def _on_motion(self, obj, x, y):
        current_allocation = self.video_widget.get_allocation()
        allocation_changed = current_allocation.width != self.last_allocation.width or current_allocation.height != self.last_allocation.height
        if allocation_changed:
            self.last_allocation = current_allocation
            return

        if not self._is_considerable_mouse_motion(x, y):
            return
        self.motion_started_time = time.time()

        if not self._fullscreen_activity:
            self.fullscreen_activity = True

        if self.video_motion_controller.props.contains_pointer:
            self._start_timer()
        self.last_motion_y = y
        self.last_motion_x = x

    def _on_enter(self, obj, x, y):
        self._start_timer()

    def _on_leave(self, obj):
        if self.activity_timer:
            self.activity_timer.cancel()

    def _start_timer(self):
        if self.activity_timer:
            self.activity_timer.cancel()
        self.activity_timer = threading.Timer(self.idle_time_seconds, self.on_activity_timer_run)
        self.activity_timer.start()

    def _is_considerable_mouse_motion(self, x, y, min_distance_px=3.):
        return abs(x - self.last_motion_x) >= min_distance_px or abs(y - self.last_motion_y) >= min_distance_px