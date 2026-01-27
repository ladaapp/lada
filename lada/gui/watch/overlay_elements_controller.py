# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import sys
import threading
import time

from gi.repository import Gtk, GObject, Gdk, Adw


class OverlayElementsController(GObject.Object):
    def __init__(self, watched_widget: Gtk.Widget, overlay_widgets: list[Gtk.Widget], additional_control_widgets: list[Gtk.Widget]):
        GObject.Object.__init__(self)
        self.watched_widget = watched_widget
        self.overlay_widgets: list[Gtk.Widget] = overlay_widgets
        self.activity_timer = None
        self.motion_started_time = None
        self.last_motion_x: float = sys.maxsize
        self.last_motion_y: float = sys.maxsize
        self.idle_time_seconds = 1.5
        self.last_allocation: Gdk.Rectangle = watched_widget.get_allocation()

        self.overlay_motions = set()
        for widget in self.overlay_widgets:
            widget.add_controller(motion := Gtk.EventControllerMotion())
            self.overlay_motions.add(motion)

        self.additional_controls_motions = set()
        for widget in additional_control_widgets:
            widget.add_controller(motion := Gtk.EventControllerMotion())
            self.additional_controls_motions.add(motion)

        self.watched_widget_motion_controller: Gtk.EventControllerMotion = Gtk.EventControllerMotion.new()
        self.watched_widget_motion_controller.connect("enter", self._on_enter)
        self.watched_widget_motion_controller.connect("leave", self._on_leave)
        self.watched_widget_motion_controller.connect("motion", self._on_motion)
        self.watched_widget.add_controller(self.watched_widget_motion_controller)

        self._reveal_animations: dict[Gtk.Widget, Adw.Animation] = {}
        self._hide_animations: dict[Gtk.Widget, Adw.Animation] = {}

    def _on_motion(self, obj, x, y):
        current_allocation = self.watched_widget.get_allocation()
        allocation_changed = self._is_resized(current_allocation)
        if allocation_changed:
            self.last_allocation = current_allocation
            return

        if not self._is_considerable_mouse_motion(x, y):
            return
        self.motion_started_time = time.time()

        self.watched_widget.set_cursor_from_name("default")

        for widget in self.overlay_widgets:
            self._reveal_overlay(widget)

        if self._is_hovering_overlay_or_control_widgets():
            self._cancel_timer()
        else:
            self._start_timer()
        self.last_motion_y = y
        self.last_motion_x = x

    def _on_enter(self, obj, x, y):
        for widget in self.overlay_widgets:
            self._reveal_overlay(widget)

    def _on_leave(self, obj):
        self._cancel_timer()
        for widget in self.overlay_widgets:
            self._hide_overlay(widget)

    def _adjust_overlay_opacity(self, widget: Gtk.Widget, reveal: bool) -> None:
        animations = self._reveal_animations if reveal else self._hide_animations
        animation = animations.get(widget)

        if animation and (animation.props.state == Adw.AnimationState.PLAYING):
            return

        animations[widget] = Adw.TimedAnimation.new(
            widget,
            widget.props.opacity,
            int(reveal),
            250,
            Adw.PropertyAnimationTarget.new(widget, "opacity"),
        )

        widget.props.can_target = reveal
        animations[widget].play()

    def _reveal_overlay(self, widget: Gtk.Widget) -> None:
        self._adjust_overlay_opacity(widget, True)

    def _hide_overlay(self, widget: Gtk.Widget) -> None:
        self._adjust_overlay_opacity(widget, False)

    def _on_activity_timer_run(self, *args):
        self.motion_started_time = None
        self.activity_timer = None
        for widget in self.overlay_widgets:
            self._hide_overlay(widget)

        if self.watched_widget_motion_controller.contains_pointer():
            self.watched_widget.set_cursor_from_name("none")

    def _cancel_timer(self):
        if self.activity_timer:
            self.activity_timer.cancel()

    def _start_timer(self):
        if self.activity_timer:
            self.activity_timer.cancel()
        self.activity_timer = threading.Timer(self.idle_time_seconds, self._on_activity_timer_run)
        self.activity_timer.start()

    def _is_considerable_mouse_motion(self, x, y, min_distance_px=3.) -> bool:
        return abs(x - self.last_motion_x) >= min_distance_px or abs(y - self.last_motion_y) >= min_distance_px

    def _is_hovering_overlay_or_control_widgets(self) -> bool:
        return any(motion.props.contains_pointer for motion in (self.overlay_motions | self.additional_controls_motions))

    def _is_resized(self, current_allocation: Gdk.Rectangle) -> bool:
        return current_allocation.width != self.last_allocation.width or current_allocation.height != self.last_allocation.height
