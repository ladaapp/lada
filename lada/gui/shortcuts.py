# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from gi.repository import GObject, Gtk

class ShortcutsManager(GObject.Object):
    def __init__(self):
        GObject.Object.__init__(self)
        self.groups = {}
        self.group_titles = {}

    def add(self, group_key, action_key, keyboard_trigger, action, action_title):
        self.groups[group_key][action_key] = (keyboard_trigger, action, action_title)

    def register_group(self, group_key, group_title):
        self.group_titles[group_key] = group_title
        if group_key not in self.groups:
            self.groups[group_key] = {}

    def init(self, shortcut_controller: Gtk.ShortcutController):
        for group_key in self.groups:
            shortcuts = self.groups.get(group_key)
            if not shortcuts:
                continue

            for action_key in shortcuts:
                keyboard_trigger, action, _ = shortcuts[action_key]
                gtk_trigger = Gtk.ShortcutTrigger.parse_string(keyboard_trigger)
                gtk_action = Gtk.CallbackAction.new(action)
                shortcut = Gtk.Shortcut.new(gtk_trigger, gtk_action)
                shortcut_controller.add_shortcut(shortcut)

class ShortcutsWindow(Gtk.ShortcutsWindow):
    def __init__(self, shortcuts_manager: ShortcutsManager):
        Gtk.ShortcutsWindow.__init__(self)
        self.shortcuts_manager = shortcuts_manager
        self.set_modal(True)
        self.populate()

    def populate(self):
        section = Gtk.ShortcutsSection()
        section.show()
        for group_key in self.shortcuts_manager.groups:
            shortcuts = self.shortcuts_manager.groups.get(group_key)
            if not shortcuts:
                continue

            group = Gtk.ShortcutsGroup(title=self.shortcuts_manager.group_titles[group_key])
            group.show()
            for action_key in shortcuts:
                keyboard_trigger, _, action_title = shortcuts[action_key]
                short = Gtk.ShortcutsShortcut(title=action_title, accelerator=keyboard_trigger)
                short.show()
                group.add_shortcut(short)
            section.add_group(group)

        self.add_section(section)