import os
import shutil
import subprocess
import sys

from gi.repository import Gio, GLib

class ShutdownError(Exception):
    pass

class ShutdownManager:
    def __init__(self):
        self.bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)

    def _call_dbus_method(self, service_name, object_path, interface_name, method_name, parameters=None):
        dbus_proxy = Gio.DBusProxy.new_sync(
            self.bus,
            Gio.DBusProxyFlags.NONE,
            None,
            service_name,
            object_path,
            interface_name,
            None
        )

        response = dbus_proxy.call_sync(
            method_name,
            parameters,
            Gio.DBusCallFlags.NONE,
            1_000,  # 1 sec timeout
            None  # Not cancellable
        )
        return response

    def shutdown_windows(self):
        try:
            subprocess.run(["shutdown", "/s", "/t", "0"], check=True)
        except subprocess.CalledProcessError as e:
            raise ShutdownError(e)

    def shutdown_linux_generic(self):
        try:
            subprocess.run(["shutdown", "now"], check=True)
        except subprocess.CalledProcessError as e:
            raise ShutdownError(e)

    def shutdown_linux_kde(self):
        try:
            self._call_dbus_method(
                "org.kde.Shutdown",
                "/Shutdown",
                "org.kde.Shutdown",
                "logoutAndShutdown"
            )
        except GLib.GError as e:
            raise ShutdownError(e)

    def shutdown_linux_gnome(self):
        try:
            self._call_dbus_method(
                "org.gnome.SessionManager",
                "/org/gnome/SessionManager",
                "org.gnome.SessionManager",
                "Shutdown"
            )
        except GLib.GError as e:
            if e.code == 24: # user clicked Cancel
                return
            print("domain", e.domain, "code", e.code, "message", e.message)
            raise ShutdownError(e)

    def is_service_registered(self, service_name: str) -> bool:
        response = self._call_dbus_method(
            "org.freedesktop.DBus",
            "/org/freedesktop/DBus",
            "org.freedesktop.DBus",
            "ListNames"
        )

        if response:
            names = response.unpack()[0]  # The result is a tuple, and the first item is the list of names
            return service_name in names
        return False

    def shutdown(self):
        linux_desktop_env = os.getenv("XDG_CURRENT_DESKTOP", "").upper()
        linux_session_desktop_env = os.getenv("XDG_SESSION_DESKTOP", "").upper()
        if sys.platform == "win32":
            self.shutdown_windows()
        elif "KDE" in linux_desktop_env or "KDE" in linux_session_desktop_env:
            self.shutdown_linux_kde()
        elif ("GNOME" in linux_desktop_env or "GNOME" in linux_session_desktop_env) and self.is_service_registered("org.gnome.SessionManager"):
            self.shutdown_linux_gnome()
        elif shutil.which("shutdown") is not None:
            self.shutdown_linux_generic()
        else:
            raise ShutdownError("Couldn't find any means to shutdown the system")

if __name__ == "__main__":
    shutdown_manager = ShutdownManager()
    shutdown_manager.shutdown()
