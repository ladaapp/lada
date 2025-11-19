import sys

from lada.gui import utils

is_running_pyinstaller_bundle = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

if sys.platform != "win32" and not  is_running_pyinstaller_bundle:
    utils.prepare_windows_gui_environment()