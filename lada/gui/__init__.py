import os
import pathlib
import sys

is_running_pyinstaller_bundle = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

def _update_env_var(env_var, paths, separator=";"):
    assert sys.platform == "win32", "_update_env_var() only works on Windows with case-insensitive paths"
    paths_to_add = [str(path).lower() for path in paths]
    if env_var in os.environ:
        existing_paths = os.environ[env_var].lower().split(separator)
        paths_to_add = [path for path in paths_to_add if path not in existing_paths]
        os.environ[env_var] = separator.join(paths_to_add + existing_paths)
    else:
        os.environ[env_var] = separator.join(paths_to_add)

def prepare_windows_gui_environment():
    gvsbuild_build_dir = pathlib.Path(__file__).parent.parent.parent.joinpath("build_gtk").absolute()
    if not gvsbuild_build_dir.is_dir():
        return

    release_dir = gvsbuild_build_dir / "gtk" / "x64" / "release"
    if not release_dir.exists():
        return

    bin_dir = release_dir / "bin"
    lib_dir = release_dir / "lib"
    includes = [
        release_dir / "include",
        release_dir / "include" / "cairo",
        release_dir / "include" / "glib-2.0",
        release_dir / "include" / "gobject-introspection-1.0",
        release_dir / "lib" / "glib-2.0" / "include",
    ]

    _update_env_var("PATH", [bin_dir])
    _update_env_var("LIB", [lib_dir])
    _update_env_var("INCLUDE", includes)

if sys.platform == "win32" and not is_running_pyinstaller_bundle:
    prepare_windows_gui_environment()

def prepare_macos_gui_environment() -> None:
    """Set up environment for GUI on macOS (e.g. XDG_DATA_DIRS for GSettings, etc.)."""
    assert sys.platform == "darwin", "prepare_macos_gui_environment() only runs on macOS"
    import subprocess
    try:
        result = subprocess.run(
            ["brew", "--prefix"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return
    if result.returncode != 0 or not result.stdout:
        return
    brew_share = os.path.join(result.stdout.strip(), "share")
    if not os.path.isdir(brew_share):
        return
    if "XDG_DATA_DIRS" not in os.environ:
        os.environ["XDG_DATA_DIRS"] = brew_share + os.pathsep + "/usr/share:/usr/local/share"
    elif brew_share not in os.environ["XDG_DATA_DIRS"].split(os.pathsep):
        os.environ["XDG_DATA_DIRS"] = brew_share + os.pathsep + os.environ["XDG_DATA_DIRS"]

if sys.platform == "darwin":
    prepare_macos_gui_environment()
