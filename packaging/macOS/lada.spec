# -*- mode: python ; coding: utf-8 -*-
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
#
# PyInstaller spec for building Lada CLI and GUI .app on macOS.
# Run from project root: pyinstaller --noconfirm packaging/macOS/lada.spec
# Output: dist/cli/ (lada-cli + _internal/), dist/Lada.app. dist/gui/ is removed after build.

import os
import sys
from os.path import join as ospj
import pathlib
import shutil
import subprocess

if sys.platform != "darwin":
    sys.stderr.write("This spec is for macOS only. Run on darwin.\n")
    sys.exit(1)

def get_macos_homebrew_path():
    """Return Homebrew prefix for GTK/Pango (harfbuzz, fontconfig) and theora (libtheoraenc). GStreamer is required and pulls in theora, so libtheoraenc is always available."""
    try:
        r = subprocess.run(["brew", "--prefix"], capture_output=True, text=True, timeout=5, check=True)
        prefix = pathlib.Path(r.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, OSError):
        return None
    return prefix

def _ensure_macos_gir_path() -> None:
    """Set XDG_DATA_DIRS so PyInstaller can find GIR/typelib when collecting gi (macOS + Homebrew)."""
    brew_prefix = get_macos_homebrew_path()
    if brew_prefix is None:
        return
    brew_share = os.path.join(brew_prefix, "share")
    if not os.path.isdir(brew_share):
        return
    existing = os.environ.get("XDG_DATA_DIRS", "")
    if brew_share not in existing.split(os.pathsep):
        os.environ["XDG_DATA_DIRS"] = brew_share + (os.pathsep + existing if existing else "")


_ensure_macos_gir_path()

def get_project_root() -> str:
    project_root = pathlib.Path(".").absolute()
    assert (project_root / "pyproject.toml").exists(), "This script must be run from the root of the project"
    return str(project_root)

def get_common_binaries(project_root):
    bin_ffmpeg = shutil.which("ffmpeg")
    assert bin_ffmpeg is not None, "ffmpeg not found (install via e.g. Homebrew)"
    bin_ffprobe = shutil.which("ffprobe")
    assert bin_ffprobe is not None, "ffprobe not found (install via e.g. Homebrew)"
    return [
        (bin_ffmpeg, "bin"),
        (bin_ffprobe, "bin"),
    ]

def get_common_datas(project_root: str):
    weight_candidates = [
        (ospj(project_root, 'model_weights/lada_mosaic_detection_model_v2.pt'), 'model_weights'),
        (ospj(project_root, 'model_weights/lada_mosaic_detection_model_v4_accurate.pt'), 'model_weights'),
        (ospj(project_root, 'model_weights/lada_mosaic_detection_model_v4_fast.pt'), 'model_weights'),
        (ospj(project_root, 'model_weights/lada_mosaic_restoration_model_generic_v1.2.pth'), 'model_weights'),
        (ospj(project_root, 'model_weights/3rd_party/clean_youknow_video.pth'), 'model_weights/3rd_party'),
    ]
    common_datas = [(src, dest) for src, dest in weight_candidates if os.path.isfile(src)]
    common_datas.append((ospj(project_root, 'lada/utils/encoding_presets.csv'), 'lada/utils'))
    common_datas += [
        (str(p), str(p.relative_to(project_root).parent))
        for p in pathlib.Path(ospj(project_root, "lada/locale")).rglob("*.mo")
    ]
    # GUI assets (required for Lada.app; optional for CLI-only builds)
    gui_assets = common_datas + [
        (str(p), str(p.relative_to(project_root).parent)) for p in (pathlib.Path(project_root) / "lada" / "gui").rglob("*.ui")
    ] + [
        (ospj(project_root, 'lada/gui/style.css'), 'lada/gui'),
        (ospj(project_root, 'lada/gui/resources.gresource'), 'lada/gui'),
    ]
    common_datas += [(src, dest) for src, dest in gui_assets if os.path.isfile(src)]
    return common_datas

project_root = get_project_root()
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from lada import VERSION
common_datas = get_common_datas(project_root)
common_binaries = get_common_binaries(project_root)
common_runtime_hooks = [ospj(project_root, "packaging/macOS/pyinstaller_runtime_hook_lada.py")]
common_icon = ospj(project_root, 'assets/io.github.ladaapp.lada.png')

# ----- CLI -----
cli_a = Analysis(
    [ospj(project_root, 'lada/cli/main.py')],
    pathex=[],
    binaries=common_binaries,
    datas=common_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=common_runtime_hooks,
    excludes=[],
    noarchive=False,
    optimize=0,
)
cli_pyz = PYZ(cli_a.pure)
cli_executable = EXE(
    cli_pyz,
    cli_a.scripts,
    exclude_binaries=True,
    name='lada-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=common_icon,
)
cli_coll = COLLECT(
    cli_executable,
    cli_a.binaries,
    cli_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='cli',
)

# ----- GUI .app (macOS only) -----
# Requires: brew install gtk4 libadwaita adwaita-icon-theme gstreamer
# macOS: use same libs as uv run lada; gi module-versions forces GTK 4 (built-in default is 3.0).
gui_hooksconfig = {
    'gi': {'module-versions': {'Gtk': '4.0', 'Gdk': '4.0'}},
    'gstreamer': {'exclude_plugins': ['gtk']},
}
gui_hookspath = [ospj(project_root, "packaging/macOS")]
gui_a = Analysis(
    [ospj(project_root, 'lada/gui/main.py')],
    pathex=[],
    binaries=common_binaries,
    datas=common_datas,
    hookspath=gui_hookspath,
    hiddenimports=[
        'gi',
        'gi.repository.Gtk',
        'gi.repository.Adw',
        'gi.repository.Gio',
        'gi.repository.GLib',
        'gi.repository.GObject',
        'gi.repository.Gdk',
        'gi.repository.Gst',
        'gi.repository.GstApp',
        'gi.repository.GdkPixbuf',
        'gi.repository.Graphene',
        'gi.repository.Gsk',
        'gi.repository.Pango',
    ],
    hooksconfig=gui_hooksconfig,
    runtime_hooks=common_runtime_hooks,
    excludes=[
        'matplotlib'
    ],
    noarchive=False,
    optimize=0,
)
# find_binary_dependencies() (during Analysis) can add SYMLINKs for dylibs in subdirs; normalize_toc() then
# keeps SYMLINK over our BINARY. We replace cv2/PIL/theoraenc dylibs with Homebrew copies via
# _replace_dylib_with_homebrew so the bundle uses harfbuzz, fontconfig, libtheoraenc from Homebrew.
homebrew_prefix = get_macos_homebrew_path()
homebrew_lib_path = os.path.join(homebrew_prefix, "lib") if homebrew_prefix else None

def _replace_dylib_with_homebrew(dest_name, source_path, kind):
    basename = os.path.basename(source_path)
    if not basename.endswith(".dylib") or homebrew_lib_path is None:
        return (dest_name, source_path, kind)
    dest = dest_name.replace(".dylibs", "__dot__dylibs")
    if basename.startswith("libfontconfig."):
        replacing_path = os.path.join(homebrew_lib_path, "libfontconfig.dylib")
        if os.path.isfile(replacing_path):
            return (dest, os.path.realpath(replacing_path), kind)
    elif basename.startswith("libtheoraenc."):
        replacing_path = os.path.join(homebrew_lib_path, "libtheoraenc.dylib")
        if os.path.isfile(replacing_path):
            return (dest, os.path.realpath(replacing_path), kind)
    elif basename.startswith("libharfbuzz."):
        replacing_path = os.path.join(homebrew_lib_path, "libharfbuzz.dylib")
        if os.path.isfile(replacing_path):
            return (dest, os.path.realpath(replacing_path), kind)
    return (dest_name, source_path, kind)

gui_a.binaries[:] = [_replace_dylib_with_homebrew(*item) for item in gui_a.binaries]
gui_pyz = PYZ(gui_a.pure)
gui_executable = EXE(
    gui_pyz,
    gui_a.scripts,
    [],
    exclude_binaries=True,
    name='Lada',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    icon=common_icon,
)
gui_coll = COLLECT(
    gui_executable,
    gui_a.binaries,
    gui_a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='gui',
)
gui_app = BUNDLE(
    gui_coll,
    name='Lada.app',
    icon=common_icon,
    bundle_identifier='io.github.ladaapp.lada',
    info_plist={
        'CFBundleShortVersionString': VERSION,
        'CFBundleVersion': VERSION,
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.13',
        'LSEnvironment': {
            'LADA_MODEL_WEIGHTS_DIR': '@executable_path/../Resources/model_weights',
            'LOCALE_DIR': '@executable_path/../Resources/lada/locale',
        },
    },
)

def _remove_gui_collect_folder():
    """Remove dist/gui/ after build; BUNDLE already produced Lada.app from it."""
    dist_dir = os.path.join(project_root, "dist")
    gui_folder = os.path.join(dist_dir, "gui")
    if os.path.isdir(gui_folder):
        shutil.rmtree(gui_folder)

import atexit
atexit.register(_remove_gui_collect_folder)
