# -*- mode: python ; coding: utf-8 -*-

import argparse
from PyInstaller.utils.hooks import collect_data_files
from os.path import join as ospj
import shutil
import os
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--gvsbuild-dir", default="C:/project/build")
args = parser.parse_args()

GVSBUILD_DIR = "C:/project/build"
GTK_RELEASE_DIR = os.path.join(args.gvsbuild_dir, "gtk/x64/release")
assert os.path.isdir(GTK_RELEASE_DIR)
LADA_BASE_DIR = str(pathlib.Path(__name__).parent.parent.parent.resolve())

bin_ffmpeg = shutil.which("ffmpeg.exe")
assert bin_ffmpeg is not None
bin_ffprobe = shutil.which("ffprobe.exe")
assert bin_ffprobe is not None

common_datas = [
        (ospj(LADA_BASE_DIR, 'model_weights/lada_mosaic_detection_model_v2.pt'), 'model_weights'),
        (ospj(LADA_BASE_DIR, 'model_weights/lada_mosaic_detection_model_v3.1_accurate.pt'), 'model_weights'),
        (ospj(LADA_BASE_DIR, 'model_weights/lada_mosaic_detection_model_v3.1_fast.pt'), 'model_weights'),
        (ospj(LADA_BASE_DIR, 'model_weights/lada_mosaic_restoration_model_generic_v1.2.pth'), 'model_weights'),
        (ospj(LADA_BASE_DIR, 'model_weights/3rd_party/clean_youknow_video.pth'), 'model_weights/3rd_party'),
]
common_datas += [(str(p), str(p.relative_to(LADA_BASE_DIR).parent)) for p in pathlib.Path(ospj(LADA_BASE_DIR, "lada/locale")).rglob("*.mo")]
gui_datas = []
gui_datas += common_datas
gui_datas += [(str(p), str(p.relative_to(LADA_BASE_DIR).parent)) for p in pathlib.Path(ospj(LADA_BASE_DIR, "lada/gui")).rglob("*.ui")]
gui_datas += [
        (ospj(LADA_BASE_DIR, 'lada/gui/style.css'), '.'),
        (ospj(LADA_BASE_DIR, 'lada/gui/resources.gresource'), '.'),
]
cli_datas = []

common_binaries = [
	(bin_ffmpeg, "bin"),
	(bin_ffprobe, "bin"),
]
gui_binaries = []
gui_binaries += common_binaries
gui_binaries += [
	(ospj(GTK_RELEASE_DIR, "bin/gdbus.exe"), "."),
	# The following fixes runtime warning at startup (AFAIK did not break anything)
	# (process:11556): GLib-GIRepository-CRITICAL **: 11:22:58.188: Unable to load platform-specific GIO introspection data: Typelib file for namespace 'GioWin32' (any version) not found
	(ospj(GTK_RELEASE_DIR, "lib/girepository-1.0/GioWin32-2.0.typelib"), "gi_typelibs"),
]
cli_binaries = []
cli_binaries += common_binaries

common_runtime_hooks = [ospj(LADA_BASE_DIR, "packaging/windows/pyinstaller_runtime_hook_lada.py")]

common_icon = [ospj(LADA_BASE_DIR, 'packaging/flatpak/share/icons/hicolor/128x128/apps/io.github.ladaapp.lada.png')]

gui_a = Analysis(
    [ospj(LADA_BASE_DIR, 'lada/gui/main.py')],
    pathex=[],
    binaries=gui_binaries,
    datas=gui_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={
        "gi": {
            "icons": ["Adwaita"],
            "themes": ["Adwaita"],
            "languages": [],
            "module-versions": {
                "Gtk": "4.0",
            },
        },
    },
    runtime_hooks=common_runtime_hooks,
    excludes=[],
    noarchive=False,
    optimize=0,
)
gui_pyz = PYZ(gui_a.pure)
gui_exe = EXE(
    gui_pyz,
    gui_a.scripts,
    [],
    exclude_binaries=True,
    name='lada',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=common_icon,
)

cli_a = Analysis(
    [ospj(LADA_BASE_DIR, 'lada/cli/main.py')],
    pathex=[],
    binaries=cli_binaries,
    datas=cli_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=common_runtime_hooks,
    excludes=[],
    noarchive=False,
    optimize=0,
)
cli_pyz = PYZ(cli_a.pure)
cli_exe = EXE(
    cli_pyz,
    cli_a.scripts,
    [],
    exclude_binaries=True,
    name='lada-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=common_icon,
)

coll = COLLECT(
    gui_exe,
    gui_a.binaries,
    gui_a.datas,
    cli_exe,
    cli_a.binaries,
    cli_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='lada',
)
