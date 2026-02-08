# -*- mode: python ; coding: utf-8 -*-
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import argparse
from PyInstaller.utils.hooks import collect_data_files
from os.path import join as ospj
import shutil
import os
import sys
import pathlib
import fnmatch

def get_project_root() -> str:
    project_root = pathlib.Path(".").absolute()
    assert (project_root / "pyproject.toml").exists(), "This script must be run from the root of the project"
    return str(project_root)

# Intel XPU
def get_intel_xpu_runtime_libs(project_root):
    if BUILD_EXTRA != "intel":
        return []

    print("-> Scanning for Intel XPU libraries...")
    venv_root = pathlib.Path(project_root) / "venv_release_win"
    found_binaries = []

    patterns = [
        "ur_win*.dll", "UR_LOADER.dll", "ur_adapter_level_zero.dll"
    ]
    
    if venv_root.exists():
        for p_file in venv_root.rglob("*.dll"):
            if any(fnmatch.fnmatch(p_file.name.lower(), pat.lower()) for pat in patterns):
                found_binaries.append((str(p_file), "."))
    
    return found_binaries

def _update_env_var(env_var, paths, separator=";"):
    assert sys.platform == "win32", "_update_env_var() only works on Windows"
    paths_to_add = [str(path).lower() for path in paths]
    if env_var in os.environ:
        existing_paths = os.environ[env_var].lower().split(separator)
        paths_to_add = [path for path in paths_to_add if path not in existing_paths]
        os.environ[env_var] = separator.join(paths_to_add + existing_paths)
    else:
        os.environ[env_var] = separator.join(paths_to_add)

def set_environment_variables(project_root_dir: str):
    release_dir = (pathlib.Path(project_root_dir) / "build_gtk_release" / "gtk" / "x64" / "release").absolute()

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

    bin_gdbus = shutil.which("gdbus.exe")
    assert bin_gdbus is not None, "gdbus.exe not found. Did gvsbuild successfully build GTK libs?"

def get_common_binaries(project_root):
    bin_ffmpeg = shutil.which("ffmpeg.exe")
    assert bin_ffmpeg is not None, "ffmpeg.exe not found"
    bin_ffprobe = shutil.which("ffprobe.exe")
    assert bin_ffprobe is not None, "ffprobe.exe not found"

    common_binaries = [
        (bin_ffmpeg, "bin"),
        (bin_ffprobe, "bin"),
    ]

    common_binaries += get_intel_xpu_runtime_libs(project_root)

    return common_binaries

def get_common_datas(project_root: str):
    common_datas = [
        (ospj(project_root, 'model_weights/lada_mosaic_detection_model_v2.pt'), 'model_weights'),
        (ospj(project_root, 'model_weights/lada_mosaic_detection_model_v4_accurate.pt'), 'model_weights'),
        (ospj(project_root, 'model_weights/lada_mosaic_detection_model_v4_fast.pt'), 'model_weights'),
        (ospj(project_root, 'model_weights/lada_mosaic_restoration_model_generic_v1.2.pth'), 'model_weights'),
        (ospj(project_root, 'model_weights/3rd_party/clean_youknow_video.pth'), 'model_weights/3rd_party'),
        (ospj(project_root, 'lada/utils/encoding_presets.csv'), 'lada/utils'),
    ]
    common_datas += [(str(p), str(p.relative_to(project_root).parent)) for p in pathlib.Path(ospj(project_root, "lada/locale")).rglob("*.mo")]
    return common_datas

def get_gui_components(project_root_dir: str, common_datas: list, common_binaries: list, common_runtime_hooks: list, common_icon):
    gui_datas = common_datas + [
        (str(p), str(p.relative_to(project_root_dir).parent)) for p in (pathlib.Path(project_root_dir) / "lada" / "gui").rglob("*.ui")
    ] + [
        (ospj(project_root_dir, 'lada/gui/style.css'), 'lada/gui'),
        (ospj(project_root_dir, 'lada/gui/resources.gresource'), 'lada/gui'),
        (ospj(project_root_dir, 'assets/io.github.ladaapp.lada.png'), 'share/icons/hicolor/128x128/apps'),
    ]

    gtk_release_dir = pathlib.Path(project_root_dir) / "build_gtk_release" / "gtk" / "x64" / "release"
    gtk_bin_dir = gtk_release_dir / "bin"

    gui_binaries = common_binaries + [
        (str(gtk_release_dir / "bin" / "gdbus.exe"), "."),
        (str(gtk_release_dir / "lib" / "girepository-1.0" / "GioWin32-2.0.typelib"), "gi_typelibs"),
    ]

    # Intel
    if BUILD_EXTRA == "intel":
        print("-> [Intel] Copying GTK DLLs to root for compatibility...")
        gui_binaries += [(str(p), ".") for p in gtk_bin_dir.glob("*.dll")]

    gui_a = Analysis(
        [ospj(project_root_dir, 'lada/gui/main.py')],
        pathex=[],
        binaries=gui_binaries,
        datas=gui_datas,
        hiddenimports=[],
        hookspath=[],
        hooksconfig={
            "gi": {
                "icons": ["Adwaita"],
                "themes": ["Adwaita"],
                "module-versions": { "Gtk": "4.0" },
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
        icon=common_icon,
    )
    return gui_a, gui_pyz, gui_exe

def get_cli_components(project_root_dir: str, common_datas: list, common_binaries: list, common_runtime_hooks: list, common_icon):
    cli_a = Analysis(
        [ospj(project_root_dir, 'lada/cli/main.py')],
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
        icon=common_icon,
    )
    return cli_a, cli_pyz, cli_exe

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli-only", action="store_true", help="Only build the CLI, skipping the GUI.")
    
    parser.add_argument(
        "--extra", 
        default="nvidia", 
        choices=["nvidia", "intel"],
        help="The installation extra/variant to build for (matches pyproject.toml extras)."
    )
    return parser.parse_args()

args = parser_args()
BUILD_EXTRA = args.extra.lower()

project_root = get_project_root()

if not args.cli_only:
    set_environment_variables(project_root)

common_datas = get_common_datas(project_root)

common_binaries = get_common_binaries(project_root)

common_runtime_hooks = [ospj(project_root, "packaging/windows/pyinstaller_runtime_hook_lada.py")]
common_icon = [ospj(project_root, 'assets/io.github.ladaapp.lada.png')]

cli_a, cli_pyz, cli_exe = get_cli_components(project_root, common_datas, common_binaries, common_runtime_hooks, common_icon)
coll = COLLECT(
    cli_exe,
    cli_a.binaries,
    cli_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='lada',
)

if args.cli_only:
    coll = COLLECT(
        cli_exe,
        cli_a.binaries,
        cli_a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='lada',
    )
else:
    gui_a, gui_pyz, gui_exe = get_gui_components(project_root, common_datas, common_binaries, common_runtime_hooks, common_icon)
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