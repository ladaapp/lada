# -*- mode: python ; coding: utf-8 -*-
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
#
# PyInstaller spec for building Lada CLI on macOS.
# Run from project root: pyinstaller --noconfirm packaging/macOS/lada.spec

import os
from os.path import join as ospj
import shutil
import pathlib

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
    return common_datas

project_root = get_project_root()
common_datas = get_common_datas(project_root)
common_binaries = get_common_binaries(project_root)
common_runtime_hooks = [ospj(project_root, "packaging/macOS/pyinstaller_runtime_hook_lada.py")]
common_icon = [ospj(project_root, 'assets/io.github.ladaapp.lada.png')]

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

coll = COLLECT(
    cli_executable,
    cli_a.binaries,
    cli_a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='lada',
)
