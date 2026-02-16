# Build Lada on macOS

This directory contains packaging for building standalone **lada-cli** and **Lada.app** (GUI) on macOS using PyInstaller.

## Prerequisites

* Python 3.12+ with [uv](https://github.com/astral-sh/uv) (or pip)
* **ffmpeg** and **ffprobe** on `PATH` (e.g. `brew install ffmpeg`)
* Project dependencies installed (e.g. `uv sync --extra cpu` for CPU/MPS, or another extra)
* PyInstaller: `uv pip install pyinstaller` (or `pip install pyinstaller`)

**For the GUI .app** you also need GTK 4, Libadwaita, and GStreamer installed (the .app uses system libs at runtime):

```bash
brew install gtk4 libadwaita adwaita-icon-theme gstreamer
```

And the GUI Python extra when building:

```bash
uv sync --extra cpu --extra gui --no-default-groups
```

Optional: place model weights under `model_weights/` so they are bundled. If any are missing, the build still succeeds; only existing files are included.

## Build

The spec sets `XDG_DATA_DIRS` on macOS so PyInstaller can find GIR/typelib (GLib, Gio, etc.) when building the GUI. If you still see "Could not find GIR file" errors, set it before running:

```bash
export XDG_DATA_DIRS="$(brew --prefix)/share"
```

From the **project root**:

```bash
uv sync --extra cpu --extra gui --no-default-groups   # include gui for Lada.app
uv pip install pyinstaller
uv run pyinstaller --noconfirm packaging/macOS/lada.spec
```

Output:

* **CLI:** `dist/cli/` — contains `lada-cli` and `_internal/`
* **GUI:** `dist/Lada.app` — macOS app bundle (PyInstaller's `BUNDLE` step; macOS only). The `dist/gui/` COLLECT folder is removed after the build.

## Run

**CLI:**

```bash
./dist/cli/lada-cli --help
./dist/cli/lada-cli --input video.mp4 --output out.mp4 --encoding-preset hevc-apple-gpu-balanced
```

**GUI:** double-click `dist/Lada.app` in Finder, or from Terminal:

```bash
open dist/Lada.app
```

The .app expects Homebrew's GTK/GStreamer. Bundle dylibs use rpath (set in spec). Data paths (model weights, locale) come from Info.plist `LSEnvironment` (macOS expands `@executable_path` at launch). The runtime hook sets PATH and, for frozen builds, `LADA_MODEL_WEIGHTS_DIR` and `LOCALE_DIR`.

## Notes

* **GUI excludes matplotlib** so the bundle has no ft2font (avoids crash during GStreamer plugin scan on macOS). GUI does not use it at runtime.
* Model weights are optional at build time; include them for a self-contained bundle.
* The spec includes only model files that exist, so the build works without downloading every weight.
* This spec is macOS-only and exits if not run on darwin.
* **macOS app bundle:** PyInstaller's one-dir build (`COLLECT`) produces `dist/gui/`. The spec uses the **BUNDLE** target (see [Spec File Options for a macOS Bundle](https://pyinstaller.org/en/stable/spec-files.html#spec-file-options-for-a-macos-bundle)), which wraps that folder as `Lada.app` with `Contents/Info.plist` and the expected structure. That is the standard way to produce a macOS app with PyInstaller.
