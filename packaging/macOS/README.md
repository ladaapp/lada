# Build Lada CLI on macOS

This directory contains packaging for building a standalone **lada-cli** executable on macOS using PyInstaller.

## Prerequisites

* Python 3.12+ with [uv](https://github.com/astral-sh/uv) (or pip)
* **ffmpeg** and **ffprobe** on `PATH` (e.g. `brew install ffmpeg`)
* Project dependencies installed (e.g. `uv sync --extra cpu` for CPU/MPS, or another extra)
* PyInstaller: `uv pip install pyinstaller` (or `pip install pyinstaller`)

Optional: place model weights under `model_weights/` so they are bundled. If any are missing, the build still succeeds; only existing files are included.

## Build

From the **project root**:

```bash
uv sync --extra cpu --no-default-groups   # or your chosen extra
uv pip install pyinstaller
uv run pyinstaller --noconfirm packaging/macOS/lada.spec
```

Output is under `dist/lada/`:

* `dist/lada/lada-cli` — CLI executable
* `dist/lada/_internal/` — bundled Python, dependencies, and data

## Run

```bash
./dist/lada/lada-cli --help
./dist/lada/lada-cli --input video.mp4 --output out.mp4 --encoding-preset hevc-apple-gpu-balanced
```

## Notes

* Only the **CLI** is packaged; there is no macOS GUI bundle in this spec.
* Model weights are optional at build time; include them for a self-contained bundle.
* The spec includes only model files that exist, so the build works without downloading every weight.
