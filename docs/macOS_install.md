## Developer Installation (macOS)
This section provides instructions for installing the lada_cli (No GUI) from source on macOS.

> [!NOTE]
> This is for macOS. If you're on Linux, follow the [Linux Installation](linux_install.md). If you're on Windows, follow the [Windows Installation](windows_install.md).
>
> A standalone CLI build can be created with PyInstaller; see [Building a standalone CLI](#building-a-standalone-cli-optional) below.

### Install CLI

1) Install system dependencies

   * [uv](https://docs.astral.sh/uv/getting-started/installation/) (e.g. `brew install uv`)
   * FFmpeg (e.g. `brew install ffmpeg`)
   * git (e.g. `brew install git`)

   ```bash
   brew install ffmpeg git uv
   ```

2) Get the source code

   ```bash
   git clone https://codeberg.org/ladaapp/lada.git
   cd lada
   ```

3) Create a virtual environment and install Python dependencies

   On macOS, use the **cpu** extra. PyTorch's default CPU build includes **MPS (Metal)** support, so you can use Apple Silicon or Intel Mac GPUs when available.

   ```bash
   uv venv
   source .venv/bin/activate
   uv sync --extra cpu
   ```

   Check that PyTorch sees your GPU (optional; skip if you only want CPU):

   ```bash
   # MPS (Metal) - Apple Silicon or Intel Mac with supported GPU
   uv run --no-project python -c "import torch; print(torch.backends.mps.is_available())"
   ```

   If this prints *True*, you can pass `--device mps` to the CLI for GPU-accelerated inference.

4) Apply patches

   ```bash
   patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/increase_mms_time_limit.patch
   patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/remove_ultralytics_telemetry.patch
   patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
   ```

5) Download model weights

   Download the necessary model weights from HuggingFace:

   ```bash
   curl -L -o model_weights/lada_mosaic_detection_model_v2.pt 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v2.pt?download=true'
   curl -L -o model_weights/lada_mosaic_detection_model_v4_accurate.pt 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_accurate.pt?download=true'
   curl -L -o model_weights/lada_mosaic_detection_model_v4_fast.pt 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_fast.pt?download=true'
   curl -L -o model_weights/lada_mosaic_restoration_model_generic_v1.2.pth 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth?download=true'
   ```

   For DeepMosaics restoration you can also download their pretrained model:

   ```bash
   curl -L -o model_weights/3rd_party/clean_youknow_video.pth 'https://drive.usercontent.google.com/download?id=1ulct4RhRxQp1v5xwEmUH7xz7AK42Oqlw&export=download&confirm=t'
   ```

You can now run the CLI with `lada-cli`.

> [!TIP]
> Remember: To start Lada ensure you:
> * `cd` into the project root directory
> * Activate the virtual environment with `source .venv/bin/activate`
> * Run the CLI with `lada-cli` (use `--device mps` for Metal GPU if available)

### Install Translations (optional)

If you prefer the app in a language other than English:

1) Install gettext

   ```bash
   brew install gettext
   ```

2) Compile translations

   ```bash
   bash translations/compile_po.sh
   ```

The CLI will use translations based on your locale (e.g. `LANG` or `LANGUAGE`).
