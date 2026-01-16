## Developer Installation (Linux)
This section describes how to install the app (CLI and GUI) from source.

> [!NOTE]
> This is the Linux guide. If you're on Windows (and don't want to use WSL) follow the [Windows Installation](windows_install.md).
> 
> Flatpak and Docker Images are available [here](../README.md#using-flatpak)

### Install CLI

1) Install system dependencies with your system package manager or compile/install from source
   * uv
   * FFmpeg >= 4.4
   * git

> [!TIP]
> Arch Linux: `sudo pacman -Syu uv ffmpeg git`
> 
> Ubuntu 25.04: `sudo apt install ffmpeg git`. `uv` is not yet available in Ubuntu/Debian repositories, see [uv | Getting Started](https://docs.astral.sh/uv/getting-started/installation/) for alternative installation methods
> 
> Ubuntu 24.04: `sudo apt install ffmpeg git`. `uv` is not yet available in Ubuntu/Debian repositories, see [uv | Getting Started](https://docs.astral.sh/uv/getting-started/installation/) for alternative installation methods

> [!TIP]
> If you have an Intel GPU and want to use QSV hardware video encoding you'll need to install [Intel VPL GPU Runtime](https://github.com/intel/vpl-gpu-rt).
> 
> Arch Linux: `sudo pacman -S vpl-gpu-rt`

2) Get the code
   ```bash
   git clone https://codeberg.org/ladaapp/lada.git
   cd lada
   ```

3) Create a virtual environment to install python dependencies
    ```bash
    uv venv
    source .venv/bin/activate
    ```

4) [Install PyTorch](https://pytorch.org/get-started/locally)

> [!TIP]
> If you're using some older hardware you should check out [RELEASE.md](https://github.com/pytorch/pytorch/blob/main/RELEASE.md). It contains support and compatibility information for all official PyTorch builds helping you to decide which version you need to install.
> 
> If your hardware is not supported in the latest release you might need to choose an older PyTorch version (You can select a specific release tag on GitHub to see an older version of that document).

> [!TIP]
> Instead of `pip install ...` as it's documented by PyTorch use `uv pip install ...`. Alternatively you can use `uv pip install torch torchvision --torch-backend auto` to let uv take care of choosing the correct PyTorch installation for your system and available hardware.

> [!TIP]
> Before continuing let's test if the PyTorch installation was successful by checking if your GPU is detected (Skip this step if you're running on CPU)
> ```bash
> uv run --no-project python -c "import torch ; print(torch.cuda.is_available())"
> ```
> If this prints *True* then you're good. It will display *False* if the GPU is not available to PyTorch. Check your GPU drivers and that you chose the correct PyTorch Installation method for your hardware.


5) Install python dependencies
    ```bash
    uv pip install -e '.'
    ````

6) Apply patches
   
    ```bash
    patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/increase_mms_time_limit.patch
    patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/remove_ultralytics_telemetry.patch
    patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
    ```

7) Download model weights
   
   Download the models from the GitHub Releases page into the `model_weights` directory. The following commands do just that
   ```shell
   wget 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v2.pt?download=true' -O model_weights/lada_mosaic_detection_model_v2.pt
   wget 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_accurate.pt?download=true' -O model_weights/lada_mosaic_detection_model_v4_accurate.pt
   wget 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_fast.pt?download=true' -O model_weights/lada_mosaic_detection_model_v4_fast.pt
   wget 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth?download=true' -O model_weights/lada_mosaic_restoration_model_generic_v1.2.pth
   ```

   If you're interested in running DeepMosaics' restoration model you can also download their pretrained model `clean_youknow_video.pth`
   ```shell
   wget 'https://drive.usercontent.google.com/download?id=1ulct4RhRxQp1v5xwEmUH7xz7AK42Oqlw&export=download&confirm=t' -O model_weights/3rd_party/clean_youknow_video.pth
   ```

Now you should be able to run the CLI by calling `lada-cli`.

> [!TIP]
> Remember: To start Lada always make sure to:
> * `cd` into the project root directory
> * Activate the virtual environment via `source .venv/bin/activate`
> * Run `lada-cli` to start the CLI

### Install GUI

1) Install everything mentioned in [Install CLI](#install-cli)

2) Install system dependencies with your system package manager or compile/install from source
   * Gstreamer >= 1.14
   * PyGObject
   * GTK >= 4.0
   * libadwaita >= 1.6 [there is a workaround mentioned below to make it work with older versions]

> [!TIP]
> Arch Linux: 
> ```bash
> sudo pacman -Syu python-gobject gtk4 libadwaita gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-plugins-base-libs gst-plugins-bad-libs gst-plugin-gtk4
> ```
>   
> Ubuntu 25.04:
> ```bash
> sudo apt install gcc python3-dev pkg-config libgirepository-2.0-dev libcairo2-dev libadwaita-1-dev gir1.2-gstreamer-1.0
> sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-pulseaudio gstreamer1.0-alsa gstreamer1.0-tools gstreamer1.0-libav gstreamer1.0-gtk4
> ```
> 
> Ubuntu 24.04:
> ```bash
> sudo apt install gcc python3-dev pkg-config libgirepository-2.0-dev libcairo2-dev libadwaita-1-dev gir1.2-gstreamer-1.0
> sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-pulseaudio gstreamer1.0-alsa gstreamer1.0-tools gstreamer1.0-libav
> #
> ####### Gstreamer #######
> # The gstreamer plugin gtk4 is not available as a binary package in Ubuntu 24.04 so we have to build it ourselves:
> # Get the source code
> git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git
> cd gst-plugins-rs
> # Install dependencies necessary to build the plugin
> sudo apt  install rustup libssl-dev
> rustup default stable
> cargo install cargo-c
> # Now we can build and install the plugin. Note that we're installing to the system directory, you might want to adjust this to another directory and set set the environment variable GST_PLUGIN_PATH accordingly
> cargo cbuild -p gst-plugin-gtk4 --libdir /usr/lib/x86_64-linux-gnu
> sudo -E cargo cinstall -p gst-plugin-gtk4 --libdir /usr/lib/x86_64-linux-gnu
> # If the following command does not return an error the plugin is correctly installed
> gst-inspect-1.0 gtk4paintablesink
> #
> ####### libadwaita #######
> # The version of libadwaita in Ubuntu 24.04 repositories is too old. Instead of building the new version the following patch will adjust the code so it's compatible with the version provided by Ubuntu 24.04:
> patch -u -p1 -i patches/adw_spinner_to_gtk_spinner.patch
> > ```

3) Install python dependencies
    ```bash
    uv pip install -e '.[gui]'
    ````

> [!TIP]
> If you intend to hack on the GUI code install also the `gui-dev` group (`--group gui-dev`).

Now you should be able to run the GUI by calling `lada`.

> [!TIP]
> Remember: To start Lada always make sure to:
> * `cd` into the project root directory
> * Activate the virtual environment via `source .venv/bin/activate`
> * Run `lada` to start the GUI

### Install Translations (optional)

If we have a translation file for your language you might want to use it instead of using the app in English.

1) Install system dependencies

> [!TIP]
> Arch Linux: `sudo pacman -Syu gettext`
> 
> Ubuntu: `sudo apt install gettext`

2) Compile translations
    ```bash
    bash translations/compile_po.sh
    ```

The app should now use the translations and be shown in your system language. If not then you may need to set the environment variable
`LANG` (or `LANGAUGE`) to your preferred language e.g. `export LANGUAGE="zh_TW"`.