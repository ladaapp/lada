## Developer Installation (Linux)
This section provides instructions for installing the app (CLI and GUI) from source on Linux.

> [!NOTE]
> This is for Linux. If you're on Windows (and don't want to use WSL), follow the [Windows Installation](windows_install.md).
> 
> Flatpak and Docker Images are available [here](../README.md#using-flatpak)

### Install CLI

1) Install system dependencies
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

2) Get the code code
   ```bash
   git clone https://codeberg.org/ladaapp/lada.git
   cd lada
   ```

3) Create a virtual environment to install python dependencies
    ```bash
    uv venv
    source .venv/bin/activate
    ```
4) Install Python dependencies
   
   | extra        | supported GPU architectures                                                                                                                                          | notes                                                              |
   |--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
   | nvida-legacy | Nvidia Maxwell(5.0), Pascal(6.0), Volta(7.0), Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0)                                                                             | for RTX 10xx, CUDA 12.6                                            |
   | nvidia       | Nvidia Volta(7.0), Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0), Blackwell(10.0, 12.0)                                                                                 | for RTX 20xx up to including RTX 50xx, CUDA 12.8                   |
   | intel        | Intel Discrete Arc GPUs: A-series (Alchemist), B-series (Battlemage)<br/>Intel Integrated Arc GPUs of Core Ultra Processors: Meteor Lake-H, Arrow Lake-H, Lunar Lake |                                                                    |
   | cpu          | -                                                                                                                                                                    | running Lada on CPU will be so slow that it's not really practical |

   Based on your hardware, select the correct *extra* from the table above and install it with uv.

   You have to choose a single option in case your system contains GPUs of multiple vendors like an integrated Intel GPU and a dedicated Nvidia GPU.

   ```bash
   uv sync --extra nvidia # Adjust extra according to your available hardware
   ```

   Before continuing let's test if the installation was successful by checking if [PyTorch](https://pytorch.org) detects your GPU (skip if using CPU):

   ```bash
   # Nvidia
   uv run --no-project python -c "import torch ; print(torch.cuda.is_available())"
   # Intel
   uv run --no-project python -c "import torch ; print(torch.xpu.is_available())"
   ```
   
   If this prints *True* then you're good. If *False*, check your GPU drivers are up-to-date and ensure you've selected the right *extra* for your hardware.

5) Apply patches
   
    ```bash
    patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/increase_mms_time_limit.patch
    patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/remove_ultralytics_telemetry.patch
    patch -u -p1 -d .venv/lib/python3.13/site-packages < patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
    ```

6) Download model weights
   
   Download the necessary model weights from HuggingFace
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

You can now run the CLI with `lada-cli`.

> [!TIP]
> Remember: To start Lada ensure you:
> * `cd` into the project root directory
> * Activate the virtual environment with `.\.venv\Scripts\Activate.ps1`
> * Run the CLI with `lada-cli`

### Install GUI

1) Install the CLI as per instructions [above](#install-cli)

2) Install system dependencies
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
> If you intend to hack on the GUI code install also the `gui-dev` group: `uv pip install --group gui-dev`.

You can now run the GUI with `lada`.

> [!TIP]
> Remember: To start Lada ensure you:
> * `cd` into the project root directory
> * Activate the virtual environment with `.\.venv\Scripts\Activate.ps1`
> * Run the GUI with `lada`

### Install Translations (optional)

If you prefer the app in a language other than English, you can use translation files if available.

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