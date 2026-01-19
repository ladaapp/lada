## Developer Installation (Windows)
This section provides instructions for installing the app (CLI and GUI) from source on Windows.

> [!NOTE]
> This guide is for Windows. If you're using Linux or WSL, follow the [Linux Installation](linux_install.md).
> 
> Standalone .exe files are available [here](../README.md#using-windows).

### Install CLI

1) Install system dependencies
   
   Open a PowerShell window as Administrator and run the following commands to install required programs via winget:
   ```Powershell
   winget install --id Gyan.FFmpeg -e --source winget
   winget install --id Git.Git -e --source winget
   winget install --id astral-sh.uv -e --source winget
   set-ExecutionPolicy RemoteSigned
   ```
   Close the PowerShell window once the installation is complete.

2) Get the source code

   Open a PowerShell window as a regular user. You will not need an Administrator Shell for any of the remaining steps.

   ```Powershell
   git clone https://codeberg.org/ladaapp/lada.git
   cd lada
   ```

3) Create a virtual environment to install python dependencies
   
   ```Powershell
   uv venv
   .\.venv\Scripts\Activate.ps1
   ```

4) Install Python dependencies
   
   | extra        | supported GPU architectures                                                                                                                                                       | notes                                                              |
   |--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
   | nvida-legacy | Nvidia Maxwell(5.0), Pascal(6.0), Volta(7.0), Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0)                                                                             | for RTX 10xx, CUDA 12.6                                            |
   | nvidia       | Nvidia Volta(7.0), Turing(7.5), Ampere(8.0, 8.6), Hopper(9.0), Blackwell(10.0, 12.0)                                                                                 | for RTX 20xx up to including RTX 50xx, CUDA 12.8                   |
   | intel        | Intel Discrete Arc GPUs: A-series (Alchemist), B-series (Battlemage)<br/>Intel Integrated Arc GPUs of Core Ultra Processors: Meteor Lake-H, Arrow Lake-H, Lunar Lake |                                                                    |
   | cpu          | -                                                                                                                                                                    | running Lada on CPU will be so slow that it's not really practical |

   Based on your hardware, select the appropriate *extra* from the table above and install it with uv.

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

    ```Powershell
    uv pip install patch
    uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/increase_mms_time_limit.patch
    uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/remove_ultralytics_telemetry.patch
    uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
    uv pip uninstall patch
    ````

6) Download model weights
   
   Download the necessary model weights from HuggingFace
   ```Powershell
   Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v2.pt?download=true' -OutFile ".\model_weights\lada_mosaic_detection_model_v2.pt"
   Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_accurate.pt?download=true' -OutFile ".\model_weights\lada_mosaic_detection_model_v4_accurate.pt"
   Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v4_fast.pt?download=true' -OutFile ".\model_weights\lada_mosaic_detection_model_v4_fast.pt"
   Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth?download=true' -OutFile ".\model_weights\lada_mosaic_restoration_model_generic_v1.2.pth"
   ```

   If you're interested in running DeepMosaics' restoration model you can also download their pretrained model `clean_youknow_video.pth`
   ```Powershell
   Invoke-WebRequest 'https://drive.usercontent.google.com/download?id=1ulct4RhRxQp1v5xwEmUH7xz7AK42Oqlw&export=download&confirm=t' -OutFile ".\model_weights\3rd_party\clean_youknow_video.pth"
   ```

You can now run the CLI with `lada-cli`.

> [!TIP]
> Remember: To start Lada ensure you:
> * `cd` into the project root directory
> * Activate the virtual environment with `.\.venv\Scripts\Activate.ps1`
> * Run the CLI with `lada-cli`

### Install GUI

1. Install the CLI as per instructions [above](#install-cli)

2. Install system dependencies

   You can choose between two methods to install system dependencies for the GUI:
   
   #### Option 1: Install pre-compiled dependencies (Recommended)
   
   * Download the pre-compiled system dependencies from [here](https://pixeldrain.com/u/SB1nGZJQ) (`sha256: 171152e8df65556f02065e080ec087153aaaa39634346b3dbe08a4f0f0d3ba1f`).
   * Extract the file and ensure the `build_gtk` folder is located in the project root. The directory should look like this:
     ```
     <lada root>
     ├── pyproject.toml
     ├── LICENSE.md
     ├── lada/
     ├── build_gtk/ # <- extracted from lada_windows_dependencies_python313_gvsbuild202611.7z
     ...
     ```

   #### Option 2: Build system dependencies yourself
  
   * Open a PowerShell window as Administrator and install the required build tools using winget:
     ```Powershell
     winget install --id MSYS2.MSYS2 -e --source winget
     winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --silent --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
     winget install --id Rustlang.Rustup -e --source winget
     winget install --id Microsoft.VCRedist.2013.x64  -e --source winget
     winget install --id Microsoft.VCRedist.2013.x86  -e --source winget
     ```
     Then restart your computer.

   * Open a PowerShell as a regular user and prepare the build envirobnment (You will not need an Administrator Shell for any of the remaining steps)
   
     ```Powershell
     uv venv venv_gtk
     .\venv_gtk\Scripts\Activate.ps1
     uv pip install gvsbuild==2026.1.0
     ```
   
   * Build the dependencies with gvsbuild. Grab a coffee, this will take a while...
   
     ```Powershell
     gvsbuild build --configuration=release --build-dir='./build_gtk' --enable-gi --py-wheel gtk4 adwaita-icon-theme pygobject libadwaita gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-plugin-gtk4 gst-libav gst-python gettext
     ```
   * Once the build is complete, deactivate the build environment and reactivate the project venv:
     ```Powershell
     deactivate
     .\.venv\Scripts\Activate.ps1
     ```

3. Install python dependencies

   Install the Python wheels we've built (or downloaded) in the previous step
    ```Powershell
    uv pip install --force-reinstall (Resolve-Path ".\build_gtk\gtk\x64\release\python\pygobject*.whl")
    uv pip install --force-reinstall (Resolve-Path ".\build_gtk\gtk\x64\release\python\pycairo*.whl")
    ````

> [!TIP]
> If you intend to hack on the GUI code install also the `gui-dev` group: `uv sync --group gui-dev --inexact`.

You can now run the GUI with `lada`.

> [!TIP]
> Remember: To start Lada ensure you:
> * `cd` into the project root directory
> * Activate the virtual environment with `.\.venv\Scripts\Activate.ps1`
> * Run the GUI with `lada`

### Install Translations (optional)

If you prefer the app in a language other than English, you can use translation files if available.

1) Install system dependencies
  
   We'll need the tool gettext. This program is already available if you've installed the GUI so you just have to run this

   ```Powershell
   $env:Path = (Resolve-Path ".\build_gtk\gtk\x64\release\bin").Path + ";" + $env:Path
   $env:LIB = (Resolve-Path ".\build_gtk\gtk\x64\release\lib").Path + ";" + $env:LIB
   ```

   Alternatively, if you've only installed the CLI then it might be the easiest to install gettext this way:

   * Go to [GNU gettext tools for Windows](https://github.com/vslavik/gettext-tools-windows/releases) and download the latest release .zip file.
   * Extract the archive
   * Add the extracted directory to the $PATH environment variable: `$env:Path = "<path/to/gettext/directory>" + ";" + $env:Path`

2) Compile translations
   ```Powershell
   powershell -ExecutionPolicy Bypass .\translations\compile_po.ps1
   ```

GUI and CLI should now show translations (if available) based on your system language settings (*Time & language | Language & region | Windows display language*).

Alternatively, you can set the environment variable `LANGUAGE` to your preferred language e.g. `$env:LANGUAGE = "zh_CN"`. Using Windows settings is the  preferred method though as just setting the environment variable may miss to set up the correct fonts.
