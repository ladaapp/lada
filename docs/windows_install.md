## Developer Installation (Windows)
This section describes how to install the app (CLI and GUI) from source.

> [!NOTE]
> This is the Windows guide. If you're on Linux (or want to use WSL) follow the [Linux Installation](linux_install.md).
> 
> Standalone .exe files are available [here](../README.md#using-windows)

### Install CLI

1) Download and install system dependencies
   
   Open a PowerShell as Administrator and install the following programs via winget
   ```Powershell
   winget install --id Gyan.FFmpeg -e --source winget
   winget install --id Git.Git -e --source winget
   winget install --id astral-sh.uv -e --source winget
   set-ExecutionPolicy RemoteSigned
   ```
   Then close this PowerShell window

2) Get the source

   Open a PowerShell as a regular user. You will not need an Administrator Shell for any of the remaining steps.

   ```Powershell
   git clone https://codeberg.org/ladaapp/lada.git
   cd lada
   ```

3) Create a virtual environment to install python dependencies
   
   ```Powershell
   uv venv
   .\.venv\Scripts\Activate.ps1
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
> ```shell
> uv run --no-project python -c "import torch ; print(torch.cuda.is_available())"
> ```
> If this prints *True* then you're good. It will display *False* if the GPU is not available to PyTorch. Check your GPU drivers and that you chose the correct PyTorch Installation method for your hardware.


> [!TIP]
> For AMD Radeon users only:
> 
> At the time of writing this, there are currently no official torch builds offered by PyTorch but AMD offers their own torch builds for Windows that you can use.
> 
> Note that using ROCm on Radeon cards on Windows is still in preview and there are known issues.
> You can find which cards are compatible and how to install PyTorch [here](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html).
> 
> You might also want to read about and try the latest nightly/experimental PyTorch builds from AMD [here](https://github.com/ROCm/TheRock/blob/main/RELEASES.md).
> 
> You might want to consider using Linux or WSL as PyTorch on ROCm is supposedly more stable there at the moment.

5) Install python dependencies
   
    ```Powershell
    uv pip install -e '.'
    ````

6) Apply patches

    ```Powershell
    uv pip install patch
    uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/increase_mms_time_limit.patch
    uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/remove_ultralytics_telemetry.patch
    uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
    uv pip uninstall patch
    ````

> [!TIP]
> For AMD Radeon users only:
> 
> At the time of writing this, neither the latest stable/preview build 6.4 nor the latest daily build 7.1 includes support for `torch.dist`
> 
> One of our dependencies (mmengine) uses it internally and will crash if `torch.dist` is not available. You can use the following patch to work around that and make Lada work regardless:
> ```Powershell
> uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/remove_use_of_torch_dist_in_mmengine.patch
> ```
> If you're reading this and AMD included `torch.dist` in their builds please create a Pull Request or create an issue to update this tip.

7) Download model weights
   
   Download the models from HuggingFace into the `model_weights` directory. The following commands do just that
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

Now you should be able to run the CLI by calling `lada-cli`.

> [!TIP]
> Remember: To start Lada always make sure to:
> * `cd` into the project root directory
> * Activate the virtual environment via `.\.venv\Scripts\Activate.ps1`
> * Run `lada-cli` to start the CLI

### Install GUI

1. Install everything mentioned in [Install CLI](#install-cli)

2. Install system dependencies

   Choose only one of these two steps in order to install system dependencies necessary to run the GUI.
   The first option to use pre-compiled dependencies is recommended as it's faster and less error-prone than building them yourself.
   
   1. Download and install system dependencies
   
      As compiling system dependencies and setting up the build system can take quite some time pre-compiled dependencies are available.
   
      First download the file [lada_windows_dependencies_python313_gvsbuild202611.7z](https://pixeldrain.com/u/SB1nGZJQ) (`sha256: 171152e8df65556f02065e080ec087153aaaa39634346b3dbe08a4f0f0d3ba1f`).
   
      Then extract it. Make sure that the extracted `build_gtk` folder is now located in the project root. Your directory should look like this:
   
      ```
      <lada root>
      ├── pyproject.toml
      ├── LICENSE.md
      ├── lada/
      ├── build_gtk/ # <- extracted from lada_windows_dependencies_python313_gvsbuild202611.7z
      ...
      ```

   2. Build and install system dependencies via gvsbuild

      Instead of using the pre-compiled dependencies you can build them yourself on your own machine.
      
      First, download and install the build dependencies:
      
      Open a PowerShell as Administrator and install the following programs via winget
      ```Powershell
      winget install --id MSYS2.MSYS2 -e --source winget
      winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --silent --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
      winget install --id Rustlang.Rustup -e --source winget
      winget install --id Microsoft.VCRedist.2013.x64  -e --source winget
      winget install --id Microsoft.VCRedist.2013.x86  -e --source winget
      ```
      Then restart your computer.

      Open a PowerShell as a regular user. You will not need an Administrator Shell for any of the remaining steps.
      
      Prepare the build environment
      ```Powershell
      uv venv venv_gtk
      .\venv_gtk\Scripts\Activate.ps1
      uv pip install gvsbuild==2026.1.0
      ```
      
      Now we can start building the dependencies with `gvsbuild`. Grab a coffee, this will take a while...
      
      ```Powershell
      gvsbuild build --configuration=release --build-dir='./build_gtk' --enable-gi --py-wheel gtk4 adwaita-icon-theme pygobject libadwaita gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-plugin-gtk4 gst-libav gst-python gettext
      ```
      
      Congrats! If this command finished successfully you've set up all system dependencies so we can now continue installing Lada and it's python dependencies.
      
      Let's exit the gvsbuild build environment and re-activate the project venv
      ```Powershell
      deactivate
      .\.venv\Scripts\Activate.ps1
      ```

4. Install python dependencies

   Install the Python wheels we've built (or downloaded) in the previous step
    ```Powershell
    uv pip install --force-reinstall (Resolve-Path ".\build_gtk\gtk\x64\release\python\pygobject*.whl")
    uv pip install --force-reinstall (Resolve-Path ".\build_gtk\gtk\x64\release\python\pycairo*.whl")
    ````

> [!TIP]
> If you intend to hack on the GUI code install also the `gui-dev` group (`--group gui-dev`).

Now you should be able to run the GUI by calling `lada`.

> [!TIP]
> Remember: To start Lada always make sure to:
> * `cd` into the project root directory
> * Activate the virtual environment via `.\.venv\Scripts\Activate.ps1`
> * Run `lada` to start the GUI

### Install Translations (optional)

If we have a translation file for your language you might want to use it instead of using the app in English.

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
