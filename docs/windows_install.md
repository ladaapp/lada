## Developer Installation (Windows)
This section describes how to install the app (CLI and GUI) from source.

> [!NOTE]
> This is the Windows guide. If you're on Linux (or want to use WSL) follow the [Linux Installation](linux_install.md).

1) Download and install system dependencies
   
   Open a PowerShell as Administrator and install the following programs via winget
   ```Powershell
   winget install --id Gyan.FFmpeg -e --source winget
   winget install --id Git.Git -e --source winget
   winget install --id Python.Python.3.13 -e --source winget
   winget install --id MSYS2.MSYS2 -e --source winget
   winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --silent --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
   winget install --id Rustlang.Rustup -e --source winget
   winget install --id Microsoft.VCRedist.2013.x64  -e --source winget
   winget install --id Microsoft.VCRedist.2013.x86  -e --source winget
   set-ExecutionPolicy RemoteSigned
   ```
   Then restart your computer

2) Get the source

   Open a PowerShell as a regular user. You will not need an Administrator Shell for any of the remaining steps.
   
   Create project directory
   ```Powershell
   $project = "C:\project"
   mkdir $project
   ```
> [!NOTE]
> You may want to adjust `$project` and point to another directory of your choice.
> In the following it will be used to build and install system dependencies, and we'll download and install Lada in this location.
   
   Get the source
   ```Powershell
   cd $project
   git clone https://codeberg.org/ladaapp/lada.git
   ```

3) Build system dependencies via gvsbuild
   
   Prepare build environment
   ```Powershell
   cd $project
   py -m venv venv_gvsbuild
   .\venv_gvsbuild\Scripts\Activate.ps1
   pip install gvsbuild==2025.10
   pip install patch
   python -m patch -p1 -d venv_gvsbuild/lib/site-packages lada/patches/gvsbuild_gstreamer_gtk4_plugin.patch
   python -m patch -p1 -d venv_gvsbuild/lib/site-packages lada/patches/gvsbuild_ffmpeg.patch
   pip uninstall patch
   ```
   
   Now we can start building the remaining system dependencies with `gvsbuild` which we couldn't install via `winget`.
   
   Grab a coffee, this will take a while...
   ```Powershell
   gvsbuild build --configuration=release --build-dir='./build' --enable-gi --py-wheel gtk4 adwaita-icon-theme pygobject libadwaita gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-plugin-gtk4 gst-libav gst-python gettext
   ```
   
   Congrats! If this command finished successfully you've set up all system dependencies so we can now continue installing Lada and it's python dependencies.
   
   Let's exit the gvsbuild build environment
   ```Powershell
   deactivate
   ```

4) Setup environment variables   

   Prepare your environment variables so the build artifacts can be found and used
   ```Powershell
   $env:Path = $project + "\build\gtk\x64\release\bin;" + $env:Path
   $env:LIB = $project + "\build\gtk\x64\release\lib;" + $env:LIB
   $env:INCLUDE = $project + "\build\gtk\x64\release\include;" + $project + "\build\gtk\x64\release\include\cairo;" + $project + "\build\gtk\x64\release\include\glib-2.0;" + $project + "\build\gtk\x64\release\include\gobject-introspection-1.0;" + $project + "\build\gtk\x64\release\lib\glib-2.0\include;" + $env:INCLUDE
   ```

> [!NOTE]
> These variables need to be set for the next steps but also whenever you start `lada`!

> [!TIP]
> You may want to set these environment variables permanently via `Start | Edit environment variables for your account | Environment variables...`

> [!TIP]
> Before continuing check if the variables are set up correctly
> ```Powershell
> gst-inspect-1.0.exe gtk4paintablesink
> ```
> If this does not return an error but prints *Plugin Details* (lots of text) then we're good.

5) Create a virtual environment to install python dependencies
   
   ```Powershell
   cd $project
   cd lada
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

6) [Install PyTorch](https://pytorch.org/get-started/locally)

7) Install python dependencies
   
    ```Powershell
    pip install -e '.[basicvsrpp]'
    ````
   
   For the GUI we'll need to install the Python wheels we've built earlier with gvsbuild
    ```Powershell
    pip install --force-reinstall (Resolve-Path ($project + "\build\gtk\x64\release\python\pygobject*.whl"))
    pip install --force-reinstall (Resolve-Path ($project + "\build\gtk\x64\release\python\pycairo*.whl"))
    ````

8) Apply patches

    ```Powershell
    pip install patch
    ````
   
   On low-end hardware running mosaic detection model could run into a timeout defined in ultralytics library and the scene would not be restored. The following patch increases this time limit:
    ```shell
    python -m patch -p1 -d .venv/lib/site-packages patches/increase_mms_time_limit.patch
    ```
   
   Disable crash-reporting / telemetry of one of our dependencies (ultralytics):
   ```shell
   python -m patch -p1 -d .venv/lib/site-packages patches/remove_ultralytics_telemetry.patch
   ```
   
   Compatibility fix for using mmengine (restoration model dependency) with latest PyTorch:
   ```shell
   python -m patch -p1 -d .venv/lib/site-packages patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
   ```
    ```Powershell
    pip uninstall patch
    ````

9) Download model weights
   
   Download the models from HuggingFace into the `model_weights` directory. The following commands do just that
   ```Powershell
   Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v3.1_accurate.pt?download=true' -OutFile ".\model_weights\lada_mosaic_detection_model_v3.1_accurate.pt"
   Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v3.1_fast.pt?download=true' -OutFile ".\model_weights\lada_mosaic_detection_model_v3.1_fast.pt"
   Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v2.pt?download=true' -OutFile ".\model_weights\lada_mosaic_detection_model_v2.pt"
   Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth?download=true' -OutFile ".\model_weights\lada_mosaic_restoration_model_generic_v1.2.pth"
   ```

   If you're interested in running DeepMosaics' restoration model you can also download their pretrained model `clean_youknow_video.pth`
   ```Powershell
   Invoke-WebRequest 'https://drive.usercontent.google.com/download?id=1ulct4RhRxQp1v5xwEmUH7xz7AK42Oqlw&export=download&confirm=t' -OutFile ".\model_weights\3rd_party\clean_youknow_video.pth"
   ```

    Now you should be able to run the CLI by calling `lada-cli`, and the GUI by `lada`.

> [!TIP]
> As mentioned earlier, make sure the environment variables are set up correctly and that you're in the right directory.
> 
> For quick access, here is a snippet to start the GUI:
> ```Powershell
> $project = "C:\project"
> cd $project
> cd lada
> $env:Path = $project + "\build\gtk\x64\release\bin;" + $env:Path
> $env:LIB = $project + "\build\gtk\x64\release\lib;" + $env:LIB
> $env:INCLUDE = $project + "\build\gtk\x64\release\include;" + $project + "\build\gtk\x64\release\include\cairo;" + $project + "\build\gtk\x64\release\include\glib-2.0;" + $project + "\build\gtk\x64\release\include\gobject-introspection-1.0;" + $project + "\build\gtk\x64\release\lib\glib-2.0\include;" + $env:INCLUDE
> .\.venv\Scripts\Activate.ps1
> lada.exe
> ```


10) Install translations (optional)

    If we have a translation file for your language you might want to use Lada in your preferred language instead of English.
    
    For this, we'll need to compile the translation files so the app can use them:
    
    ```Powershell
    .\translations/compile_po.ps1
    ```
    
    GUI and CLI should now show translations (if available) based on your system language settings (*Time & language | Language & region | Windows display language*).

    Alternatively, you can set the environment variable `LANGUAGE` to your preferred language e.g. `$env:LANGUAGE = "zh_CN"`. Using Windows settings is the  preferred method though as just setting the environment variable may miss to set up the correct fonts.