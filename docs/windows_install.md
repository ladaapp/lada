## Developer Installation (Windows)
This section describes how to install the app (CLI and GUI) from source.

> [!NOTE]
> You may want to use Windows WSL instead of Windows native installation and follow the instructions in the README.md instead.

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
   winget install --id GnuWin32.Patch  -e --source winget
   set-ExecutionPolicy RemoteSigned
   ```
   Then restart your computer

2) Build system dependencies via gvsbuild
   
   Open a PowerShell as regular user and prepare the build environment. You may want to adjust the `$project` and point to another directory of your choice.
   In the following it will be used to build and install system dependencies, and we'll download and install Lada it in this location.
   ```Powershell
   $project = "C:\project"
   mkdir $project
   cd $project
   py -m venv venv_gvsbuild
   .\venv_gvsbuild\Scripts\Activate.ps1
   pip install gvsbuild pip-system-certs
   ```
   
   Now that the `gvsbuild` build environment is set up we can build the remaining system dependencies which we couldn't install via winget.
   Grab a coffee, this will take a while...
   ```Powershell
   gvsbuild build --configuration=release --build-dir='./build' --enable-gi --py-wheel gtk4 adwaita-icon-theme pygobject libadwaita gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-rtsp-server gst-python --extra-opts ogg:-DCMAKE_POLICY_VERSION_MINIMUM=3.5;
   ```
   
> [!TIP]
> If the build fails with
> 
> > AttributeError: module 'distutils' has no attribute 'ccompiler'. Did you mean: 'compilers'?
> 
> add the following line as a workaround
> 
> ```
> import distutils.ccompiler
> ```
> 
> at the beginning of the file
> ```
> .\build\build\x64\release\gobject-introspection\_gvsbuild-meson\giscanner\ccompiler.py
> ```
> 
> then re-run the gvsbuild command again.
   
   Once the build is done prepare your environment variables to include the build artifacts of gvsbuild
   ```Powershell
   $env:Path = $project + "\build\gtk\x64\release\bin;" + $env:Path
   $env:LIB = $project + "\build\gtk\x64\release\lib;" + $env:LIB
   $env:INCLUDE = $project + "\build\gtk\x64\release\include;" + $project + "\build\gtk\x64\release\include\cairo;" + $project + "\build\gtk\x64\release\include\glib-2.0;" + $project + "\build\gtk\x64\release\include\gobject-introspection-1.0;" + $project + "\build\gtk\x64\release\lib\glib-2.0\include;" + $env:INCLUDE
   ```

> [!NOTE]
> These variables need to be set for the next steps but also whenever you start `lada`.

3) Build remaining system dependencies
   
   We need to build the Gstreamer GTK4 plugin (needed for the GUI video player) ourselves as it cannot be built with gvsbuild yet
   ```Powershell
   git clone https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs.git -b 0.13
   cd gst-plugins-rs
   cargo install cargo-c
   cargo cinstall -p gst-plugin-gtk4 --prefix ($project + "\build\gtk\x64\release") --libdir ($project + "\build\gtk\x64\release\lib") 
   ```
   If `gst-inspect-1.0.exe gtk4paintablesink` does not return an error everything went fine.
   
   Congrats! You've setup up all system dependencies at this point so we can now install Lada (and it's python dependencies) itself.

4) Get the source
   
   ```Powershell
   cd $project
   git clone https://github.com/ladaapp/lada.git
   cd lada
   ```
5) Create a virtual environment to install python dependencies
   
    ```Powershell
    py -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

6) [Install PyTorch](https://pytorch.org/get-started/locally)

7) Install python dependencies
   
    ```Powershell
    pip install -e '.[basicvsrpp]'
    ````
   For the GUI we'll need to install the Python wheels we've built earlier with gvsbuild:
    ```Powershell
    pip install --force-reinstall (Resolve-Path ($project + "\build\gtk\x64\release\python\pygobject*.whl"))
    pip install --force-reinstall (Resolve-Path ($project + "\build\gtk\x64\release\python\pycairo*.whl"))
    ````

8) Apply patches
   
   On low-end hardware running mosaic detection model could run into a timeout defined in ultralytics library and the scene would not be restored. The following patch increases this time limit:
    ```bash
    patch -u .venv/lib/site-packages/ultralytics/utils/ops.py patches/increase_mms_time_limit.patch
    ```
   
   Disable crash-reporting / telemetry of one of our dependencies (ultralytics):
   ```bash
   patch -u .venv/lib/site-packages/ultralytics/utils/__init__.py  patches/remove_ultralytics_telemetry.patch
   ```
   
   Compatibility fix for using mmengine (restoration model dependency) with latest PyTorch:
   ```bash
   patch -u .venv/lib/site-packages/mmengine/runner/checkpoint.py  patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
   ```

9) Download model weights
   
   Download the models from the GitHub Releases page into the `model_weights` directory. The following commands do just that
   ```shell
   Invoke-WebRequest 'https://github.com/ladaapp/lada/releases/download/v0.7.1/lada_mosaic_detection_model_v3.1_accurate.pt' -OutFile ".\model_weights\lada_mosaic_detection_model_v3.1_accurate.pt"
   Invoke-WebRequest 'https://github.com/ladaapp/lada/releases/download/v0.7.1/lada_mosaic_detection_model_v3.1_fast.pt' -OutFile ".\model_weights\lada_mosaic_detection_model_v3.1_fast.pt"
   Invoke-WebRequest 'https://github.com/ladaapp/lada/releases/download/v0.2.0/lada_mosaic_detection_model_v2.pt' -OutFile ".\model_weights\lada_mosaic_detection_model_v2.pt"
   Invoke-WebRequest 'https://github.com/ladaapp/lada/releases/download/v0.6.0/lada_mosaic_restoration_model_generic_v1.2.pth' -OutFile ".\model_weights\lada_mosaic_restoration_model_generic_v1.2.pth"
   ```

   If you're interested in running DeepMosaics' restoration model you can also download their pretrained model `clean_youknow_video.pth`
   ```shell
   Invoke-WebRequest 'https://drive.usercontent.google.com/download?id=1ulct4RhRxQp1v5xwEmUH7xz7AK42Oqlw&export=download&confirm=t' -OutFile ".\model_weights\3rd_party\clean_youknow_video.pth"
   ```

    Now you should be able to run the CLI by calling `lada-cli`, and the GUI by `lada`.
