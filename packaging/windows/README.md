## System preparation
Prerequisite is an environment set up as described in [Windows build instructions](../../docs/windows_install.md).

## Package a new version

Summary: Setup new temporary venv, checkout lada and install it.

TODO: Like we do for Docker and Flatpak packages we should pin the dependencies

```powershell
$project = "C:\project"
cd $project

git clone https://codeberg.org/ladaapp/lada.git -b v0.8.2 release_lada
cd release_lada

py -m venv .venv
.\.venv\Scripts\Activate.ps1

$env:Path = $project + "\build\gtk\x64\release\bin;" + $project + "\gettext\bin;" + $env:Path
$env:LIB = $project + "\build\gtk\x64\release\lib;" + $env:LIB
$env:INCLUDE = $project + "\build\gtk\x64\release\include;" + $project + "\build\gtk\x64\release\include\cairo;" + $project + "\build\gtk\x64\release\include\glib-2.0;" + $project + "\build\gtk\x64\release\include\gobject-introspection-1.0;" + $project + "\build\gtk\x64\release\lib\glib-2.0\include;" + $env:INCLUDE

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install av==15.1.0
pip install '.[basicvsrpp]'

pip install --force-reinstall (Resolve-Path ($project + "\build\gtk\x64\release\python\pygobject*.whl"))
pip install --force-reinstall (Resolve-Path ($project + "\build\gtk\x64\release\python\pycairo*.whl"))

patch -u .venv/lib/site-packages/ultralytics/utils/ops.py patches/increase_mms_time_limit.patch
patch -u .venv/lib/site-packages/ultralytics/utils/__init__.py  patches/remove_ultralytics_telemetry.patch
patch -u .venv/lib/site-packages/mmengine/runner/checkpoint.py  patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff

Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v3.1_accurate.pt?download=true' -OutFile ".\model_weights\lada_mosaic_detection_model_v3.1_accurate.pt"
Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v3.1_fast.pt?download=true' -OutFile ".\model_weights\lada_mosaic_detection_model_v3.1_fast.pt"
Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v2.pt?download=true' -OutFile ".\model_weights\lada_mosaic_detection_model_v2.pt"
Invoke-WebRequest 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth?download=true' -OutFile ".\model_weights\lada_mosaic_restoration_model_generic_v1.2.pth"
Invoke-WebRequest 'https://drive.usercontent.google.com/download?id=1ulct4RhRxQp1v5xwEmUH7xz7AK42Oqlw&export=download&confirm=t' -OutFile ".\model_weights\3rd_party\clean_youknow_video.pth"

powershell .\translations/compile_po.ps1 --release
```

Just do a quick test `lada`, drop in a video and see if it loads. If all looks good lets continue and create a package:

```powershell
pip install pyinstaller 
```

On my Windows build machine I got a crash of pyinstaller caused by polars dependency pulled in by ultralytics.
Problem seems to be that it expects AVX512 capable CPU which this machine doesn't offer. Fortunately there is an alternative package:

```powershell
pip uninstall polars
pip install polars-lts-cpu
```
Now we can build the package using pyinstaller:

```powershell
pyinstaller ./packaging/windows/lada.spec
```

This will create a `dist` directory in the project root.

Copy this over to another pristine Windows VM for testing if any of the dependencies changed.
This machine would not have any of the environment changes, libraries and binaries of the build machine so it can be used to test if all necessary dependencies are bundled correctly.

If the dependencies did not change (changes just within lada package) then testing on the same build machine should be fine.

If all looks good we can zip it up and upload it to GitHub:

* Open 7-Zip
* Select the `lada` directory within `dist` directory created by pyinstaller.
* Add
* Set `Archive format` to 7z
* Set `Archive filename` to lada-<version>.7z
* Set `Split to volume, bytes` to 1999M # TODO: maybe not necessary now that we're not bound by the 2GB limit of GitHub
* OK

Then calculate the files' checksum

```powershell
Get-FileHash -Algorithm SHA256 .\path\to\lada-<version>.7z
```

Copy the hash into a file named `lada-<version>.sha256.txt`
It should contain a single line with checksum followed by a space and then the .7z file name

Attach these files then to the Release on GitHub (drag-and-drop to the Draft Release)