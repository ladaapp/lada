# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

$global:PYINSTALLER_VERSION = "6.16.0"
$global:GVSBUILD_VERSION = "2025.11.1"
$global:PYTHON_VERSION = "3.13"
$global:UV_VERSION = "0.9.10"

function Ask-YesNo {
    param([Parameter(Mandatory)] [string]$Question)

    while ($true) {
        $response = Read-Host "$Question (Y/N)"

        switch ($response.ToUpper()) {
            'Y' { return $true }
            'N' { return $false }
            default { Write-Host "Please enter Y or N." -ForegroundColor Yellow }
        }
    }
}

function Install-SystemDependencies {
    Write-Host "Installing system dependencies..."

    winget install --id Gyan.FFmpeg -e --source winget
    winget install --id Git.Git -e --source winget
    winget install --id=astral-sh.uv -e --source winget --version $global:UV_VERSION --force
    winget install --id MSYS2.MSYS2 -e --source winget
    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --silent --override "--wait --quiet --add ProductLang En-us --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    winget install --id Rustlang.Rustup -e --source winget
    winget install --id Microsoft.VCRedist.2013.x64  -e --source winget
    winget install --id Microsoft.VCRedist.2013.x86  -e --source winget
    winget install --id=7zip.7zip -e --source winget
}

function Build-SystemDependencies {
    param([Parameter(Mandatory)] [boolean]$clean)

    Write-Host "Building system dependencies..."

    uv venv --clear --python $global:PYTHON_VERSION venv_gtk_release
    .\venv_gtk_release\Scripts\Activate.ps1

    uv pip install gvsbuild==$global:GVSBUILD_VERSION
    uv pip install patch
    uv run --no-project python -m patch -p1 -d venv_gtk_release/lib/site-packages patches/gvsbuild_ffmpeg.patch
    uv pip uninstall patch

    $cleanArgument = if ($clean) { '--clean' } else { '' }

    gvsbuild build `
        --configuration=release `
        $cleanArgument `
        --build-dir='./build_gtk_release' `
        --enable-gi `
        --py-wheel `
        gtk4 adwaita-icon-theme pygobject libadwaita gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-plugins-ugly gst-plugin-gtk4 gst-libav gst-python gettext

    deactivate
}

function Compile-Translations {
    Write-Host "Compiling translations..."
    # Adjust PATH to include gettext tools used in translation compile script
    $path_backup = $env:Path
    $gtk_bin_dir = (Resolve-Path ".\build_gtk_release\gtk\x64\release\bin").Path
    $env:Path = $gtk_bin_dir + ";" + $env:Path

    & ./translations/compile_po.ps1 --release

    $env:Path = $path_backup
}

function Download-ModelWeights {
    Write-Host "Downloading model weights..."

    function Is-Downloaded($file_name, $sha256) {
        $path = ".\model_weights\" + $file_name
        return (Test-Path $path) -And ((Get-FileHash -Algorithm SHA256 $path).Hash -eq $sha256)
    }

    function Download($url, $file_name, $sha256) {
        if (Is-Downloaded $file_name $sha256) {
            return
        }
        Invoke-WebRequest $url -OutFile (".\model_weights\" + $file_name)
        if (!(Is-Downloaded $file_name $sha256)) {
            Write-Warning "Error downloading " + $url
            exit 1
        }
    }

    Download 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v3.1_accurate.pt?download=true' "lada_mosaic_detection_model_v3.1_accurate.pt" "2b6e5d6cd5a795a4dcc1205b817a7323a4bd3725cef1a7de3a172cb5689f0368"
    Download 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v3.1_fast.pt?download=true' "lada_mosaic_detection_model_v3.1_fast.pt" "25d62894c16bba00468f3bcc160360bb84726b2f92751b5e235578bf2f9b0820"
    Download 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_detection_model_v2.pt?download=true' "lada_mosaic_detection_model_v2.pt" "056756fcab250bcdf0833e75aac33e2197b8809b0ab8c16e14722dcec94269b5"
    Download 'https://huggingface.co/ladaapp/lada/resolve/main/lada_mosaic_restoration_model_generic_v1.2.pth?download=true' "lada_mosaic_restoration_model_generic_v1.2.pth" "d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee"
    Download 'https://drive.usercontent.google.com/download?id=1ulct4RhRxQp1v5xwEmUH7xz7AK42Oqlw&export=download&confirm=t' "3rd_party\clean_youknow_video.pth" "5643ca297c13920b8ffd39a0d85296e494683a69e5e8204d662653d24c582766"
}

function Install-PythonDependencies {
    Write-Host "Installing Python dependencies..."

    uv venv --clear --python $global:PYTHON_VERSION venv_release_win
    .\venv_release_win\Scripts\Activate.ps1

    $release_dir = (Resolve-Path ".\build_gtk_release\gtk\x64\release").Path

    uv pip install --no-deps --requirement packaging/windows/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
    # Fix crash due to polars package requiring AVX512 CPU which isn't available on my build machine (use legacy version)
    # This dependency is not used by Lada. It gets pulled in by ultralytics which uses it outside of inferencing paths
    uv pip uninstall polars
    uv pip install polars-lts-cpu

    uv pip install --force-reinstall (Resolve-Path ".\build_gtk_release\gtk\x64\release\python\pygobject*.whl").Path
    uv pip install --force-reinstall (Resolve-Path ".\build_gtk_release\gtk\x64\release\python\pycairo*.whl").Path

    uv pip install --no-deps '.[gui]'

    uv pip install pyinstaller==$global:PYINSTALLER_VERSION

    uv pip install patch
    uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/increase_mms_time_limit.patch
    uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/remove_ultralytics_telemetry.patch
    uv run --no-project python -m patch -p1 -d .venv/lib/site-packages patches/fix_loading_mmengine_weights_on_torch26_and_higher.diff
    uv pip uninstall patch

    deactivate
}

function Create-EXE {
    param([Parameter(Mandatory)] [boolean]$cli_only)

    Write-Host "Creating executable..."

    .\venv_release_win\Scripts\Activate.ps1

    $release_dir = (Resolve-Path ".\build_gtk_release\gtk\x64\release").Path
    $env:Path = $release_dir + "\bin;" + $env:Path
    $env:LIB = $release_dir + "\lib;" + $env:LIB
    $env:INCLUDE = $release_dir + "\include;" + $release_dir + "\include\cairo;" + $release_dir + "\include\glib-2.0;" + $release_dir + "\include\gobject-introspection-1.0;" + $release_dir + "\lib\glib-2.0\include;" + $env:INCLUDE

    $cli_only_arg = if ($cli_only) { '--cli-only' } else { '' }

    uv run --no-project pyinstaller --noconfirm ./packaging/windows/lada.spec -- $cli_only_arg

    deactivate
}

function Create-7ZArchive {
    Write-Host "Creating 7z archive..."

    .\venv_release_win\Scripts\Activate.ps1
    $version = $(uv run --no-project python -c 'import lada; print(lada.VERSION)')
    deactivate

    $env:Path = ($env:Programfiles + "\7-Zip;") + $env:Path

    $archive_path = "./dist/" + "lada-" + $version + ".7z"
    $tmp_archive_path = "./dist/" + "lada-" + $version + ".tmp.7z"

    # Delete files from prior runs
    Get-ChildItem "./dist" -filter "*.7z*" | ForEach-Object {
        rm $_.FullName
    }

    # Create single-file .7z archive
    7z.exe a $tmp_archive_path "./dist/lada/*"

    # Split .7z archive into 2GB chunks so they can be uploaded to GitHub Releases
    7z.exe a -v1999m $archive_path "./dist/lada/*"

    mv $tmp_archive_path $archive_path

    Get-ChildItem "./dist" -filter "*.7z*" | ForEach-Object {
        $sha256 = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLower()
        echo ($sha256 + " " + $_.Basename + $_.Extension) > ($_.FullName + ".sha256")
    }
}

function Check-ProjectRoot {
    if (!(Test-Path ".\pyproject.toml")) {
        Write-Warning "This script needs to be run from the project root directory."
        exit 1
    }
}

# ---------------------
# EXECUTE PACKAGING STEPS
# ---------------------

$ErrorActionPreference = "Stop"

Check-ProjectRoot

if ($args -notcontains "--skip-winget") {
    Install-SystemDependencies
    if (!(Ask-YesNo "Installing/Upgrading winget programs finished. Check the winget install output above. You may want to stop and restart this script in a new shell for certain installs/updates. Do you want to continue?")) {
        exit 0
    }
}
if (($args -notcontains "--skip-gvsbuild") -Or ($args -notcontains "--cli-only")) {
    Build-SystemDependencies ($args -contains "--clean-gvsbuild")
}
Compile-Translations
Download-ModelWeights
Install-PythonDependencies
Create-EXE ($args -contains "--cli-only")
Create-7ZArchive
