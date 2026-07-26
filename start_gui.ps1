# Launch Lada GUI from a source checkout on Windows.
param(
    [string]$ModelWeightsDir = "",
    [int]$SharedDecodeMaxMb = 4096,
    [int]$BasicVsrppRestoreWindowFrames = 160,
    [int]$BasicVsrppRestoreWindowOverlap = 32,
    [switch]$Console
)

$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$GtkRoot = Join-Path $ProjectRoot "build_gtk\gtk\x64\release"
$VenvScripts = Join-Path $ProjectRoot ".venv\Scripts"

if (-not (Test-Path -LiteralPath $GtkRoot)) {
    throw "GTK runtime not found: $GtkRoot"
}

if (-not $ModelWeightsDir) {
    $BundledModelDir = "D:\lada\_internal\model_weights"
    $LocalModelDir = Join-Path $ProjectRoot "model_weights"
    if (Test-Path -LiteralPath $LocalModelDir) {
        $ModelWeightsDir = $LocalModelDir
    } elseif (Test-Path -LiteralPath $BundledModelDir) {
        $ModelWeightsDir = $BundledModelDir
    } else {
        throw "Model weights directory not found. Pass -ModelWeightsDir <path>."
    }
}

if (-not (Test-Path -LiteralPath $ModelWeightsDir)) {
    throw "Model weights directory not found: $ModelWeightsDir"
}

$Python = Join-Path $VenvScripts "pythonw.exe"
if ($Console) {
    $Python = Join-Path $VenvScripts "python.exe"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python executable not found: $Python. Run dependency installation first."
}

$env:PATH = (Join-Path $GtkRoot "bin") + ";" + $env:PATH
$env:GI_TYPELIB_PATH = Join-Path $GtkRoot "lib\girepository-1.0"
$env:GST_PLUGIN_PATH = Join-Path $GtkRoot "lib\gstreamer-1.0"
$env:XDG_DATA_DIRS = Join-Path $GtkRoot "share"
$env:LADA_MODEL_WEIGHTS_DIR = $ModelWeightsDir
$env:LADA_SHARED_DECODE_MAX_MB = [string]$SharedDecodeMaxMb
$env:LADA_BASICVSRPP_RESTORE_WINDOW_FRAMES = [string]$BasicVsrppRestoreWindowFrames
$env:LADA_BASICVSRPP_RESTORE_WINDOW_OVERLAP = [string]$BasicVsrppRestoreWindowOverlap
$env:PYTHONUTF8 = "1"

Set-Location -LiteralPath $ProjectRoot
Write-Host "Using model weights: $ModelWeightsDir"
Write-Host "Using shared decode memory limit: $SharedDecodeMaxMb MB"
Write-Host "Using BasicVSR++ restore window: $BasicVsrppRestoreWindowFrames frames, overlap: $BasicVsrppRestoreWindowOverlap frames"

if ($Console) {
    & $Python -m lada.gui.main
} else {
    $Process = Start-Process -FilePath $Python -ArgumentList "-m lada.gui.main" -WorkingDirectory $ProjectRoot -PassThru
    Write-Host "Started Lada GUI. PID: $($Process.Id)"
}
