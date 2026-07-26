# Lada export completion notification.
# Usage:
#   powershell -ExecutionPolicy Bypass -File notify_export_complete.ps1

param(
    [string]$Title = "Lada",
    [string]$Message = "Export completed",
    [int]$ToastDuration = 8000
)

# Audio.
$soundPlayer = New-Object System.Media.SoundPlayer
$soundPath = "$env:SystemRoot\Media\tada.wav"
if (Test-Path $soundPath) {
    $soundPlayer.SoundLocation = $soundPath
    $soundPlayer.Play()
}

# Visual: system tray balloon. This also works when the app is minimized.
Add-Type -AssemblyName System.Windows.Forms
$icon = New-Object System.Windows.Forms.NotifyIcon
$icon.Icon = [System.Drawing.SystemIcons]::Information
$icon.BalloonTipTitle = $Title
$icon.BalloonTipText = $Message
$icon.Visible = $true
$icon.ShowBalloonTip($ToastDuration)
Start-Sleep -Milliseconds ($ToastDuration + 500)
$icon.Dispose()
