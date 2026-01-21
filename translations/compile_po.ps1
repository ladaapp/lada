# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

$cwd_backup = (Get-Location).Path
$translationsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ((Get-Location).Path -ne $translationsDir) {
    Set-Location $translationsDir
}

if ((Get-Command "msgfmt" -ErrorAction SilentlyContinue) -eq $null) {
    # gettext not found
    if (Test-Path "..\build_gtk\gtk\x64\release\bin") {
        # If Lada GUI was built according to installation docs then gettext will be available at build_gtk
        $env:Path = (Resolve-Path "..\build_gtk\gtk\x64\release\bin").Path + ";" + $env:Path
	}
}

$global:lang_filter = ""
if ($args -contains "--release") {
    $global:lang_filter = (Get-Content .\release_ready_translations.txt -Raw).Trim() -replace '\s+', ' '
    if ($global:lang_filter -eq "") {
        Write-Host "No translations in .\release_ready_translations.txt"
        return
    }
}

function should_compile_po {
    param (
        [string]$lang
    )

    if (-not $global:lang_filter) {
        return $true
    }

	return ($global:lang_filter.split(" ") -contains $lang)
}

# Clean up compiled translations if there is no corresponding .po file anymore (deleted translations)
Get-ChildItem -Directory -Path ..\lada\locale | ForEach-Object {
    $langDir = $_.FullName
    $lang = $_.Name

    $poFile = "$lang.po"
    if (-not (Test-Path $poFile)) {
        $lcMessagesDir = Join-Path $langDir "LC_MESSAGES"
        if (Test-Path $lcMessagesDir) {
            Write-Host "Removing outdated compiled translations for language '$lang' at '$langDir'"
            Remove-Item -Path $langDir -Recurse -Force
        }
    }
}

# Compile .po files
Get-ChildItem -File -Filter "*.po" | ForEach-Object {
    $poFile = $_.Name
    $lang = $poFile -replace "\.po$"

    if (-not (should_compile_po $lang)) {
        $_langDir = "..\lada\locale\$lang"
        if (Test-Path $_langDir) {
            Remove-Item -Path $_langDir -Recurse -Force
        }
        return # actually a continue in a ForEach-Object loop
    }

    $langDir = "..\lada\locale\$lang\LC_MESSAGES"
    if (-not (Test-Path -Path $langDir)) {
        New-Item -ItemType Directory -Path $langDir -Force | Out-Null
    }

    Write-Host "Compiling language '$lang' .po file into .mo file"
    & msgfmt $poFile -o "$langDir\lada.mo"
}

Set-Location $cwd_backup