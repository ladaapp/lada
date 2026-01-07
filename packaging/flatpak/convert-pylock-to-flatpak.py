# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
#
# /// script
# dependencies = [
#   "toml",
#   "wheel-filename>=2.0.0",
#   "pyyaml",
# ]
# ///

"""
This script creates a Flatpak module file in YAML format by parsing a pylock.toml file.
Other tools exist (req2flatpak, flatpak-pip-generator) but they do not work with PyTorch multi-index setup and assume as single index/PyPi is used to get all wheels.
Make sure to validate the generated file. This script is not robust but should work fine for dependencies needed by Lada.
"""

import argparse
import os
import platform
import re
import subprocess
import sys
from urllib.parse import unquote

import toml
import yaml
from wheel_filename import WheelFilename

# Define variables for marker eval
platform_machine = platform.machine()
sys_platform = sys.platform


def read_pylock_toml(path: str) -> dict:
    if path == "-":
        return toml.load(sys.stdin)
    else:
        with open(path, 'r', encoding='utf-8') as file:
            return toml.load(file)

def write_flatpak_module_yaml(dependencies: list[tuple[str, str, str]], path: str) -> None:
    data = {
        "name": "lada-python-dependencies",
        "sources": [],
        "buildsystem": "simple",
        "build-commands": build_pip_install_commands(dependencies),
        "cleanup": ["/bin", "/share/man"]
    }

    for name, url, sha in sorted(dependencies, key=lambda x: x[0]):
        data["sources"].append({
            "type": "file",
            "url": url,
            "sha256": sha,
            "dest-filename": unquote(os.path.basename(url))
        })

    with open(path, 'w') as file:
        yaml.dump(data, file, width=float("inf"))


def build_pip_install_commands(dependencies: list[tuple[str, str, str]]) -> list[str]:
    packages = [name for name, _, _ in dependencies]
    return [
        "pip3 install --verbose --exists-action=i --no-index "
        "--find-links=\"file://${PWD}\" --prefix=${FLATPAK_DEST} "
        f"--no-build-isolation {' '.join(packages)}"
    ]


def get_runtime_version_info(gnome_runtime_version: int, module_name: str) -> str:
    command = f"flatpak run --command=cat org.gnome.Platform//{gnome_runtime_version} /usr/manifest.json | " \
              f"jq -r '.modules | unique | .[] | select(.name == \"{module_name}\").\"x-cpe\".version'"
    return subprocess.check_output(command, shell=True, text=True).strip()


def get_gnome_runtime_glib_version(gnome_runtime_version: int) -> tuple[int, int]:
    glib_version = get_runtime_version_info(gnome_runtime_version, "bootstrap/glibc.bst")
    return tuple(map(int, glib_version.split(".")))


def get_gnome_runtime_python_version(gnome_runtime_version: int) -> tuple[int, int, int]:
    python_version = get_runtime_version_info(gnome_runtime_version, "components/python3.bst")
    return tuple(map(int, python_version.split(".")))


def gather_dependencies(pylock: dict, glib_version: tuple[int, int], python_version: tuple[int, int, int]) -> list[tuple[str, str, str]]:
    glib_major, glib_minor = glib_version
    python_major, python_minor, python_patch = python_version

    dependencies = []
    for package in pylock['packages']:
        dependencies.extend(process_package(package, glib_major, glib_minor, python_major, python_minor, python_patch))

    return dependencies


def process_package(package: dict, glib_major: int, glib_minor: int, python_major: int, python_minor: int, python_patch: int) -> list[tuple[str, str, str]]:
    name, version = package['name'], package['version']
    compatible_wheels = []

    if 'marker' in package and not eval(package['marker']):
        return []

    if 'wheels' in package:
        compatible_wheels = get_compatible_wheels(package['wheels'], glib_major, glib_minor, python_major, python_minor)

    wheel_url, wheel_sha256 = get_wheel_or_sdist(compatible_wheels, package)

    if wheel_url:
        return [(name, wheel_url, wheel_sha256)]
    elif 'sdist' in package:
        sdist_url = package['sdist']['url']
        sdist_sha256 = package['sdist']['hashes']['sha256']
        return [(name, sdist_url, sdist_sha256)]
    return []


def get_compatible_wheels(wheels: list[dict], glib_major: int, glib_minor: int, python_major: int, python_minor: int) -> list:
    compatible_wheels = []
    for wheel in wheels:
        wheel_url = wheel['url']
        parsed_wheel = WheelFilename.parse(unquote(wheel_url))
        is_compatible = is_compatible_wheel(parsed_wheel, glib_major, glib_minor, python_major, python_minor)
        if isinstance(is_compatible, tuple) and is_compatible[0]:
            _, binary_wheel, wheel_glib_version_comparitor = is_compatible
            wheel_sha256 = wheel['hashes'].get('sha256')
            compatible_wheels.append((wheel_url, wheel_sha256, binary_wheel, wheel_glib_version_comparitor))
    return compatible_wheels


def is_compatible_wheel(parsed_wheel, glib_major: int, glib_minor: int, python_major: int, python_minor: int) -> bool | tuple[bool, bool, int]:
    incompatible = True
    wheel_glib_version_comparitor = None
    for tag in parsed_wheel.platform_tags:
        if tag == "any":
            incompatible = False
        elif tag.startswith("manylinux") and tag.endswith("x86_64"):
            incompatible = False
            result = re.search(r"manylinux_(\d+)_(\d+)_.*", tag)
            if result and len(result.groups()) == 2:
                wheel_glibc_major = int(result.group(1))
                wheel_glibc_minor = int(result.group(2))
                wheel_glib_version_comparitor = int(f"{wheel_glibc_major}{wheel_glibc_minor}")
                incompatible = wheel_glibc_major > glib_major or wheel_glibc_major == glib_major and wheel_glibc_minor > glib_minor
    if incompatible:
        return False
    incompatible = True
    binary_wheel = False
    for tag in parsed_wheel.python_tags:
        if tag == "py3":
            incompatible = False
        elif tag.startswith("cp") and tag == f"cp{python_major}{python_minor}":
            binary_wheel = True
            incompatible = False
        elif tag.startswith(f"cp{python_major}") and "abi3" in parsed_wheel.abi_tags:
            binary_wheel = True
            incompatible = False
    if incompatible:
        return False
    incompatible = False
    for tag in parsed_wheel.abi_tags:
        if tag == f"cp{python_major}{python_minor}t":
            incompatible = True
            break
    if incompatible:
        return False

    return True, binary_wheel, wheel_glib_version_comparitor

def get_wheel_or_sdist(compatible_wheels: list, package: dict) -> tuple[str, str]:
    if len(compatible_wheels) > 1:
        compatible_binary_wheels = [compatible_wheels[i] for i, (wheel_url, wheel_sha256, binary_wheel, wheel_glib_version_comparitor) in enumerate(compatible_wheels) if binary_wheel]
        if len(compatible_binary_wheels) > 1:
            assert all([wheel_glib_version_comparitor for _, _, _, wheel_glib_version_comparitor in compatible_binary_wheels])
            compatible_binary_wheels.sort(key=lambda x: x[3], reverse=True)  # Sort by glibc version comparitor
            return compatible_binary_wheels[0][0], compatible_binary_wheels[0][1]
        elif len(compatible_binary_wheels) == 1:
            return compatible_binary_wheels[0][0], compatible_binary_wheels[0][1]
        else:
            compatible_non_binary_wheels = [compatible_wheels[i] for i, (wheel_url, wheel_sha256, binary_wheel, wheel_glib_version_comparitor) in enumerate(compatible_wheels) if not binary_wheel]
            return compatible_non_binary_wheels[0][0], compatible_non_binary_wheels[0][1]
    elif len(compatible_wheels) == 1:
        return compatible_wheels[0][0], compatible_wheels[0][1]

    if 'sdist' in package:
        sdist_url = package['sdist']['url']
        sdist_sha256 = package['sdist']['hashes']['sha256']
        return sdist_url, sdist_sha256
    raise ValueError("Package without any compatible wheel or sdist")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='-', help="path to pylock.toml file or '-' to read from stdin")
    parser.add_argument('--output', type=str, default='packaging/flatpak/lada-python-dependencies.yaml')
    parser.add_argument('--gnome-runtime-version', type=int, default=49)
    return parser.parse_args()


def main(pylock_toml_path: str, gnome_runtime_version: int, flatpak_module_yaml_path: str):
    pylock = read_pylock_toml(pylock_toml_path)
    glib_version = get_gnome_runtime_glib_version(gnome_runtime_version)
    python_version = get_gnome_runtime_python_version(gnome_runtime_version)
    dependencies = gather_dependencies(pylock, glib_version, python_version)
    write_flatpak_module_yaml(dependencies, flatpak_module_yaml_path)


if __name__ == "__main__":
    args = parse_args()
    main(args.input, args.gnome_runtime_version, args.output)
