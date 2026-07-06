# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import errno
import os
import shutil

SUPPORTED_VIDEO_FILE_EXTENSIONS = {
    ".asf",
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".ts",
    ".wmv",
    ".webm",
}

SUPPORTED_IMAGE_FILE_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def get_file_extension(file_path: str) -> str:
    return os.path.splitext(file_path)[1].lower()


def is_video_file(file_path: str) -> bool:
    return get_file_extension(file_path) in SUPPORTED_VIDEO_FILE_EXTENSIONS


def is_image_file(file_path: str) -> bool:
    return get_file_extension(file_path) in SUPPORTED_IMAGE_FILE_EXTENSIONS


def is_media_file(file_path: str) -> bool:
    return is_video_file(file_path) or is_image_file(file_path)


def get_output_file_path(input_file_path: str, output_directory: str, output_file_pattern: str) -> str:
    input_file = os.path.basename(input_file_path)
    input_stem, input_ext = os.path.splitext(input_file)
    output_file_name = output_file_pattern.replace("{orig_file_name}", input_stem)

    if is_image_file(input_file_path):
        output_stem, output_ext = os.path.splitext(output_file_name)
        if get_file_extension(output_file_name) not in SUPPORTED_IMAGE_FILE_EXTENSIONS:
            output_file_name = f"{output_stem}{input_ext.lower()}"
        elif output_ext != output_ext.lower():
            output_file_name = f"{output_stem}{output_ext.lower()}"
    elif is_video_file(input_file_path):
        output_stem, output_ext = os.path.splitext(output_file_name)
        if get_file_extension(output_file_name) not in SUPPORTED_VIDEO_FILE_EXTENSIONS:
            output_file_name = f"{output_stem}.mp4"
        elif output_ext != output_ext.lower():
            output_file_name = f"{output_stem}{output_ext.lower()}"

    return os.path.join(output_directory, output_file_name)


def replace_file(source_path: str, destination_path: str):
    destination_dir = os.path.dirname(destination_path)
    if destination_dir:
        os.makedirs(destination_dir, exist_ok=True)

    try:
        os.replace(source_path, destination_path)
    except OSError as error:
        if error.errno == errno.EXDEV or getattr(error, "winerror", None) == 17:
            shutil.move(source_path, destination_path)
            return
        raise
