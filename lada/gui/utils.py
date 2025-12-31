# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import os
import subprocess
import xml.etree.ElementTree as ET

import torch
from gi.repository import Gio
from gi.repository import Gtk, GLib, Gdk

from lada import LOG_LEVEL
from lada.gui.config.config import Config
from lada.utils import video_utils
from lada.utils.video_utils import EncodingPreset

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

def is_device_available(device: str) -> bool:
    device = device.lower()
    if device == 'cpu':
        return True
    elif device.startswith("cuda:"):
        return device_to_gpu_id(device) < torch.cuda.device_count()
    return False


def device_to_gpu_id(device) -> int | None:
    if device.startswith("cuda:"):
        return int(device.split(":")[-1])
    return None


def get_available_gpus():
    gpus = []
    for id in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_properties(id).name
        # We're using these GPU names in a ComboBox but libadwaita sets up the label with max-width-chars: 20 and there does not
        # seem to be a way to overwrite this. So let's try to make sure GPU names are below 20 characters to be readable
        if gpu_name.startswith("NVIDIA GeForce RTX"):
            gpu_name = gpu_name.replace("NVIDIA GeForce RTX", "RTX")
        gpus.append((id, gpu_name))
    return gpus

def skip_if_uninitialized(f):
    def noop(*args):
        return
    def wrapper(*args):
        return f(*args) if args[0].init_done else noop
    return wrapper

def set_validation_css_classes(widget: Gtk.Widget, is_valid: bool):
    focused = "focused" in widget.get_css_classes()
    all_classes = {"success", "warning", "error"}
    def add_if_not_present(class_name):
        if class_name not in widget.get_css_classes():
            for other_class_names in all_classes.difference({class_name}):
                widget.remove_css_class(other_class_names)
            if class_name:
                widget.add_css_class(class_name)
    if is_valid:
        if focused:
            add_if_not_present("success")
        else:
            add_if_not_present(None)
    else:
        if focused:
            add_if_not_present("warning")
        else:
            add_if_not_present("error")

def validate_file_name_pattern(file_name_pattern: str) -> bool:
    if not "{orig_file_name}" in file_name_pattern:
        return False
    if os.sep in file_name_pattern:
        return False
    file_extension = os.path.splitext(file_name_pattern)[1].lower()
    if file_extension not in [".mp4", ".mkv", ".mov", ".m4v"]:
        return False
    return True

def validate_preset_description(description: str, config: Config, original_description: str | None) -> bool:
    presets = []
    presets.extend(config.custom_encoding_presets if original_description is None else [p for p in config.custom_encoding_presets if p.description != original_description])
    presets.extend(video_utils.get_encoding_presets())
    presets_descriptions = [preset.description.lower() for preset in presets]
    return description.lower() not in presets_descriptions

def filter_video_files(files: list[Gio.File]) -> list[Gio.File]:
    def is_video_file(file: Gio.File):
        file_info: Gio.FileInfo = file.query_info("standard::content-type", Gio.FileQueryInfoFlags.NONE)
        content_type = file_info.get_content_type() # on linux content_type is MIME type but on windows it's just a file extension
        if content_type is None: return False
        mime_type = Gio.content_type_get_mime_type(content_type)
        if mime_type is None: return False
        return mime_type.startswith("video/")
    filtered_files = [file for file in files if is_video_file(file)]
    return filtered_files

def show_open_files_dialog(callback, dismissed_callback):
    file_dialog = Gtk.FileDialog()
    video_file_filter = Gtk.FileFilter()
    video_file_filter.add_mime_type("video/*")
    file_dialog.set_default_filter(video_file_filter)
    file_dialog.set_title(_("Select one or multiple video files"))
    def on_open_multiple(_file_dialog, result):
        try:
            video_files = _file_dialog.open_multiple_finish(result)
            if len(video_files) > 0:
                callback(video_files)
        except GLib.Error as error:
            if error.code == 2: # "Dismissed by user"
                dismissed_callback()
                logger.debug("FileDialog cancelled: Dismissed by user")
            else:
                logger.error(f"Error opening file: {error.message}")
                raise error
    file_dialog.open_multiple(callback=on_open_multiple)

def create_video_files_drop_target(callback):
    drop_target = Gtk.DropTarget.new(Gio.File, actions=Gdk.DragAction.COPY)
    drop_target.set_gtypes((Gdk.FileList,))
    def on_file_drop(_drop_target, files: list[Gio.File], x, y):
        video_files = filter_video_files(files)
        if len(video_files) > 0:
            callback(video_files)
    drop_target.connect("drop", on_file_drop)
    return drop_target

def translate_ui_xml(path: str) -> str:
    with open(path, 'r', encoding="utf-8") as file:
        element = file.read()
    tree = ET.fromstring(element)
    for node in tree.iter():
        if 'translatable' in node.attrib:
            node.text = _(node.text)
            del node.attrib["translatable"]
    as_str = ET.tostring(tree, encoding='utf-8', method='xml')
    return as_str

def dump_encoder_options(encoder: str) -> str:
    result = subprocess.run(["ffmpeg", "-loglevel", "quiet", "-h", f"encoder={encoder}"], capture_output=True, text=True)
    text = result.stdout.strip().replace("Exiting with exit code 0", "").strip()
    return text

def get_next_custom_preset(config: Config) -> EncodingPreset:
    num = len(config.custom_encoding_presets) + 1
    description = _("Custom Preset {custom_preset_num}").format(custom_preset_num=num)
    return EncodingPreset(f"custom-preset-{num}", description, True, "libx264", "")

def get_selected_preset(config: Config) -> EncodingPreset:
    presets = []
    presets.extend(video_utils.get_encoding_presets())
    presets.extend(config.custom_encoding_presets)
    for preset in presets:
        if preset.name == config.encoding_preset_name:
            return preset
    raise ValueError("Selected preset not found")

def get_preset_by_name(config: Config, name: str) -> EncodingPreset:
    presets = []
    presets.extend(video_utils.get_encoding_presets())
    presets.extend(config.custom_encoding_presets)
    for preset in presets:
        if preset.name == name:
            return preset
    raise ValueError("Invalid preset name")

def is_unique_preset_description(description) -> bool:
    return not any([preset.description == description for preset in video_utils.get_encoding_presets()])