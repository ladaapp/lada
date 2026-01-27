# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import json
import logging
import tempfile
import threading
from enum import Enum
from pathlib import Path

from gi.repository import GLib, GObject, Adw

from lada import LOG_LEVEL, ModelFiles
from lada.gui import utils
from lada.utils import os_utils, video_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class ColorScheme(Enum):
    SYSTEM = 'system'
    LIGHT = 'light'
    DARK = 'dark'

class PostExportAction(Enum):
    NONE = 'none'
    SHUTDOWN = 'shutdown'
    CUSTOM_COMMAND = 'custom_command'

class Config(GObject.Object):
    _defaults = {
        'color_scheme': ColorScheme.SYSTEM,
        'device': os_utils.get_default_torch_device(),
        'custom_encoding_presets': set(),
        'encoding_preset_name': video_utils.get_default_preset_name(),
        'export_directory': None,
        'file_name_pattern': "{orig_file_name}.restored.mp4",
        'fp16_enabled': os_utils.gpu_has_fp16_acceleration(),
        'initial_view': 'watch',
        'max_clip_duration': 180,
        'mosaic_detection_model': 'v4-fast',
        'mosaic_restoration_model': 'basicvsrpp-v1.2',
        'mp4_fast_start': False,
        'mute_audio': False,
        'post_export_action': PostExportAction.NONE,
        'post_export_custom_command': '',
        'preview_buffer_duration': 0,
        'seek_preview_enabled': True,
        'show_mosaic_detections': False,
        'temp_directory': tempfile.gettempdir(),
        'detect_face_mosaics': False,
        'subtitles_font_size': 16,
    }

    def __init__(self, style_manager: Adw.StyleManager):
        super().__init__()
        self._color_scheme = self._defaults['color_scheme']
        self._device = self._defaults['device']
        self._encoding_preset_name = self._defaults['encoding_preset_name']
        self._custom_encoding_presets = self._defaults['custom_encoding_presets']
        self._export_directory = self._defaults['export_directory']
        self._file_name_pattern = self._defaults['file_name_pattern']
        self._initial_view = self._defaults['initial_view']
        self._max_clip_duration: int = self._defaults['max_clip_duration']
        self._mosaic_detection_model = self._defaults['mosaic_detection_model']
        self._mosaic_restoration_model = self._defaults['mosaic_restoration_model']
        self._mp4_fast_start = self._defaults['mp4_fast_start']
        self._mute_audio = self._defaults['mute_audio']
        self._preview_buffer_duration = self._defaults['preview_buffer_duration']
        self._seek_preview_enabled = self._defaults['seek_preview_enabled']
        self._show_mosaic_detections = self._defaults['show_mosaic_detections']
        self._post_export_action = self._defaults['post_export_action']
        self._post_export_custom_command = self._defaults['post_export_custom_command']
        self._temp_directory = self._defaults['temp_directory']
        self._fp16_enabled = self._defaults['fp16_enabled']
        self._detect_face_mosaics = self._defaults['detect_face_mosaics']
        self._subtitles_font_size = self._defaults['subtitles_font_size']

        self.save_lock = threading.Lock()
        self._style_manager = style_manager

    @GObject.Property()
    def seek_preview_enabled(self):
        return self._seek_preview_enabled

    @seek_preview_enabled.setter
    def seek_preview_enabled(self, value):
        if value == self._seek_preview_enabled:
            return
        self._seek_preview_enabled = value
        self.save()

    # Removed seek_preview_size property - now auto-scaled

    @GObject.Property()
    def show_mosaic_detections(self):
        return self._show_mosaic_detections

    @show_mosaic_detections.setter
    def show_mosaic_detections(self, value):
        if value == self._show_mosaic_detections:
            return
        self._show_mosaic_detections = value
        self.save()

    @GObject.Property()
    def mosaic_restoration_model(self):
        return self._mosaic_restoration_model

    @mosaic_restoration_model.setter
    def mosaic_restoration_model(self, value):
        if value == self._mosaic_restoration_model:
            return
        self._mosaic_restoration_model = value
        self.save()

    @GObject.Property()
    def mp4_fast_start(self):
        return self._mp4_fast_start

    @mp4_fast_start.setter
    def mp4_fast_start(self, value):
        if value == self._mp4_fast_start:
            return
        self._mp4_fast_start = value
        self.save()

    @GObject.Property()
    def mosaic_detection_model(self):
        return self._mosaic_detection_model

    @mosaic_detection_model.setter
    def mosaic_detection_model(self, value):
        if value == self._mosaic_detection_model:
            return
        self._mosaic_detection_model = value
        self.save()

    @GObject.Property()
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if value == self._device:
            return
        self._device = value
        self.save()

    @GObject.Property()
    def preview_buffer_duration(self):
        return self._preview_buffer_duration

    @preview_buffer_duration.setter
    def preview_buffer_duration(self, value):
        if value == self._preview_buffer_duration:
            return
        self._preview_buffer_duration = value
        self.save()

    @GObject.Property()
    def max_clip_duration(self) -> int:
        return int(self._max_clip_duration)

    @max_clip_duration.setter
    def max_clip_duration(self, value):
        if value == self._max_clip_duration:
            return
        self._max_clip_duration = value
        self.save()

    @GObject.Property()
    def mute_audio(self):
        return self._mute_audio

    @mute_audio.setter
    def mute_audio(self, value):
        if value == self._mute_audio:
            return
        self._mute_audio = value
        self.save()

    @GObject.Property()
    def custom_encoding_presets(self):
        return self._custom_encoding_presets

    @custom_encoding_presets.setter
    def custom_encoding_presets(self, value: set[video_utils.EncodingPreset]):
        if value == self._custom_encoding_presets and all([a == b for a, b in zip(value, self._custom_encoding_presets)]):
            return
        self._custom_encoding_presets = value
        self.save()

    @GObject.Property()
    def encoding_preset_name(self):
        return self._encoding_preset_name

    @encoding_preset_name.setter
    def encoding_preset_name(self, value: str):
        if value == self._encoding_preset_name:
            return
        self._encoding_preset_name = value
        self.save()

    @GObject.Property()
    def color_scheme(self):
        return self._color_scheme

    @color_scheme.setter
    def color_scheme(self, value):
        if value == self._color_scheme:
            return
        self._update_style(value)
        self._color_scheme = value
        self.save()

    @GObject.Property()
    def export_directory(self):
        return self._export_directory

    @export_directory.setter
    def export_directory(self, value):
        if value == self._export_directory:
            return
        self._export_directory = value
        self.save()

    @GObject.Property()
    def file_name_pattern(self):
        return self._file_name_pattern

    @file_name_pattern.setter
    def file_name_pattern(self, value):
        if value == self._file_name_pattern:
            return
        self._file_name_pattern = value
        self.save()

    @GObject.Property()
    def initial_view(self):
        return self._initial_view

    @initial_view.setter
    def initial_view(self, value):
        if value == self._initial_view:
            return
        self._initial_view = value
        self.save()

    @GObject.Property()
    def post_export_action(self):
        return self._post_export_action

    @post_export_action.setter
    def post_export_action(self, value):
        if value == self._post_export_action:
            return
        self._post_export_action = value
        self.save()

    @GObject.Property()
    def post_export_custom_command(self):
        return self._post_export_custom_command

    @post_export_custom_command.setter
    def post_export_custom_command(self, value):
        if value == self._post_export_custom_command:
            return
        self._post_export_custom_command = value
        self.save()

    @GObject.Property()
    def temp_directory(self):
        return self._temp_directory

    @temp_directory.setter
    def temp_directory(self, value):
        if value == self._temp_directory:
            return
        self._temp_directory = value
        self.save()

    @GObject.Property()
    def fp16_enabled(self):
        return self._fp16_enabled

    @fp16_enabled.setter
    def fp16_enabled(self, value):
        if value == self._fp16_enabled:
            return
        self._fp16_enabled = value
        self.save()

    @GObject.Property()
    def detect_face_mosaics(self):
        return self._detect_face_mosaics

    @detect_face_mosaics.setter
    def detect_face_mosaics(self, value):
        if value == self._detect_face_mosaics:
            return
        self._detect_face_mosaics = value
        self.save()

    @GObject.Property()
    def subtitles_font_size(self):
        return self._subtitles_font_size

    @subtitles_font_size.setter
    def subtitles_font_size(self, value):
        if value == self._subtitles_font_size:
            return
        self._subtitles_font_size = value
        self.save()

    def save(self):
        self.save_lock.acquire_lock()
        config_file_path = self.get_config_file_path()
        try:
            if not config_file_path.parent.exists():
                config_file_path.parent.mkdir(parents=True)
            with open(config_file_path, 'w') as f:
                config_dict = self._as_dict()
                json.dump(config_dict, f)
                logger.info(f"Saved config file {config_file_path}: {config_dict}")
        finally:
            self.save_lock.release_lock()

    def load_config(self):
        config_file_path = self.get_config_file_path()
        if not config_file_path.exists():
            logger.info(f"Config file doesn't exist at {config_file_path}")
            self.save()
            return

        try:
            with open(config_file_path, 'r') as f:
                config_dict = json.load(f)
                self._from_dict(config_dict)
                logger.info(f"Loaded config file {config_file_path}: {config_dict}")
        except Exception as e:
            logger.error(f"Error loading config file {config_file_path}, falling back to defaults: {e}")
        # The config might have changed in case of new or invalid values. Let's save it.
        self.save()
        self._update_style(self._color_scheme)

    def reset_to_default_values(self):
        self.color_scheme = self._defaults['color_scheme']
        self.encoding_preset_name = self._defaults['encoding_preset_name']
        self.custom_encoding_presets = self._defaults['custom_encoding_presets']
        self.export_directory = self._defaults['export_directory']
        self.file_name_pattern = self._defaults['file_name_pattern']
        self.fp16_enabled = self._defaults['fp16_enabled']
        self.initial_view = self._defaults['initial_view']
        self.max_clip_duration = self._defaults['max_clip_duration']
        self.mosaic_detection_model = self._defaults['mosaic_detection_model']
        self.mosaic_restoration_model = self._defaults['mosaic_restoration_model']
        self.mp4_fast_start = self._defaults['mp4_fast_start']
        self.mute_audio = self._defaults['mute_audio']
        self.post_export_action = self._defaults['post_export_action']
        self.post_export_custom_command = self._defaults['post_export_custom_command']
        self.preview_buffer_duration = self._defaults['preview_buffer_duration']
        self.seek_preview_enabled = self._defaults['seek_preview_enabled']
        self.show_mosaic_detections = self._defaults['show_mosaic_detections']
        self.temp_directory = self._defaults['temp_directory']
        self.validate_and_set_device(self._defaults['device'])
        self.detect_face_mosaics = self._defaults['detect_face_mosaics']
        self.subtitles_font_size = self._defaults['subtitles_font_size']
        self.save()

    def _update_style(self, color_scheme: ColorScheme):
        if color_scheme == ColorScheme.LIGHT: self._style_manager.set_color_scheme(Adw.ColorScheme.FORCE_LIGHT)
        elif color_scheme == ColorScheme.DARK: self._style_manager.set_color_scheme(Adw.ColorScheme.FORCE_DARK)
        elif color_scheme == ColorScheme.SYSTEM or color_scheme is None: self._style_manager.set_color_scheme(Adw.ColorScheme.DEFAULT)
        else:
            raise ValueError(f"unknown color scheme: {color_scheme}")

    def _as_dict(self) -> dict:
        return {
            'color_scheme': self._color_scheme.value,
            'device': self._device,
            'custom_encoding_presets': [self._encoding_preset_as_dict(preset) for preset in self._custom_encoding_presets],
            'encoding_preset_name': self._encoding_preset_name,
            'export_directory': self._export_directory,
            'file_name_pattern': self._file_name_pattern,
            'fp16_enabled': self._fp16_enabled,
            'initial_view': self._initial_view,
            'max_clip_duration': self._max_clip_duration,
            'mosaic_detection_model': self._mosaic_detection_model,
            'mosaic_restoration_model': self._mosaic_restoration_model,
            'mp4_fast_start': self._mp4_fast_start,
            'mute_audio': self._mute_audio,
            'post_export_action': self._post_export_action.value,
            'post_export_custom_command': self._post_export_custom_command,
            'preview_buffer_duration': self._preview_buffer_duration,
            'seek_preview_enabled': self._seek_preview_enabled,
            'show_mosaic_detections': self._show_mosaic_detections,
            'temp_directory': self._temp_directory,
            'detect_face_mosaics': self._detect_face_mosaics,
            'subtitles_font_size': self._subtitles_font_size,
        }

    def _encoding_preset_as_dict(self, encoding_preset: video_utils.EncodingPreset):
        return {
            'name': encoding_preset.name,
            'description': encoding_preset.description,
            'encoder_name': encoding_preset.encoder_name,
            'encoder_options': encoding_preset.encoder_options,
        }

    def get_default_value(self, key):
        return self._defaults.get(key)

    def _from_dict(self, dict):
        for key in self._defaults:
            if key in dict and dict[key] is not None:
                if key == 'device':
                    self.validate_and_set_device(dict[key])
                elif key == 'mosaic_restoration_model':
                    self.validate_and_set_restoration_model(dict[key])
                elif key == 'mosaic_detection_model':
                    self.validate_and_set_detection_model(dict[key])
                elif key == 'color_scheme':
                    self._color_scheme = ColorScheme(dict[key])
                elif key == 'post_export_action':
                    self._post_export_action = PostExportAction(dict[key])
                elif key == 'encoding_preset_name':
                    self.validate_and_set_encoding_preset_name(dict[key])
                elif key == 'custom_encoding_presets':
                    self.validate_and_set_custom_encoding_presets(dict[key])
                elif key == 'export_directory':
                    self.validate_and_set_export_directory(dict[key])
                elif key == 'temp_directory':
                    self.validate_and_set_temp_directory(dict[key])
                elif key == 'file_name_pattern':
                    self.validate_and_set_file_name_pattern(dict[key])
                elif key == 'initial_view':
                    self.validate_and_set_initial_view(dict[key])
                else:
                    setattr(self, f"_{key}", dict[key])

    def get_config_file_path(self) -> Path:
        base_config_dir = GLib.get_user_config_dir()
        return Path(base_config_dir).joinpath('lada').joinpath('lada.conf')

    def validate_and_set_device(self, configured_device: str):
        available_gpus = utils.get_available_gpus()
        is_configured_device_available = utils.is_device_available(configured_device)
        is_no_gpu_available = len(available_gpus) == 0
        if is_no_gpu_available:
            self._device = "cpu"
            logger.warning(f"No GPU available, falling back to device cpu.")
        else:
            if is_configured_device_available and configured_device != "cpu":
                self._device = configured_device
            else:
                if configured_device == "cpu":
                    logger.info(
                        f"Configured device is CPU but as GPU(s) are available will choose {self._device} instead. Available gpus: {available_gpus}")
                else:
                    logger.info(
                        f"Configured device {configured_device} is not available choose {self._device} instead. Available gpus: {available_gpus}")
                self._device = available_gpus[0][0]

    def validate_and_set_restoration_model(self, restoration_model_name: str):
        available_models = [modelfile.name for modelfile in ModelFiles.get_restoration_models()]
        if restoration_model_name in available_models:
            self._mosaic_restoration_model = restoration_model_name
        else:
            default_model = self.get_default_value('mosaic_restoration_model')
            if default_model not in available_models:
                raise EnvironmentError(
                    f"Neither the configured restoration model {restoration_model_name} nor the default model {default_model} is not available on the filesystem")
            logger.warning(
                f"configured restoration model {restoration_model_name} is not available on the filesystem, falling back to model {default_model}")
            self._mosaic_restoration_model = default_model

    def validate_and_set_detection_model(self, detection_model_name: str):
        available_models = [modelfile.name for modelfile in ModelFiles.get_detection_models()]
        if detection_model_name in available_models:
            self._mosaic_detection_model = detection_model_name
        else:
            default_model = self.get_default_value('mosaic_detection_model')
            if default_model not in available_models:
                raise EnvironmentError(
                    f"Neither the configured detection model {detection_model_name} nor the default model {default_model} is not available on the filesystem")
            logger.warning(
                f"configured detection model {detection_model_name} is not available on the filesystem, falling back to model {default_model}")
            self._mosaic_detection_model = default_model

    def validate_and_set_encoding_preset_name(self, encoding_preset_name: str):
        if encoding_preset_name in [preset.name for preset in video_utils.get_encoding_presets()] or encoding_preset_name in [preset.name for preset in self._custom_encoding_presets]:
            self._encoding_preset_name = encoding_preset_name
        else:
            logger.warning(f"Configured encoding preset {encoding_preset_name} not found in custom or system presets, falling back to '{self._encoding_preset_name}'")
            self._encoding_preset_name = self.get_default_value('encoding_preset_name')

    def validate_and_set_custom_encoding_presets(self, custom_encoding_presets: list[dict]):
        self._custom_encoding_presets = set()
        for custom_preset in custom_encoding_presets:
            try:
                self._custom_encoding_presets.add(video_utils.EncodingPreset(custom_preset["name"], custom_preset["description"], True, custom_preset["encoder_name"], custom_preset["encoder_options"]))
            except:
                logger.warning(f"Couldn't parse custom preset '{custom_preset}' as EncodingPreset. Ignoring...")

    def validate_and_set_export_directory(self, export_directory: str | None):
        if export_directory is None:
            self._export_directory = None
        else:
            path = Path(export_directory)
            if path.is_dir():
                self._export_directory = export_directory
            else:
                self._export_directory = None
                logger.warning(f"Configured export directory '{export_directory}' does not exist or is not a directory on the filesystem, falling back to '{self._export_directory}'")

    def validate_and_set_temp_directory(self, temp_directory: str):
        path = Path(temp_directory)
        if path.is_dir():
            self._temp_directory = temp_directory
        else:
            self._temp_directory = self.get_default_value('temp_directory')
            logger.warning(f"Configured temp directory '{temp_directory}' does not exist or is not a directory on the filesystem, falling back to '{self._temp_directory}'")

    def validate_and_set_file_name_pattern(self, file_name_pattern: str):
        if utils.validate_file_name_pattern(file_name_pattern):
            self._file_name_pattern = file_name_pattern
        else:
            self._file_name_pattern = self.get_default_value('file_name_pattern')
            logger.warning(f"Configured file name pattern '{file_name_pattern}' is invalid, falling back to '{self._file_name_pattern}'")

    def validate_and_set_initial_view(self, initial_view: str):
        if initial_view in ["watch", "export"]:
            self._initial_view = initial_view
        elif initial_view == "preview": # previously, watch view was named preview view
            self._initial_view = "watch"
        else:
            self._initial_view = self.get_default_value('initial_view')
            logger.warning(f"Configured initial view '{initial_view}' is invalid, falling back to '{self._initial_view}'")
