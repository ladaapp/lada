import json
import logging
import threading
from enum import Enum
from pathlib import Path

from gi.repository import GLib, GObject, Adw

from lada import LOG_LEVEL
from lada import get_available_restoration_models, get_available_detection_models
from lada.gui import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class ColorScheme(Enum):
    SYSTEM = 'system'
    LIGHT = 'light'
    DARK = 'dark'

class Config(GObject.Object):
    _defaults = {
        'color_scheme': ColorScheme.SYSTEM,
        'custom_ffmpeg_encoder_options': '',
        'device': 'cuda:0',
        'export_codec': 'libx264',
        'export_crf': 20,
        'export_directory': None,
        'file_name_pattern': "{orig_file_name}.restored.mp4",
        'initial_view': 'preview',
        'max_clip_duration': 180,
        'mosaic_detection_model': 'v3.1-fast',
        'mosaic_restoration_model': 'basicvsrpp-v1.2',
        'mute_audio': False,
        'preview_buffer_duration': 0,
        'show_mosaic_detections': False,
    }

    def __init__(self, style_manager: Adw.StyleManager):
        super().__init__()
        self._color_scheme = self._defaults['color_scheme']
        self._custom_ffmpeg_encoder_options = self._defaults['custom_ffmpeg_encoder_options']
        self._device = self._defaults['device']
        self._export_codec = self._defaults['export_codec']
        self._export_crf = self._defaults['export_crf']
        self._export_directory = self._defaults['export_directory']
        self._file_name_pattern = self._defaults['file_name_pattern']
        self._initial_view = self._defaults['initial_view']
        self._max_clip_duration = self._defaults['max_clip_duration']
        self._mosaic_detection_model = self._defaults['mosaic_detection_model']
        self._mosaic_restoration_model = self._defaults['mosaic_restoration_model']
        self._mute_audio = self._defaults['mute_audio']
        self._preview_buffer_duration = self._defaults['preview_buffer_duration']
        self._show_mosaic_detections = self._defaults['show_mosaic_detections']

        self.save_lock = threading.Lock()
        self._style_manager = style_manager

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
    def max_clip_duration(self):
        return self._max_clip_duration

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
    def export_crf(self):
        return self._export_crf

    @export_crf.setter
    def export_crf(self, value):
        if value == self._export_crf:
            return
        self._export_crf = value
        self.save()

    @GObject.Property()
    def export_codec(self):
        return self._export_codec

    @export_codec.setter
    def export_codec(self, value):
        if value == self._export_codec:
            return
        self._export_codec = value
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
    def custom_ffmpeg_encoder_options(self):
        return self._custom_ffmpeg_encoder_options

    @custom_ffmpeg_encoder_options.setter
    def custom_ffmpeg_encoder_options(self, value):
        if value == self._custom_ffmpeg_encoder_options:
            return
        self._custom_ffmpeg_encoder_options = value
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
        self.custom_ffmpeg_encoder_options = self._defaults['custom_ffmpeg_encoder_options']
        self.export_codec = self._defaults['export_codec']
        self.export_crf = self._defaults['export_crf']
        self.export_directory = self._defaults['export_directory']
        self.file_name_pattern = self._defaults['file_name_pattern']
        self.initial_view = self._defaults['initial_view']
        self.max_clip_duration = self._defaults['max_clip_duration']
        self.mosaic_detection_model = self._defaults['mosaic_detection_model']
        self.mosaic_restoration_model = self._defaults['mosaic_restoration_model']
        self.mute_audio = self._defaults['mute_audio']
        self.preview_buffer_duration = self._defaults['preview_buffer_duration']
        self.show_mosaic_detections = self._defaults['show_mosaic_detections']
        self.validate_and_set_device(self._defaults['device'])
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
            'custom_ffmpeg_encoder_options': self._custom_ffmpeg_encoder_options,
            'device': self._device,
            'export_codec': self._export_codec,
            'export_crf': self._export_crf,
            'export_directory': self._export_directory,
            'file_name_pattern': self._file_name_pattern,
            'initial_view': self._initial_view,
            'max_clip_duration': self._max_clip_duration,
            'mosaic_detection_model': self._mosaic_detection_model,
            'mosaic_restoration_model': self._mosaic_restoration_model,
            'mute_audio': self._mute_audio,
            'preview_buffer_duration': self._preview_buffer_duration,
            'show_mosaic_detections': self._show_mosaic_detections,
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
                elif key == 'export_codec':
                    self.validate_and_set_export_codec(dict[key])
                elif key == 'export_directory':
                    self.validate_and_set_export_directory(dict[key])
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
                self._device = f"cuda:{available_gpus[0][0]}"

    def validate_and_set_restoration_model(self, restoration_model_name: str):
        available_models = get_available_restoration_models()
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
        available_models = get_available_detection_models()
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

    def validate_and_set_export_codec(self, export_codec: str):
        if export_codec == 'h264':
            self._export_codec = 'libx264'
        elif export_codec == 'h265' or export_codec == 'hevc':
            self._export_codec = 'libx265'
        elif export_codec not in utils.get_available_video_codecs():
            self._export_codec = self.get_default_value('export_codec')
            logger.warning(f"Configured codec {export_codec} not the list of available/recommended list of codecs, falling back to '{self._export_codec}'")
        else:
            self._export_codec = export_codec

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

    def validate_and_set_file_name_pattern(self, file_name_pattern: str):
        if utils.validate_file_name_pattern(file_name_pattern):
            self._file_name_pattern = file_name_pattern
        else:
            self._file_name_pattern = self.get_default_value('file_name_pattern')
            logger.warning(f"Configured file name pattern '{file_name_pattern}' is invalid, falling back to '{self._file_name_pattern}'")

    def validate_and_set_initial_view(self, initial_view: str):
        if initial_view in ["preview", "export"]:
            self._initial_view = initial_view
        else:
            self._initial_view = self.get_default_value('initial_view')
            logger.warning(f"Configured initial view '{initial_view}' is invalid, falling back to '{self._initial_view}'")
