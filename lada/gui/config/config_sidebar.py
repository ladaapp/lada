# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pathlib

from gi.repository import Gtk, GObject, Adw, Gio, GLib

from lada import get_available_restoration_models, get_available_detection_models, LOG_LEVEL
from lada.gui import utils
from lada.gui.config.config import Config, ColorScheme, PostExportAction
from lada.gui.utils import skip_if_uninitialized, get_available_video_codecs, validate_file_name_pattern

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'config_sidebar.ui'))
class ConfigSidebar(Gtk.Box):
    __gtype_name__ = 'ConfigSidebar'

    combo_row_gpu = Gtk.Template.Child()
    combo_row_mosaic_removal_models = Gtk.Template.Child()
    combo_row_mosaic_detection_models = Gtk.Template.Child()
    spin_row_export_crf = Gtk.Template.Child()
    combo_row_export_codec = Gtk.Template.Child()
    spin_row_preview_buffer_duration = Gtk.Template.Child()
    spin_row_clip_max_duration = Gtk.Template.Child()
    switch_row_mute_audio = Gtk.Template.Child()
    preferences_page = Gtk.Template.Child()
    light_color_scheme_button = Gtk.Template.Child()
    dark_color_scheme_button = Gtk.Template.Child()
    system_color_scheme_button = Gtk.Template.Child()
    action_row_export_directory: Adw.ActionRow = Gtk.Template.Child()
    check_button_export_directory_alwaysask: Gtk.CheckButton = Gtk.Template.Child()
    check_button_export_directory_defaultdir: Gtk.CheckButton = Gtk.Template.Child()
    action_row_temp_directory: Adw.ActionRow = Gtk.Template.Child()
    entry_row_file_name_pattern: Adw.EntryRow = Gtk.Template.Child()
    toggle_button_initial_view_preview: Gtk.ToggleButton = Gtk.Template.Child()
    toggle_button_initial_view_export: Gtk.ToggleButton = Gtk.Template.Child()
    entry_row_custom_ffmpeg_encoder_options: Adw.EntryRow = Gtk.Template.Child()
    expander_row_post_export_action: Adw.ExpanderRow = Gtk.Template.Child()
    check_button_post_export_shutdown: Gtk.CheckButton = Gtk.Template.Child()
    check_button_post_export_custom_command: Gtk.CheckButton = Gtk.Template.Child()
    entry_row_post_export_custom_command: Adw.EntryRow = Gtk.Template.Child()
    check_button_show_mosaic_detections: Gtk.CheckButton = Gtk.Template.Child()
    switch_row_seek_preview = Gtk.Template.Child()
    switch_row_fp16: Adw.SwitchRow = Gtk.Template.Child()
    switch_row_detect_faces = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config: Config | None = None
        self.init_done = False
        self._show_playback_section = True
        self._show_export_section = True

    def init_sidebar_from_config(self, config: Config):
        self.init_done = False

        self.check_button_show_mosaic_detections.props.active = config.show_mosaic_detections

        # init device
        combo_row_gpu_list = Gtk.StringList.new([])
        available_gpus = utils.get_available_gpus()
        configured_gpu_selection_idx = None
        for gpu_selection_idx, (device_id, device_name) in enumerate(available_gpus):
            combo_row_gpu_list.append(device_name)
            if config.device and utils.device_to_gpu_id(config.device) == device_id:
                configured_gpu_selection_idx = gpu_selection_idx
        self.combo_row_gpu.set_model(combo_row_gpu_list)
        if configured_gpu_selection_idx:
            self.combo_row_gpu.set_selected(configured_gpu_selection_idx)

        # init restoration model
        combo_row_models_list = Gtk.StringList.new([])
        available_models = get_available_restoration_models()
        for model_name in available_models:
            combo_row_models_list.append(model_name)
        self.combo_row_mosaic_removal_models.set_model(combo_row_models_list)
        idx = available_models.index(config.get_property("mosaic_restoration_model"))
        self.combo_row_mosaic_removal_models.set_selected(idx)

        # init detection model
        combo_row_detection_models_list = Gtk.StringList.new([])
        available_detection_models = get_available_detection_models()
        for model_name in available_detection_models:
            combo_row_detection_models_list.append(model_name)
        self.combo_row_mosaic_detection_models.set_model(combo_row_detection_models_list)
        idx = available_detection_models.index(config.mosaic_detection_model)
        self.combo_row_mosaic_detection_models.set_selected(idx)

        # init codec
        combo_row_export_codec_models_list = Gtk.StringList.new([])
        codecs = get_available_video_codecs()
        for codec_name in codecs:
            combo_row_export_codec_models_list.append(codec_name)
        self.combo_row_export_codec.set_model(combo_row_export_codec_models_list)
        idx = codecs.index(config.export_codec)
        self.combo_row_export_codec.set_selected(idx)

        self.spin_row_export_crf.set_property('value', config.export_crf)

        self.spin_row_preview_buffer_duration.set_value(config.preview_buffer_duration)
        self.spin_row_clip_max_duration.set_value(config.max_clip_duration)
        self.switch_row_mute_audio.set_active(config.mute_audio)

        self.switch_row_seek_preview.set_active(config.seek_preview_enabled)
        self.switch_row_fp16.set_active(config.fp16_enabled)
        self.switch_row_detect_faces.set_active(config.detect_face_mosaics)
        self.switch_row_detect_faces.set_visible(config.mosaic_detection_model != 'v2')

        # init color scheme
        if config.color_scheme == ColorScheme.LIGHT: self.light_color_scheme_button.set_property("active", True)
        elif config.color_scheme == ColorScheme.DARK: self.dark_color_scheme_button.set_property("active", True)
        else: self.system_color_scheme_button.set_property("active", True)

        # init export directory
        if config.export_directory:
            self.action_row_export_directory.set_subtitle(config.export_directory)
            self.check_button_export_directory_defaultdir.set_active(True)
        else:
            self.action_row_export_directory.set_subtitle(_("Click the folder button to choose a default"))
            self.check_button_export_directory_alwaysask.set_active(True)

        self.entry_row_file_name_pattern.set_text(config.file_name_pattern)

        # init temp directory
        self.action_row_temp_directory.set_subtitle(config.temp_directory)

        self.toggle_button_initial_view_preview.set_active(config.initial_view == "preview")
        self.toggle_button_initial_view_export.set_active(config.initial_view == "export")

        self.entry_row_custom_ffmpeg_encoder_options.set_text(config.custom_ffmpeg_encoder_options)

        # init post-export action
        self.check_button_post_export_shutdown.set_active(config.post_export_action == PostExportAction.SHUTDOWN)
        self.check_button_post_export_custom_command.set_active(config.post_export_action == PostExportAction.CUSTOM_COMMAND)
        self.expander_row_post_export_action.set_enable_expansion(config.post_export_action != PostExportAction.NONE)
        self.expander_row_post_export_action.set_expanded(config.post_export_action != PostExportAction.NONE)
        self.entry_row_post_export_custom_command.set_text(config.post_export_custom_command)
        self.update_custom_command_visibility(config.post_export_action)

        self.init_done = True

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value: Config):
        self._config = value
        self.init_sidebar_from_config(value)

    @GObject.Property()
    def disabled(self):
        return self.get_property("sensitive")

    @disabled.setter
    def disabled(self, value):
        self.set_property("sensitive", not value)

    @GObject.Property(type=bool, default=True)
    def show_playback_section(self):
        return self._show_playback_section

    @show_playback_section.setter
    def show_playback_section(self, value):
        self._show_playback_section = value

    @GObject.Property(type=bool, default=True)
    def show_export_section(self):
        return self._show_export_section

    @show_export_section.setter
    def show_export_section(self, value):
        self._show_export_section = value

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def combo_row_mosaic_removal_models_selected_callback(self, combo_row, value):
        self._config.mosaic_restoration_model = combo_row.get_property("selected_item").get_string()

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def combo_row_mosaic_detection_models_selected_callback(self, combo_row, value):
        self._config.mosaic_detection_model = combo_row.get_property("selected_item").get_string()
        self.switch_row_detect_faces.set_visible(self._config.mosaic_detection_model != 'v2')

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def combo_row_mosaic_export_codec_selected_callback(self, combo_row, value):
        self._config.export_codec = combo_row.get_property("selected_item").get_string()

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def spin_row_preview_export_crf_selected_callback(self, spin_row, value):
        self._config.export_crf = spin_row.get_property("value")

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def combo_row_gpu_selected_callback(self, combo_row, value):
        selected_gpu_name = combo_row.get_property("selected_item").get_string()
        for id, name in utils.get_available_gpus():
            if name == selected_gpu_name:
                self._config.device = f"cuda:{id}"
                break

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def spin_row_preview_buffer_duration_selected_callback(self, spin_row, value):
        self._config.preview_buffer_duration = spin_row.get_property("value")

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def spin_row_clip_max_duration_selected_callback(self, spin_row, value):
        self._config.max_clip_duration = int(spin_row.get_property("value"))

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def switch_row_mute_audio_active_callback(self, switch_row, active):
        self._config.mute_audio = switch_row.get_property("active")

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def button_config_reset_callback(self, button_clicked):
        self.init_done = False
        self._config.reset_to_default_values()
        self.init_sidebar_from_config(self._config)

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def toggle_button_system_color_scheme_callback(self, button_clicked):
        self._config.color_scheme = ColorScheme.SYSTEM

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def toggle_button_light_color_scheme_callback(self, button_clicked):
        self._config.color_scheme = ColorScheme.LIGHT

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def toggle_button_dark_color_scheme_callback(self, button_clicked):
        self._config.color_scheme = ColorScheme.DARK

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def check_button_export_directory_alwaysask_callback(self, button_clicked):
        if self.check_button_export_directory_alwaysask.get_active():
            self._config.export_directory = None
            self.action_row_export_directory.set_subtitle(_("Click the folder button to choose a default"))

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def check_button_export_directory_defaultdir_callback(self, button_clicked):
        if self.check_button_export_directory_defaultdir.get_active() and not self._config.export_directory:
            self.show_select_folder()

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def toggle_button_export_directory_filepicker_callback(self, button_clicked):
        self.show_select_folder()

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def toggle_button_temp_directory_filepicker_callback(self, button_clicked):
        self.show_select_temp_folder()

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def entry_row_file_name_pattern_changed_callback(self, entry_row):
        self.set_file_name_pattern_row_styles()
        if validate_file_name_pattern(self.entry_row_file_name_pattern.get_text()):
            self._config.file_name_pattern = self.entry_row_file_name_pattern.get_text()

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def entry_row_file_name_pattern_focused_callback(self, row_entry, param_spec):
        self.set_file_name_pattern_row_styles()

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def toggle_button_initial_view_preview_callback(self, button_clicked):
        self._config.initial_view = "preview"

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def toggle_button_initial_view_export_callback(self, button_clicked):
        self._config.initial_view = "export"

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def entry_row_custom_ffmpeg_encoder_options_changed_callback(self, entry_row):
        self._config.custom_ffmpeg_encoder_options = self.entry_row_custom_ffmpeg_encoder_options.get_text()

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def check_button_show_mosaic_detections_callback(self, check_button):
        self._config.show_mosaic_detections = self.check_button_show_mosaic_detections.props.active

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def expander_row_post_export_action_enable_callback(self, expander_row: Adw.ExpanderRow, param_spec):
        enabled: bool = expander_row.get_property(param_spec.name)
        if enabled:
            self.check_button_post_export_shutdown.set_active(True)
            self._config.post_export_action = PostExportAction.SHUTDOWN
        else:
            self._config.post_export_action = PostExportAction.NONE
        self.update_custom_command_visibility(self._config.post_export_action)

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def check_button_post_export_shutdown_callback(self, check_button):
        if check_button.get_active():
            self._config.post_export_action = PostExportAction.SHUTDOWN
        self.update_custom_command_visibility(self._config.post_export_action)

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def check_button_post_export_custom_command_callback(self, check_button):
        if check_button.get_active():
            self._config.post_export_action = PostExportAction.CUSTOM_COMMAND
        self.update_custom_command_visibility(self._config.post_export_action)

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def entry_row_post_export_custom_command_changed_callback(self, entry_row):
        self._config.post_export_custom_command = self.entry_row_post_export_custom_command.get_text()

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def switch_row_seek_preview_active_callback(self, switch_row, active):
        self._config.seek_preview_enabled = switch_row.get_property("active")

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def switch_row_fp16_active_callback(self, switch_row, active):
        self._config.fp16_enabled = switch_row.get_property("active")

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def switch_row_detect_face_mosaics_callback(self, switch_row, active):
        self._config.detect_face_mosaics = switch_row.get_property("active")

    def set_file_name_pattern_row_styles(self):
        is_valid = validate_file_name_pattern(self.entry_row_file_name_pattern.get_text())
        focused = "focused" in self.entry_row_file_name_pattern.get_css_classes()
        all_classes = {"success", "warning", "error"}
        def add_if_not_present(class_name):
            if class_name not in self.entry_row_file_name_pattern.get_css_classes():
                for other_class_names in all_classes.difference({class_name}):
                    self.entry_row_file_name_pattern.remove_css_class(other_class_names)
                if class_name:
                    self.entry_row_file_name_pattern.add_css_class(class_name)
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

    def show_select_folder(self):
        file_dialog = Gtk.FileDialog()
        file_dialog.set_title(_("Select a folder where restored videos should be saved"))
        def on_select_folder(_file_dialog, result):
            try:
                selected_folder: Gio.File = _file_dialog.select_folder_finish(result)
                selected_folder_path = selected_folder.get_path()
                self._config.export_directory = selected_folder_path
                self.action_row_export_directory.set_subtitle(selected_folder_path)
                if not self.check_button_export_directory_defaultdir.get_active(): self.check_button_export_directory_defaultdir.set_active(True)
            except GLib.Error as error:
                if error.code == 2: # "Dismissed by user"
                    logger.debug("FileDialog cancelled: Dismissed by user")
                else:
                    logger.error(f"Error selecting folder: {error.message}")
                    raise error
                if self.check_button_export_directory_defaultdir and not self._config.export_directory:
                    self.check_button_export_directory_alwaysask.set_active(True)
        file_dialog.select_folder(callback=on_select_folder)

    def show_select_temp_folder(self):
        file_dialog = Gtk.FileDialog()
        file_dialog.set_title(_("Select a folder for temporary files"))
        file_dialog.set_initial_folder(Gio.File.new_for_path(self._config.temp_directory))
        def on_select_temp_folder(_file_dialog, result):
            try:
                selected_folder: Gio.File = _file_dialog.select_folder_finish(result)
                selected_folder_path = selected_folder.get_path()
                self._config.temp_directory = selected_folder_path
                self.action_row_temp_directory.set_subtitle(selected_folder_path)
            except GLib.Error as error:
                if error.code == 2: # "Dismissed by user"
                    logger.debug("FileDialog cancelled: Dismissed by user")
                else:
                    logger.error(f"Error selecting folder: {error.message}")
                    raise error
        file_dialog.select_folder(callback=on_select_temp_folder)

    def update_custom_command_visibility(self, action: PostExportAction):
        self.entry_row_post_export_custom_command.set_visible(action == PostExportAction.CUSTOM_COMMAND)
