# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0
import logging
import pathlib
from typing import Callable

from gi.repository import Gtk, GObject, Adw, Gio, GLib

from lada import LOG_LEVEL, ModelFiles, ModelFile
from lada.gui import utils
from lada.gui.config.config import Config, ColorScheme, PostExportAction
from lada.gui.config.encoding_preset_dialog import EncodingPresetDialog
from lada.gui.utils import skip_if_uninitialized, validate_file_name_pattern
from lada.utils import video_utils
from lada.utils.video_utils import EncodingPreset

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(string=utils.translate_ui_xml(here / 'config_sidebar.ui'))
class ConfigSidebar(Gtk.Box):
    __gtype_name__ = 'ConfigSidebar'

    combo_row_gpu = Gtk.Template.Child()
    spin_row_preview_buffer_duration = Gtk.Template.Child()
    spin_row_clip_max_duration = Gtk.Template.Child()
    switch_row_mute_audio = Gtk.Template.Child()
    switch_row_mp4_fast_start = Gtk.Template.Child()
    preferences_page = Gtk.Template.Child()
    light_color_scheme_button = Gtk.Template.Child()
    dark_color_scheme_button = Gtk.Template.Child()
    system_color_scheme_button = Gtk.Template.Child()
    expander_row_export_directory: Adw.ExpanderRow = Gtk.Template.Child()
    action_row_export_directory: Adw.ActionRow = Gtk.Template.Child()
    action_row_export_directory_alwaysask: Adw.ActionRow = Gtk.Template.Child()
    check_button_export_directory_alwaysask: Gtk.CheckButton = Gtk.Template.Child()
    check_button_export_directory_defaultdir: Gtk.CheckButton = Gtk.Template.Child()
    action_row_temp_directory: Adw.ActionRow = Gtk.Template.Child()
    entry_row_file_name_pattern: Adw.EntryRow = Gtk.Template.Child()
    toggle_button_initial_view_preview: Gtk.ToggleButton = Gtk.Template.Child()
    toggle_button_initial_view_export: Gtk.ToggleButton = Gtk.Template.Child()
    expander_row_post_export_action: Adw.ExpanderRow = Gtk.Template.Child()
    check_button_post_export_shutdown: Gtk.CheckButton = Gtk.Template.Child()
    check_button_post_export_custom_command: Gtk.CheckButton = Gtk.Template.Child()
    entry_row_post_export_custom_command: Adw.EntryRow = Gtk.Template.Child()
    check_button_show_mosaic_detections: Gtk.CheckButton = Gtk.Template.Child()
    switch_row_seek_preview = Gtk.Template.Child()
    switch_row_fp16: Adw.SwitchRow = Gtk.Template.Child()
    switch_row_detect_faces = Gtk.Template.Child()
    expander_row_encoding_presets: Adw.ExpanderRow = Gtk.Template.Child()
    expander_row_detection_models: Adw.ExpanderRow = Gtk.Template.Child()
    expander_row_restoration_models: Adw.ExpanderRow = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config: Config | None = None
        self.init_done = False
        self._show_playback_section = True
        self._show_export_section = True
        self._active_preset_button_group: Gtk.CheckButton | None = None
        self._create_preset_action_row: Adw.ActionRow | None = None
        self._presets_radio_buttons: list[Gtk.CheckButton] = []
        self._presets_action_rows: list[Adw.ActionRow] = []
        self._detection_models_actions_rows: list[Adw.ActionRow] = []
        self._restoration_models_actions_rows: list[Adw.ActionRow] = []

    def init_sidebar_from_config(self, config: Config):
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
        for row in self._restoration_models_actions_rows:
            self.expander_row_restoration_models.remove(row)
        self._restoration_models_actions_rows = self.get_action_rows_for_restoration_model(config.mosaic_restoration_model)
        for row in self._restoration_models_actions_rows:
            self.expander_row_restoration_models.add_row(row)
        self.expander_row_restoration_models.set_subtitle(config.mosaic_restoration_model)

        # init detection model
        for row in self._detection_models_actions_rows:
            self.expander_row_detection_models.remove(row)
        self._detection_models_actions_rows = self.get_action_rows_for_detection_model(config.mosaic_detection_model)
        for row in self._detection_models_actions_rows:
            self.expander_row_detection_models.add_row(row)
        self.expander_row_detection_models.set_subtitle(config.mosaic_detection_model)

        # init encoding presets
        selected_preset = utils.get_selected_preset(config)
        self._active_preset_button_group = Gtk.CheckButton.new()
        self.expander_row_encoding_presets.set_subtitle(selected_preset.description if selected_preset.user_preset else _(selected_preset.description))
        while len(self._presets_action_rows) > 0:
            self.delete_preset_row(0)
        assert len(self._presets_action_rows) == 0 and len(self._presets_radio_buttons) == 0
        if self._create_preset_action_row:
            self.expander_row_encoding_presets.remove(self._create_preset_action_row)
        presets = []
        presets.extend(video_utils.get_encoding_presets())
        presets.extend(config.custom_encoding_presets)
        for preset in presets:
            active = False
            if preset.name == config.encoding_preset_name:
                active = True
            action_row, radio_button = self.get_action_row_for_existing_preset(preset, active=active, localized_description=True)
            self.add_preset_row(action_row, radio_button)

        self._create_preset_action_row = self.get_action_row_for_add_new_preset()
        self.expander_row_encoding_presets.add_row(self._create_preset_action_row)

        self.spin_row_preview_buffer_duration.set_value(config.preview_buffer_duration)
        self.spin_row_clip_max_duration.set_value(config.max_clip_duration)
        self.switch_row_mute_audio.set_active(config.mute_audio)

        self.switch_row_seek_preview.set_active(config.seek_preview_enabled)
        self.switch_row_fp16.set_active(config.fp16_enabled)
        self.switch_row_detect_faces.set_active(config.detect_face_mosaics)
        self.switch_row_detect_faces.set_visible(config.mosaic_detection_model != 'v2')
        self.switch_row_mp4_fast_start.set_active(config.mp4_fast_start)

        # init color scheme
        if config.color_scheme == ColorScheme.LIGHT: self.light_color_scheme_button.set_property("active", True)
        elif config.color_scheme == ColorScheme.DARK: self.dark_color_scheme_button.set_property("active", True)
        else: self.system_color_scheme_button.set_property("active", True)

        # init export directory
        if config.export_directory:
            self.action_row_export_directory.set_subtitle(config.export_directory)
            self.expander_row_export_directory.set_subtitle(config.export_directory)
            self.check_button_export_directory_defaultdir.set_active(True)
        else:
            self.action_row_export_directory.set_subtitle(_("Click the folder button to choose a default"))
            self.expander_row_export_directory.set_subtitle(self.action_row_export_directory_alwaysask.get_title())
            self.check_button_export_directory_alwaysask.set_active(True)

        self.entry_row_file_name_pattern.set_text(config.file_name_pattern)

        # init temp directory
        self.action_row_temp_directory.set_subtitle(config.temp_directory)

        self.toggle_button_initial_view_preview.set_active(config.initial_view == "preview")
        self.toggle_button_initial_view_export.set_active(config.initial_view == "export")

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
    def combo_row_gpu_selected_callback(self, combo_row, value):
        selected_item = combo_row.get_property("selected_item")
        if selected_item is None:
            # CPU device, no GPU available
            return
        selected_gpu_name = selected_item.get_string()
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
            self.expander_row_export_directory.set_subtitle(self.action_row_export_directory_alwaysask.get_title())

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
        is_valid = validate_file_name_pattern(self.entry_row_file_name_pattern.get_text())
        utils.set_validation_css_classes(self.entry_row_file_name_pattern, is_valid)
        if is_valid:
            self._config.file_name_pattern = self.entry_row_file_name_pattern.get_text()

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def entry_row_file_name_pattern_focused_callback(self, row_entry, param_spec):
        is_valid = validate_file_name_pattern(self.entry_row_file_name_pattern.get_text())
        utils.set_validation_css_classes(self.entry_row_file_name_pattern, is_valid)

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

    @Gtk.Template.Callback()
    @skip_if_uninitialized
    def switch_row_mp4_fast_start_active_callback(self, switch_row, active):
        self._config.mp4_fast_start = switch_row.get_property("active")

    @skip_if_uninitialized
    def button_create_preset_callback(self, button):
        preset = utils.get_next_custom_preset(self.config)
        dialog = EncodingPresetDialog(preset, self.config, True)
        dialog.connect("preset-changed", self.on_preset_created)
        dialog.present(self)

    @skip_if_uninitialized
    def button_edit_preset_callback(self, button, preset_name: str, action_row: Adw.ActionRow):
        preset_now = utils.get_preset_by_name(self.config, preset_name)
        preset_before = preset_now.clone()
        preset_after = preset_now.clone()
        dialog = EncodingPresetDialog(preset_after, self.config, False)
        dialog.connect("preset-changed", self.on_preset_changed, preset_before, action_row)
        dialog.present(self)

    @skip_if_uninitialized
    def button_delete_preset_callback(self, button, preset_name: str, action_row: Adw.ActionRow):
        preset = utils.get_preset_by_name(self.config, preset_name)
        idx = self._presets_action_rows.index(action_row)
        is_last = len(self._presets_radio_buttons) - 1 == idx
        new_selection_idx = idx - 1 if is_last else idx + 1
        new_selected_preset_check_button = self._presets_radio_buttons[new_selection_idx]
        self.delete_preset_row(idx)
        new_selected_preset_check_button.set_active(True)

        updated_presets = set(self._config.custom_encoding_presets)
        updated_presets.remove(preset)
        self._config.custom_encoding_presets = updated_presets

    def add_preset_row(self, action_row, radio_button):
        self.expander_row_encoding_presets.add_row(action_row)
        self._presets_action_rows.append(action_row)
        self._presets_radio_buttons.append(radio_button)

    def delete_preset_row(self, idx):
        self.expander_row_encoding_presets.remove(self._presets_action_rows[idx])
        del self._presets_action_rows[idx]
        del self._presets_radio_buttons[idx]

    def on_preset_selected(self, _check_button, preset_name: str):
        preset = utils.get_preset_by_name(self.config, preset_name)
        self.expander_row_encoding_presets.set_subtitle(preset.description if preset.user_preset else _(preset.description))
        self._config.encoding_preset_name = preset_name

    def on_preset_changed(self, _dialog, preset_now: EncodingPreset, preset_old: EncodingPreset, action_row: Adw.ActionRow):
        updated_presets = set(self._config.custom_encoding_presets)
        updated_presets.remove(preset_old)
        updated_presets.add(preset_now)
        self._config.custom_encoding_presets = updated_presets

        is_preset_selected = self.expander_row_encoding_presets.get_subtitle() == preset_old.description
        if is_preset_selected:
            self.expander_row_encoding_presets.set_subtitle(preset_now.description)

        action_row.set_title(preset_now.description)

    def on_preset_created(self, _dialog, preset: EncodingPreset):
        updated_presets = set(self._config.custom_encoding_presets)
        updated_presets.add(preset)
        self._config.custom_encoding_presets = updated_presets

        action_row, radio_button = self.get_action_row_for_existing_preset(preset, active=True, localized_description=False)
        self.expander_row_encoding_presets.remove(self._create_preset_action_row)
        self.add_preset_row(action_row, radio_button)
        self.expander_row_encoding_presets.add_row(self._create_preset_action_row)

        self.expander_row_encoding_presets.set_subtitle(preset.description)

    def _get_action_row_for_model(self, modelfile: ModelFile, selected_radio_button: Gtk.CheckButton, selected_model_name, on_toggled: Callable[[Gtk.CheckButton, str], None]):
        action_row = Adw.ActionRow.new()
        action_row.set_title(modelfile.name)
        if modelfile.description:
            action_row.set_subtitle(modelfile.description)

        if modelfile.name != selected_model_name:
            radio_button = Gtk.CheckButton.new()
            radio_button.set_group(selected_radio_button)
            radio_button.set_active(False)
            radio_button.connect("toggled", on_toggled, modelfile.name)
            action_row.add_prefix(radio_button)
        else:
            action_row.add_prefix(selected_radio_button)
        return action_row

    def get_action_rows_for_restoration_model(self, selected_model: str) -> list[Adw.ActionRow]:
        def on_toggled(_button, model_name):
            self._config.mosaic_restoration_model = model_name
            self.expander_row_restoration_models.set_subtitle(model_name)
        active_button = Gtk.CheckButton.new()
        active_button.set_active(True)
        active_button.connect("toggled", on_toggled, selected_model)
        return [self._get_action_row_for_model(modelfile, active_button, selected_model, on_toggled) for modelfile in ModelFiles.get_restoration_models()]

    def get_action_rows_for_detection_model(self, selected_model: str) -> list[Adw.ActionRow]:
        def on_toggled(_button, model_name):
            self._config.mosaic_detection_model = model_name
            self.expander_row_detection_models.set_subtitle(model_name)
            self.switch_row_detect_faces.set_visible(self._config.mosaic_detection_model != 'v2')
        active_button = Gtk.CheckButton.new()
        active_button.set_active(True)
        active_button.connect("toggled", on_toggled, selected_model)
        return [self._get_action_row_for_model(modelfile, active_button, selected_model, on_toggled) for modelfile in ModelFiles.get_detection_models()]

    def get_action_row_for_existing_preset(self, preset: EncodingPreset, active: bool, localized_description: bool) -> tuple[Adw.ActionRow, Gtk.CheckButton]:
        action_row = Adw.ActionRow.new()
        action_row.set_title(_(preset.description) if localized_description else preset.description)

        radio_button = Gtk.CheckButton.new()
        radio_button.set_group(self._active_preset_button_group)
        radio_button.set_active(active)
        radio_button.connect("toggled", self.on_preset_selected, preset.name)
        action_row.add_prefix(radio_button)

        if preset.user_preset:
            edit_button = Gtk.Button.new()
            edit_button.set_icon_name("edit-symbolic")
            edit_button.set_valign(Gtk.Align.CENTER)
            edit_button.connect("clicked", self.button_edit_preset_callback, preset.name, action_row)
            context = edit_button.get_style_context()
            context.add_class("flat")
            action_row.add_suffix(edit_button)

            delete_button = Gtk.Button.new()
            delete_button.set_icon_name("cross-large-symbolic")
            delete_button.set_valign(Gtk.Align.CENTER)
            delete_button.connect("clicked", self.button_delete_preset_callback, preset.name, action_row)
            context = delete_button.get_style_context()
            context.add_class("flat")
            action_row.add_suffix(delete_button)

        return action_row, radio_button

    def get_action_row_for_add_new_preset(self) -> Adw.ActionRow:
        action_row = Adw.ActionRow.new()
        action_row.set_title(_("Create Preset…"))
        button_create_preset = Gtk.Button.new()
        button_create_preset.set_icon_name("plus-large-symbolic")
        button_create_preset.set_valign(Gtk.Align.CENTER)
        button_create_preset.connect("clicked", self.button_create_preset_callback)
        action_row.add_suffix(button_create_preset)
        return action_row

    def show_select_folder(self):
        file_dialog = Gtk.FileDialog()
        file_dialog.set_title(_("Select a folder where restored videos should be saved"))
        def on_select_folder(_file_dialog, result):
            try:
                selected_folder: Gio.File = _file_dialog.select_folder_finish(result)
                selected_folder_path = selected_folder.get_path()
                self._config.export_directory = selected_folder_path
                self.action_row_export_directory.set_subtitle(selected_folder_path)
                self.expander_row_export_directory.set_subtitle(selected_folder_path)
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
