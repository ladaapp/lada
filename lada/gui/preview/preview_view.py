import logging
import pathlib
import threading
from gettext import gettext as _

from gi.repository import Gtk, GObject, GLib, Gio, Gst, Adw

from lada import LOG_LEVEL
from lada.gui import utils
from lada.gui.config.config import Config
from lada.gui.config.config_sidebar import ConfigSidebar
from lada.gui.config.no_gpu_banner import NoGpuBanner
from lada.gui.frame_restorer_provider import FrameRestorerProvider, FrameRestorerOptions, FRAME_RESTORER_PROVIDER
from lada.gui.preview.fullscreen_mouse_activity_controller import FullscreenMouseActivityController
from lada.gui.preview.gstreamer_pipeline_manager import PipelineManager, PipelineState
from lada.gui.preview.headerbar_files_drop_down import HeaderbarFilesDropDown
from lada.gui.preview.timeline import Timeline
from lada.gui.shortcuts import ShortcutsManager
from lada.lib import audio_utils, video_utils

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

@Gtk.Template(filename=here / 'preview_view.ui')
class PreviewView(Gtk.Widget):
    __gtype_name__ = 'PreviewView'

    button_play_pause = Gtk.Template.Child()
    button_mute_unmute = Gtk.Template.Child()
    picture_video_preview: Gtk.Picture = Gtk.Template.Child()
    widget_timeline: Timeline = Gtk.Template.Child()
    button_image_play_pause = Gtk.Template.Child()
    button_image_mute_unmute = Gtk.Template.Child()
    label_current_time = Gtk.Template.Child()
    label_cursor_time = Gtk.Template.Child()
    box_playback_controls = Gtk.Template.Child()
    box_video_preview = Gtk.Template.Child()
    drop_down_files: HeaderbarFilesDropDown = Gtk.Template.Child()
    spinner_overlay = Gtk.Template.Child()
    banner_no_gpu: NoGpuBanner = Gtk.Template.Child()
    config_sidebar: ConfigSidebar = Gtk.Template.Child()
    header_bar: Adw.HeaderBar = Gtk.Template.Child()
    button_toggle_fullscreen: Gtk.Button = Gtk.Template.Child()
    stack_video_preview: Gtk.Stack = Gtk.Template.Child()
    view_switcher: Adw.ViewSwitcher = Gtk.Template.Child()
    button_open_files: Gtk.Button = Gtk.Template.Child()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._frame_restorer_options: FrameRestorerOptions | None = None
        self._video_preview_init_done = False
        self._buffer_queue_min_thresh_time = 0
        self._buffer_queue_min_thresh_time_auto_min = 2.
        self._buffer_queue_min_thresh_time_auto_max = 8.
        self._buffer_queue_min_thresh_time_auto = self._buffer_queue_min_thresh_time_auto_min
        self._shortcuts_manager: ShortcutsManager | None = None

        self.eos = False

        self.frame_restorer_provider: FrameRestorerProvider = FRAME_RESTORER_PROVIDER
        self.file_duration_ns = 0
        self.frame_duration_ns = None
        self.files: list[Gio.File] = []
        self.video_metadata: video_utils.VideoMetadata | None = None
        self.has_audio: bool = True
        self.should_be_paused = False
        self.seek_in_progress = False
        self.waiting_for_data = False

        self._config: Config | None = None

        self.widget_timeline.connect('seek_requested', lambda widget, seek_position: self.seek_video(seek_position))
        self.widget_timeline.connect('cursor_position_changed', lambda widget, cursor_position: self.show_cursor_position(cursor_position))

        self.fullscreen_mouse_activity_controller = None

        self.pipeline_manager: PipelineManager | None = None

        self.stack_video_preview.set_visible_child_name("spinner")

        self._view_stack: Adw.ViewStack | None = None

        self.drop_down_selected_handler_id = self.drop_down_files.connect("notify::selected", lambda obj, spec: self.play_file(obj.get_property(spec.name)))

        self.setup_double_click_fullscreen()

        drop_target = utils.create_video_files_drop_target(lambda files: self.emit("files-opened", files))
        self.add_controller(drop_target)

        def on_files_opened(obj, files):
            self.button_open_files.set_sensitive(True)
            self.add_files(files)
            if self._video_preview_init_done:
                last_file_idx = len(self.files) - 1
                if self.drop_down_files.get_selected() != last_file_idx:
                    self.drop_down_files.handler_block(self.drop_down_selected_handler_id)
                    self.drop_down_files.set_selected(last_file_idx)
                    self.drop_down_files.handler_unblock(self.drop_down_selected_handler_id)
                    self.play_file(last_file_idx)
            else:
                self.drop_down_files.set_sensitive(False)
        self.connect("files-opened", on_files_opened)

    @GObject.Property(type=Config)
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
        self.setup_config_signal_handlers()

    @GObject.Property()
    def buffer_queue_min_thresh_time(self):
        return self._buffer_queue_min_thresh_time

    @buffer_queue_min_thresh_time.setter
    def buffer_queue_min_thresh_time(self, value):
        if self._buffer_queue_min_thresh_time == value:
            return
        self._buffer_queue_min_thresh_time = value
        if self._video_preview_init_done:
            self.update_gst_buffers()

    @GObject.Property(type=ShortcutsManager)
    def shortcuts_manager(self):
        return self._shortcuts_manager

    @shortcuts_manager.setter
    def shortcuts_manager(self, value):
        self._shortcuts_manager = value
        self._setup_shortcuts()

    @GObject.Property(type=Adw.ViewStack)
    def view_stack(self):
        return self._view_stack

    @view_stack.setter
    def view_stack(self, value: Adw.ViewStack):
        self._view_stack = value
        def on_visible_child_name_changed(object, spec):
            visible_child_name = object.get_property(spec.name)
            if visible_child_name != "preview":
                self.should_be_paused = True
                self.pause_if_currently_playing()
            elif visible_child_name == "preview" and not self._video_preview_init_done:
                self.play_file(0)
        self._view_stack.connect("notify::visible-child-name", on_visible_child_name_changed)

    @GObject.Signal(name="toggle-fullscreen-requested")
    def toggle_fullscreen_requested(self):
        pass

    @GObject.Signal(name="files-opened", arg_types=(GObject.TYPE_PYOBJECT,))
    def files_opened_signal(self, files: list[Gio.File]):
        pass

    @Gtk.Template.Callback()
    def button_toggle_fullscreen_callback(self, button_clicked):
        self.emit("toggle-fullscreen-requested")

    @Gtk.Template.Callback()
    def button_play_pause_callback(self, button_clicked):
        if not self._video_preview_init_done or self.seek_in_progress:
            return

        if self.pipeline_manager.state == PipelineState.PLAYING:
            self.should_be_paused = True
            self.pipeline_manager.pause()
        elif self.pipeline_manager.state == PipelineState.PAUSED:
            self.should_be_paused = False
            if self.eos:
                self.seek_video(0)
            self.pipeline_manager.play()
        else:
            logger.warning(f"unhandled pipeline state in button_play_pause_callback: {self.pipeline_manager.state}")

    @Gtk.Template.Callback()
    def button_mute_unmute_callback(self, button_clicked):
        if not (self.has_audio and self._video_preview_init_done):
            return
        new_mute_state = not self.pipeline_manager.muted
        self.pipeline_manager.muted = new_mute_state
        self.set_speaker_icon(new_mute_state)

    @Gtk.Template.Callback()
    def button_open_files_callback(self, button_clicked):
        self.button_open_files.set_sensitive(False)
        callback = lambda files: self.emit("files-opened", files)
        dismissed_callback = lambda *args: self.button_open_files.set_sensitive(True)
        utils.show_open_files_dialog(callback, dismissed_callback)

    @property
    def frame_restorer_options(self):
        return self._frame_restorer_options

    @frame_restorer_options.setter
    def frame_restorer_options(self, value: FrameRestorerOptions):
        if self._frame_restorer_options == value:
            return
        if self._video_preview_init_done and self._buffer_queue_min_thresh_time == 0 and self._frame_restorer_options.max_clip_length != value.max_clip_length:
            self.buffer_queue_min_thresh_time_auto = float(value.max_clip_length / value.video_metadata.video_fps_exact)
        self._frame_restorer_options = value
        if self._video_preview_init_done:
            self.reset_appsource_worker()

    @property
    def buffer_queue_min_thresh_time_auto(self):
        return self._buffer_queue_min_thresh_time_auto

    @buffer_queue_min_thresh_time_auto.setter
    def buffer_queue_min_thresh_time_auto(self, value):
        value = min(self._buffer_queue_min_thresh_time_auto_max, max(self._buffer_queue_min_thresh_time_auto_min, value))
        if self._buffer_queue_min_thresh_time_auto == value:
            return
        logger.info(f"adjusted buffer_queue_min_thresh_time_auto to {value}")
        self._buffer_queue_min_thresh_time_auto = value
        if self._video_preview_init_done:
            self.update_gst_buffers()

    def setup_double_click_fullscreen(self):
            click_gesture = Gtk.GestureClick()
            def on_click(click_obj, n_press, x, y):
                if n_press == 2:
                    # double-click
                    self.emit("toggle-fullscreen-requested")
            click_gesture.connect( "pressed", on_click)
            self.box_video_preview.add_controller(click_gesture)

    def setup_config_signal_handlers(self):
        def on_show_mosaic_detections(*args):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_mosaic_detection(self._config.show_mosaic_detections)
        self._config.connect("notify::show-mosaic-detections", on_show_mosaic_detections)

        def on_device(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_device(self._config.device)
        self._config.connect("notify::device", on_device)

        def on_mosaic_restoration_model(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_mosaic_restoration_model_name(self._config.mosaic_restoration_model)
        self._config.connect("notify::mosaic-restoration-model", on_mosaic_restoration_model)

        def on_mosaic_detection_model(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_mosaic_detection_model_name(self._config.mosaic_detection_model)
        self._config.connect("notify::mosaic-detection-model", on_mosaic_detection_model)

        self._config.connect("notify::preview-buffer-duration", lambda object, spec: self.set_property('buffer-queue-min-thresh-time', object.get_property(spec.name)))

        def on_max_clip_duration(object, spec):
            if self._frame_restorer_options:
                self.frame_restorer_options = self._frame_restorer_options.with_max_clip_length(self._config.max_clip_duration)
        self._config.connect("notify::max-clip-duration", on_max_clip_duration)

    def set_speaker_icon(self, mute: bool):
        icon_name = "speaker-0-symbolic" if mute else "speaker-4-symbolic"
        self.button_image_mute_unmute.set_property("icon-name", icon_name)

    def update_gst_buffers(self):
        buffer_queue_min_thresh_time, buffer_queue_max_thresh_time = self.get_gst_buffer_bounds()
        self.pipeline_manager.update_gst_buffers(buffer_queue_min_thresh_time, buffer_queue_max_thresh_time)

    def seek_video(self, seek_position_ns):
        if self.seek_in_progress:
            return
        # pipeline.seek_simple is blocking. As we're stopping/starting our appsrc on seek this could potentially take a few seconds and freeze the UI
        def seek():
            self.eos = False
            self.seek_in_progress = True
            self.spinner_overlay.set_visible(True)
            self.label_current_time.set_text(self.get_time_label_text(seek_position_ns))
            self.widget_timeline.set_property("playhead-position", seek_position_ns)
            self.pipeline_manager.seek(seek_position_ns)
            self.seek_in_progress = False
            if not self.waiting_for_data:
                self.spinner_overlay.set_visible(False)

        seek_thread = threading.Thread(target=seek, daemon=True)
        seek_thread.start()

    def show_cursor_position(self, cursor_position_ns):
        if cursor_position_ns > 0:
            self.label_cursor_time.set_visible(True)
            label_text = self.get_time_label_text(cursor_position_ns)
            self.label_cursor_time.set_text(label_text)
        else:
            self.label_cursor_time.set_visible(False)

    def play_file(self, idx):
        self._show_spinner()
        self._reinit_open_file_async(self.files[idx])

    def add_files(self, files: list[Gio.File]):
        unique_files_to_add = []
        for file_to_add in files:
            if any(file_to_add.get_path() == file_already_added.get_path() for file_already_added in self.files):
                # duplicate
                continue
            self.files.append(file_to_add)
            unique_files_to_add.append(file_to_add)

        if len(unique_files_to_add) > 0:
            self.drop_down_files.handler_block(self.drop_down_selected_handler_id)
            self.drop_down_files.add_files(files)
            self.drop_down_files.handler_unblock(self.drop_down_selected_handler_id)

    def _reinit_open_file_async(self, file: Gio.File):
        def run():
            if self._video_preview_init_done:
                self._video_preview_init_done = False
                self.pipeline_manager.close_video_file()
            GLib.idle_add(lambda: self._open_file(file))

        threading.Thread(target=run).start()

    def _open_file(self, file: Gio.File):
        self.frame_restorer_options = FrameRestorerOptions(self.config.mosaic_restoration_model, self.config.mosaic_detection_model, video_utils.get_video_meta_data(file.get_path()), self.config.device, self.config.max_clip_duration, self.config.show_mosaic_detections, False)
        file_path = file.get_path()

        assert not self._video_preview_init_done
        self.video_metadata = video_utils.get_video_meta_data(file_path)
        self._frame_restorer_options = self._frame_restorer_options.with_video_metadata(self.video_metadata)
        self.has_audio = audio_utils.get_audio_codec(self.video_metadata.video_file) is not None
        self.button_mute_unmute.set_sensitive(self.has_audio)
        self.set_speaker_icon(mute=not self.has_audio or self.config.mute_audio)

        self.should_be_paused = False
        self.seek_in_progress = False
        self.waiting_for_data = False

        self.frame_duration_ns = (1 / self.video_metadata.video_fps) * Gst.SECOND
        self.file_duration_ns = int((self.video_metadata.frames_count * self.frame_duration_ns))
        self._buffer_queue_min_thresh_time_auto_min = float(self._frame_restorer_options.max_clip_length / self.video_metadata.video_fps_exact)
        self.buffer_queue_min_thresh_time_auto = self._buffer_queue_min_thresh_time_auto_min

        self.widget_timeline.set_property("duration", self.file_duration_ns)

        self.frame_restorer_provider.init(self._frame_restorer_options)

        if self.pipeline_manager:
            self.pipeline_manager.init_pipeline(self.video_metadata)
        else:
            buffer_queue_min_thresh_time, buffer_queue_max_thresh_time = self.get_gst_buffer_bounds()
            self.pipeline_manager = PipelineManager(self.frame_restorer_provider, buffer_queue_min_thresh_time, buffer_queue_max_thresh_time, self.config.mute_audio)
            self.pipeline_manager.init_pipeline(self.video_metadata)
            self.picture_video_preview.set_paintable(self.pipeline_manager.paintable)
            self.pipeline_manager.connect("eos", self.on_eos)
            self.pipeline_manager.connect("waiting-for-data", lambda obj, waiting_for_data: self.on_waiting_for_data(waiting_for_data))
            self.pipeline_manager.connect("notify::state", lambda object, spec: self.on_pipeline_state(object.get_property(spec.name)))
            GLib.timeout_add(100, self.update_current_position)

        threading.Thread(target=self.pipeline_manager.play).start()

    def on_eos(self, *args):
        self.eos = True
        self.button_image_play_pause.set_property("icon-name", "media-playback-start-symbolic")

    def on_pipeline_state(self, state: PipelineState):
        if state == PipelineState.PLAYING:
            self.button_image_play_pause.set_property("icon-name", "media-playback-pause-symbolic")
        elif state == PipelineState.PAUSED:
            self.button_image_play_pause.set_property("icon-name", "media-playback-start-symbolic")
        if not self._video_preview_init_done and state == PipelineState.PLAYING:
            self._video_preview_init_done = True
            self._show_video_preview()

    def pause_if_currently_playing(self):
        if not self._video_preview_init_done:
            return
        if self.pipeline_manager.state == PipelineState.PLAYING:
            self.should_be_paused = True
            self.pipeline_manager.pause()

    def grab_focus(self):
        self.button_play_pause.grab_focus()

    def on_waiting_for_data(self, waiting_for_data):
        self.waiting_for_data = waiting_for_data
        self.spinner_overlay.set_visible(waiting_for_data)
        if waiting_for_data:
            self.pipeline_manager.pause()
            if self._buffer_queue_min_thresh_time == 0 and self._video_preview_init_done:
                self.buffer_queue_min_thresh_time_auto *= 1.5
                self.update_gst_buffers()
        else:
            if not self.should_be_paused:
                self.pipeline_manager.play()
            elif not self._video_preview_init_done:
                # when app started in preview mode then user switched to export while still waiting for data
                self._video_preview_init_done = True
                self._show_video_preview()
                self.button_image_play_pause.set_property("icon-name", "media-playback-start-symbolic")

    def get_gst_buffer_bounds(self):
        buffer_queue_min_thresh_time = self._buffer_queue_min_thresh_time if self._buffer_queue_min_thresh_time > 0 else self._buffer_queue_min_thresh_time_auto
        buffer_queue_max_thresh_time = buffer_queue_min_thresh_time * 2
        return buffer_queue_min_thresh_time, buffer_queue_max_thresh_time

    def reset_appsource_worker(self):
        self._show_spinner()

        def reinit():
            self._video_preview_init_done = False
            self.pipeline_manager.pause()
            self.frame_restorer_provider.init(self._frame_restorer_options)
            self.pipeline_manager.reinit_appsrc()
            self.pipeline_manager.play()

        reinit_thread = threading.Thread(target=reinit)
        reinit_thread.start()

    def update_current_position(self):
        position = self.pipeline_manager.get_position_ns()
        if position is not None:
            label_text = self.get_time_label_text(position)
            self.label_current_time.set_text(label_text)
            self.widget_timeline.set_property("playhead-position", position)
        return True

    def get_time_label_text(self, time_ns):
        if not time_ns or time_ns == -1:
            return '00:00:00'
        else:
            seconds = int(time_ns / Gst.SECOND)
            minutes = int(seconds / 60)
            hours = int(minutes / 60)
            seconds = seconds % 60
            minutes = minutes % 60
            hours, minutes, seconds = int(hours), int(minutes), int(seconds)
            time = f"{minutes}:{seconds:02d}" if hours == 0 else f"{hours}:{minutes:02d}:{seconds:02d}"
            return time

    def on_fullscreen_activity(self, fullscreen_activity: bool):
        if fullscreen_activity:
            GLib.idle_add(lambda: self.header_bar.set_visible(True))
            self.set_cursor_from_name("default")
            self.box_playback_controls.set_visible(True)
            self.button_play_pause.grab_focus()
        else:
            GLib.idle_add(lambda: self.header_bar.set_visible(False))
            self.set_cursor_from_name("none")
            self.box_playback_controls.set_visible(False)

    def on_fullscreened(self, fullscreened: bool):
        if fullscreened:
            self.fullscreen_mouse_activity_controller = FullscreenMouseActivityController(self, self.box_video_preview)
            self.banner_no_gpu.set_revealed(False)
            self.button_toggle_fullscreen.set_property("icon-name", "view-restore-symbolic")
            self.box_video_preview.set_css_classes(["fullscreen-preview"])
        else:
            self.header_bar.set_visible(True)
            self.set_cursor_from_name("default")
            self.button_toggle_fullscreen.set_property("icon-name", "view-fullscreen-symbolic")
            self.box_playback_controls.set_visible(True)
            self.button_play_pause.grab_focus()
            self.box_video_preview.remove_css_class("fullscreen-preview")
            if self._config.get_property('device') == 'cpu':
                self.banner_no_gpu.set_revealed(True)
        self.fullscreen_mouse_activity_controller.on_fullscreened(fullscreened)
        self.fullscreen_mouse_activity_controller.connect("notify::fullscreen-activity", lambda object, spec: self.on_fullscreen_activity(object.get_property(spec.name)))

    def _show_spinner(self, *args):
        self.config_sidebar.set_property("disabled", True)
        self.drop_down_files.set_sensitive(False)
        self.view_switcher.set_sensitive(False)
        self.button_open_files.set_sensitive(False)
        self.stack_video_preview.set_visible_child_name("spinner")

    def _show_video_preview(self, *args):
        self.config_sidebar.set_property("disabled", False)
        self.drop_down_files.set_sensitive(True)
        self.view_switcher.set_sensitive(True)
        self.button_open_files.set_sensitive(True)
        self.stack_video_preview.set_visible_child_name("video-player")
        self.grab_focus()

    def _setup_shortcuts(self):
        self._shortcuts_manager.register_group("preview", _("Watch"))
        self._shortcuts_manager.add("preview", "toggle-mute-unmute", "m", lambda *args: self.button_mute_unmute_callback(self.button_mute_unmute), _("Mute/Unmute"))
        self._shortcuts_manager.add("preview", "toggle-play-pause", "<Ctrl>space", lambda *args: self.button_play_pause_callback(self.button_play_pause), _("Play/Pause"))
        self._shortcuts_manager.add("preview", "toggle-fullscreen", "f", lambda *args: self.emit("toggle-fullscreen-requested"), _("Enable/Disable fullscreen"))

    def close(self, block=False):
        if not self.pipeline_manager:
            return
        self._video_preview_init_done = False
        if block:
            self.pipeline_manager.close_video_file()
        else:
            GLib.idle_add(self.pipeline_manager.close_video_file)