from gi.repository import GObject, Gio

class ExportItemState(GObject.GEnum):
    QUEUED = 0
    PROCESSING = 1
    FINISHED = 2
    FAILED = 3
    PAUSED = 4

class ExportItemDataProgress(GObject.Object):
    __gtype_name__ = 'ExportItemDataProgress'

    def __init__(self):
        super().__init__()

        self._fraction: float = 0.0
        self._time_done_s: float = 0.0
        self._time_remaining_s: float = 0.0
        self._frames_done: int = 0
        self._frames_remaining: int = 0
        self._speed_fps: float = 0.0
        self._enough_datapoints = False

    @GObject.Property(type=float)
    def fraction(self):
        return self._fraction

    @fraction.setter
    def fraction(self, value):
        self._fraction = value

    @GObject.Property(type=float)
    def time_done_s(self):
        return self._time_done_s

    @time_done_s.setter
    def time_done_s(self, value):
        self._time_done_s = value

    @GObject.Property(type=float)
    def time_remaining_s(self):
        return self._time_remaining_s

    @time_remaining_s.setter
    def time_remaining_s(self, value):
        self._time_remaining_s = value
        
    @GObject.Property(type=float)
    def frames_done(self):
        return self._frames_done

    @frames_done.setter
    def frames_done(self, value):
        self._frames_done = int(value)

    @GObject.Property(type=float)
    def frames_remaining(self):
        return self._frames_remaining

    @frames_remaining.setter
    def frames_remaining(self, value):
        self._frames_remaining = int(value)

    @GObject.Property(type=float)
    def speed_fps(self):
        return self._speed_fps

    @speed_fps.setter
    def speed_fps(self, value):
        self._speed_fps = value

    @GObject.Property(type=bool, default=False)
    def enough_datapoints(self):
        return self._enough_datapoints

    @enough_datapoints.setter
    def enough_datapoints(self, value):
        self._enough_datapoints = value

    def complete(self):
        self._fraction = 1.0
        self._frames_remaining = 0
        self._time_remaining_s = 0.0
        self._speed_fps = 0.0

class ExportItemData(GObject.Object):
    __gtype_name__ = 'ExportItemData'

    def __init__(self, original_file: Gio.File, restored_file: Gio.File):
        super().__init__()

        self._progress: ExportItemDataProgress = ExportItemDataProgress()
        self._original_file: Gio.File = original_file
        self._restored_file: Gio.File = restored_file
        self._state: ExportItemState = ExportItemState.QUEUED
        self._error_details: str = ""

    @GObject.Property(type=ExportItemDataProgress)
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, value):
        self._progress = value

    @GObject.Property(type=ExportItemState, default=ExportItemState.QUEUED)
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @GObject.Property(type=Gio.File)
    def original_file(self):
        return self._original_file

    @GObject.Property(type=Gio.File)
    def restored_file(self):
        return self._restored_file

    @restored_file.setter
    def restored_file(self, value):
        self._restored_file = value

    @GObject.Property(type=str)
    def error_details(self):
        return self._error_details

    @error_details.setter
    def error_details(self, value):
        self._error_details = value

    def __repr__(self):
        return f"{{{self._original_file.get_basename()}, {self._restored_file.get_basename()}, {self._state}, {self._progress.fraction}}}"
