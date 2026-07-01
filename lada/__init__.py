import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from logging.handlers import RotatingFileHandler
from pathlib import Path

if "LADA_MODEL_WEIGHTS_DIR" in os.environ:
  MODEL_WEIGHTS_DIR = os.environ["LADA_MODEL_WEIGHTS_DIR"]
else:
  MODEL_WEIGHTS_DIR = "model_weights"

os.environ["ALBUMENTATIONS_OFFLINE"] = "1"
os.environ["ALBUMENTATIONS_NO_TELEMETRY"] = "1"
os.environ["YOLO_VERBOSE"] = "false"

def _get_version(version: str):
    if not version.endswith("dev"):
        return version

    try:
        import pathlib
        import subprocess
        from lada.utils import os_utils
        here = pathlib.Path(__file__).parent.resolve()

        commit_id_short = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(here), stderr=subprocess.DEVNULL, startupinfo=os_utils.get_subprocess_startup_info()).decode("ascii").strip()
        return f"{version}+{commit_id_short}"
    except Exception:
        return version

VERSION = _get_version('0.11.1-dev')

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
DEFAULT_LOG_MAX_BYTES = 20 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT = 5


def _get_config_dirs() -> list[Path]:
    """Config directories used by the GUI, with the primary path first."""
    if sys.platform == "win32":
        dirs = [
            Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "lada",
            Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "lada",
        ]
    elif sys.platform == "darwin":
        dirs = [Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "lada"]
    else:
        dirs = [Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "lada"]

    unique_dirs = []
    for directory in dirs:
        if directory not in unique_dirs:
            unique_dirs.append(directory)
    return unique_dirs


def _get_config_dir() -> Path:
    return _get_config_dirs()[0]


def _get_default_log_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "lada" / "logs"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "lada"
    else:
        state_home = os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
        return Path(state_home) / "lada" / "logs"


def _get_log_dir() -> Path:
    if "LADA_LOG_DIR" in os.environ:
        return Path(os.environ["LADA_LOG_DIR"])

    # Read log_directory from saved config so early-boot logs also go there
    for config_dir in _get_config_dirs():
        config_file = config_dir / "lada.conf"
        try:
            if config_file.exists():
                config = json.loads(config_file.read_text(encoding="utf-8"))
                configured = config.get("log_directory")
                if configured:
                    log_dir = Path(configured)
                    log_dir.mkdir(parents=True, exist_ok=True)
                    return log_dir
        except Exception:
            pass

    return _get_default_log_dir()


_LOG_FORMAT = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _get_log_rollover_settings() -> tuple[int, int]:
    try:
        max_bytes = int(os.environ.get("LADA_LOG_MAX_BYTES", DEFAULT_LOG_MAX_BYTES))
    except ValueError:
        max_bytes = DEFAULT_LOG_MAX_BYTES
    try:
        backup_count = int(os.environ.get("LADA_LOG_BACKUP_COUNT", DEFAULT_LOG_BACKUP_COUNT))
    except ValueError:
        backup_count = DEFAULT_LOG_BACKUP_COUNT
    return max(0, max_bytes), max(0, backup_count)


def _create_file_handler(log_file: Path) -> logging.FileHandler:
    max_bytes, backup_count = _get_log_rollover_settings()
    if max_bytes > 0 and backup_count > 0:
        return RotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    return logging.FileHandler(str(log_file), encoding="utf-8")


def set_log_file(log_file_path: str | Path, propagate_directory: bool = True) -> None:
    """Switch the root logger's file handler to a specific log file."""
    root = logging.getLogger()
    log_file = Path(log_file_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    os.environ["LADA_LOG_FILE"] = str(log_file)
    if propagate_directory:
        os.environ["LADA_LOG_DIR"] = str(log_file.parent)

    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()

    file_handler = _create_file_handler(log_file)
    file_handler.setFormatter(_LOG_FORMAT)
    root.addHandler(file_handler)


def set_log_directory(new_path: str | None) -> None:
    """Switch the root logger's file handler and propagate it to child processes."""
    new_dir = Path(new_path) if new_path else _get_default_log_dir()
    new_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LADA_LOG_DIR"] = str(new_dir)
    new_log_file = new_dir / "lada.log"
    set_log_file(new_log_file)


def _setup_logging():
    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LADA_LOG_DIR"] = str(log_dir)
    log_file = Path(os.environ["LADA_LOG_FILE"]) if "LADA_LOG_FILE" in os.environ else log_dir / "lada.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(_LOG_FORMAT)
    root.addHandler(stream_handler)

    file_handler = _create_file_handler(log_file)
    file_handler.setFormatter(_LOG_FORMAT)
    root.addHandler(file_handler)

    # Notify early (before any logger is active) where logs are written
    print(
        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Logging to {log_file}",
        file=sys.stderr,
    )


_setup_logging()

IS_FLATPAK = "FLATPAK_ID" in os.environ and "XDG_RUNTIME_DIR" in os.environ
if IS_FLATPAK and "TMPDIR" not in os.environ:
    # The path $XDG_RUNTIME_DIR/app/$FLATPAK_ID should be accessible from inside the sandbox as well as from the host
    # We need this so that the media player launched on the host is able to access the file if .mp4 Fast start is enabled and the Export Preview button is clicked.
    os.environ["TMPDIR"] = os.path.join(os.environ["XDG_RUNTIME_DIR"], "app", os.environ["FLATPAK_ID"])

def _get_language_from_os() -> str:
    if sys.platform == "darwin":
        # source: https://github.com/gaphor/gaphor/blob/ba7f9092d57c5d23b727136f13923cc355204d96/gaphor/i18n.py#L30
        from Cocoa import NSUserDefaults

        defaults = NSUserDefaults.standardUserDefaults()
        langs = defaults.objectForKey_("AppleLanguages")
        if language := langs.objectAtIndex_(0):
            assert isinstance(language, str)
            return language.replace("-", "_")
    elif sys.platform == "win32":
        # source: https://stackoverflow.com/questions/3425294/how-to-detect-the-os-default-language-in-python/25691701#25691701
        import ctypes
        import locale
        windll = ctypes.windll.kernel32
        if language := locale.windows_locale.get(windll.GetUserDefaultUILanguage()):
            return language
    return ""

def _init_translations():
    import gettext
    DOMAIN = 'lada'
    if "LOCALE_DIR" in os.environ:
        LOCALE_DIR = os.environ["LOCALE_DIR"]
    else:
        LOCALE_DIR = os.path.join(os.path.dirname(__file__), "locale")
    is_language_set = False
    for var_name in ["LANGUAGE", "LANG"]:
        if var_name in os.environ:
            is_language_set = True
            break
    if not is_language_set:
        os.environ["LANGUAGE"] = _get_language_from_os()
    gettext.bindtextdomain(DOMAIN, LOCALE_DIR)
    gettext.textdomain(DOMAIN)
    gettext.install(DOMAIN, LOCALE_DIR)

_init_translations()

@dataclass(frozen=True)
class ModelFile:
    name: str
    description: str | None
    path: str

class ModelFiles:
    _WELL_KNOWN_RESTORATION_MODELS = [
        ModelFile('basicvsrpp-v1.0', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic.pth')),
        ModelFile('basicvsrpp-v1.1', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic_v1.1.pth')),
        ModelFile('basicvsrpp-v1.2', _("Latest Lada restoration model. Recommended"), os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic_v1.2.pth')),
        ModelFile('deepmosaics', _("Restoration model from abandoned DeepMosaics project"), os.path.join(MODEL_WEIGHTS_DIR, '3rd_party', 'clean_youknow_video.pth')),
    ]
    _WELL_KNOWN_DETECTION_MODELS = [
        ModelFile('v2', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v2.pt')),
        ModelFile('v3', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v3.pt')),
        ModelFile('v3.1-fast', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v3.1_fast.pt')),
        ModelFile('v3.1-accurate', None, os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v3.1_accurate.pt')),
        ModelFile('v4-fast', _("Fast and efficient. Recommended"), os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v4_fast.pt')),
        ModelFile('v4-accurate', _("Can be slightly more accurate than v4-fast but slower"), os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v4_accurate.pt')),
    ]

    @staticmethod
    def _get_custom_detection_models() -> list[ModelFile]:
        models = []
        if not os.path.exists(MODEL_WEIGHTS_DIR):
            return models
        well_known_filenames = [os.path.basename(model.path) for model in ModelFiles._WELL_KNOWN_DETECTION_MODELS]
        for filename in os.listdir(MODEL_WEIGHTS_DIR):
            if filename.endswith('.pt') and filename.startswith('lada_mosaic_detection_model_') and filename not in well_known_filenames:
                model_name = os.path.splitext(filename)[0].split("lada_mosaic_detection_model_")[1]
                if len(model_name) == 0:
                    continue
                model_path = os.path.join(MODEL_WEIGHTS_DIR, filename)
                models.append(ModelFile(model_name, None, model_path))
        return models

    @staticmethod
    def _get_custom_restoration_models() -> list[ModelFile]:
        models = []
        if not os.path.exists(MODEL_WEIGHTS_DIR):
            return models
        well_known_filenames = [os.path.basename(model.path) for model in ModelFiles._WELL_KNOWN_RESTORATION_MODELS]
        for filename in os.listdir(MODEL_WEIGHTS_DIR):
            if filename.endswith('.pth') and filename.startswith('lada_mosaic_restoration_model_') and filename not in well_known_filenames:
                model_name = os.path.splitext(filename)[0].split("lada_mosaic_restoration_model_")[1]
                if len(model_name) == 0:
                    continue
                if not model_name.startswith("deepmosaics") and "deepmosaics" in model_name:
                    model_name = f"deepmosaics-{model_name}"
                elif not model_name.startswith("basicvsrpp"):
                    model_name = f"basicvsrpp-{model_name}"
                model_path = os.path.join(MODEL_WEIGHTS_DIR, filename)
                models.append(ModelFile(model_name, None, model_path))
        return models

    @staticmethod
    def _get_well_known_detection_models():
        models = []
        for model in ModelFiles._WELL_KNOWN_DETECTION_MODELS:
            if os.path.exists(model.path):
                models.append(model)
        return models

    @staticmethod
    def _get_well_known_restoration_models():
        models = []
        for model in ModelFiles._WELL_KNOWN_RESTORATION_MODELS:
            if os.path.exists(model.path):
                models.append(model)
        return models

    @staticmethod
    @cache
    def get_detection_models() -> list[ModelFile]:
        return ModelFiles._get_well_known_detection_models() + ModelFiles._get_custom_detection_models()

    @staticmethod
    @cache
    def get_restoration_models() -> list[ModelFile]:
        return ModelFiles._get_well_known_restoration_models() + ModelFiles._get_custom_restoration_models()

    @staticmethod
    def get_restoration_model_by_name(model_name: str) -> ModelFile | None:
        for model in ModelFiles.get_restoration_models():
            if model.name == model_name:
                return model
        return None

    @staticmethod
    def get_detection_model_by_name(model_name: str) -> ModelFile | None:
        for model in ModelFiles.get_detection_models():
            if model.name == model_name:
                return model
        return None

    @staticmethod
    def get_detection_model_by_path(model_path: str) -> ModelFile | None:
        for model in ModelFiles.get_detection_models():
            if model.path == model_path:
                return model
        return None
