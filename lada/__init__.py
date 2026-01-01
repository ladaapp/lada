import os
import sys
from dataclasses import dataclass
from functools import cache

if "LADA_MODEL_WEIGHTS_DIR" in os.environ:
  MODEL_WEIGHTS_DIR = os.environ["LADA_MODEL_WEIGHTS_DIR"]
else:
  MODEL_WEIGHTS_DIR = "model_weights"

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["YOLO_VERBOSE"] = "false"

VERSION = '0.10.0'

LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")

is_running_inside_flatpak = "FLATPAK_ID" in os.environ and "XDG_RUNTIME_DIR" in os.environ
if is_running_inside_flatpak and "TMPDIR" not in os.environ:
    # The path $XDG_RUNTIME_DIR/app/$FLATPAK_ID should be accessible from inside the sandbox as well as from the host
    # We need this so that the media player launched on the host is able to access the file if .mp4 Fast start is enabled and the Export Preview button is clicked.
    os.environ["TMPDIR"] = os.path.join(os.environ["XDG_RUNTIME_DIR"], "app", os.environ["FLATPAK_ID"])

def _get_language_from_os() -> str:
    if sys.platform == "win32":
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