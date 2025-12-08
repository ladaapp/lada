import os
import sys

if "LADA_MODEL_WEIGHTS_DIR" in os.environ:
  MODEL_WEIGHTS_DIR = os.environ["LADA_MODEL_WEIGHTS_DIR"]
else:
  MODEL_WEIGHTS_DIR = "model_weights"

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["YOLO_VERBOSE"] = "false"

VERSION = '0.9.1-dev'

LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")

RESTORATION_MODEL_FILES_TO_NAMES = {
    os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic.pth'): 'basicvsrpp-v1.0',
    os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic_v1.1.pth'): 'basicvsrpp-v1.1',
    os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_restoration_model_generic_v1.2.pth'): 'basicvsrpp-v1.2',
    os.path.join(MODEL_WEIGHTS_DIR, '3rd_party', 'clean_youknow_video.pth'): 'deepmosaics',
}
RESTORATION_MODEL_NAMES_TO_FILES = {v: k for k, v in RESTORATION_MODEL_FILES_TO_NAMES.items()}

DETECTION_MODEL_FILES_TO_NAMES = {
    os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v2.pt'): 'v2',
    os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v3.pt'): 'v3',
    os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v3.1_fast.pt'): 'v3.1-fast',
    os.path.join(MODEL_WEIGHTS_DIR, 'lada_mosaic_detection_model_v3.1_accurate.pt'): 'v3.1-accurate',
}
DETECTION_MODEL_NAMES_TO_FILES = {v: k for k, v in DETECTION_MODEL_FILES_TO_NAMES.items()}

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
        LOCALE_DIR = "lada/locale"
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


def get_available_restoration_models():
  available_models = []
  for file_path in RESTORATION_MODEL_FILES_TO_NAMES:
    if os.path.exists(file_path):
      available_models.append(RESTORATION_MODEL_FILES_TO_NAMES[file_path])
  return available_models


def get_available_detection_models():
  available_models = []
  for file_path in DETECTION_MODEL_FILES_TO_NAMES:
    if os.path.exists(file_path):
      available_models.append(DETECTION_MODEL_FILES_TO_NAMES[file_path])
  return available_models

_init_translations()