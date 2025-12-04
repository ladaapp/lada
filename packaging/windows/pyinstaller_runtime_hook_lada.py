import os
import sys

base_path = sys._MEIPASS

os.environ["LADA_MODEL_WEIGHTS_DIR"] = os.path.join(base_path, "model_weights")
os.environ["PATH"] = os.path.join(base_path, "bin")
os.environ["LOCALE_DIR"] = os.path.join(base_path, "lada", "locale")