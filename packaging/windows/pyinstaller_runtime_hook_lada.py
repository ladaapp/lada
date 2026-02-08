import os
import sys

base_path = sys._MEIPASS

paths = [base_path,os.path.join(base_path, "bin")]

intel_marker = os.path.join(base_path, "ur_adapter_level_zero.dll")
if os.path.exists(intel_marker):
    system_path = os.environ.get("PATH", "")
    if system_path:
        paths.append(system_path)

os.environ["PATH"] = os.pathsep.join(paths)
os.environ["LADA_MODEL_WEIGHTS_DIR"] = os.path.join(base_path, "model_weights")
os.environ["LOCALE_DIR"] = os.path.join(base_path, "lada", "locale")