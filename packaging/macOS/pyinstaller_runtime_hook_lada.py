# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import os
import sys

base_path = sys._MEIPASS

paths = [base_path, os.path.join(base_path, "bin")]

os.environ["PATH"] = os.pathsep.join(paths)
os.environ["LADA_MODEL_WEIGHTS_DIR"] = os.path.join(base_path, "model_weights")
os.environ["LOCALE_DIR"] = os.path.join(base_path, "lada", "locale")
