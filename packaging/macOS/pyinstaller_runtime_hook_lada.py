# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import os
import sys

if sys.platform != "darwin":
    sys.exit(1)

# When launched from Finder/Spotlight on macOS, cwd can be wrong; set to user home.
try:
    os.chdir(os.path.expanduser("~"))
except Exception:
    pass

# Prefer 'spawn' for multiprocessing in frozen apps to avoid fork-related issues.
try:
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

os.environ["PATH"] = os.pathsep.join([sys._MEIPASS, os.path.join(sys._MEIPASS, "bin")])
os.environ["LADA_MODEL_WEIGHTS_DIR"] = os.path.join(sys._MEIPASS, "model_weights")
os.environ["LOCALE_DIR"] = os.path.join(sys._MEIPASS, "lada", "locale")
