# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import pathlib
import sys

from lada import IS_FLATPAK, LOG_LEVEL

load_fallback = False
try:
    import torch
except ModuleNotFoundError:
    if IS_FLATPAK:
        load_fallback = True
    else:
        raise

here = pathlib.Path(__file__).parent.resolve()

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

def main():
    if load_fallback:
        from lada.gui.missing_flatpak_extension_application import MissingFlatpakExtensionApplication
        app = MissingFlatpakExtensionApplication()
    else:
        from lada.gui.application import LadaApplication
        app = LadaApplication()
    try:
        return app.run(sys.argv)
    except KeyboardInterrupt:
        logger.info("Received Ctrl-C, quitting")
        app.on_shutdown()

if __name__ == "__main__":
    main()