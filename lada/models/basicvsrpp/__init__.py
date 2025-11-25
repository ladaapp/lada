# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

def register_all_modules():
    from lada.models.basicvsrpp.mmagic import register_all_modules
    register_all_modules()
    from lada.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet, BasicVSRPlusPlusGan
    from lada.models.basicvsrpp.mosaic_video_dataset import MosaicVideoDataset