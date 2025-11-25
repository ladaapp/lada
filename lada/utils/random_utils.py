# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import random

import numpy as np

repeatable_rng_random = random.Random(42)
repeatable_rng_numpy = np.random.RandomState(42)

def get_rngs(repeatable) -> tuple[random, np.random]:
    if repeatable:
        rng_random = repeatable_rng_random
        rng_numpy = repeatable_rng_numpy
    else:
        rng_random = random
        rng_numpy = np.random
    return rng_random, rng_numpy