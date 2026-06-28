# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class StageTiming:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0


class StageTimer:
    def __init__(self, name: str):
        self.name = name
        self._stats: dict[str, StageTiming] = {}
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, time.perf_counter() - start)

    def record(self, stage: str, duration_s: float):
        with self._lock:
            timing = self._stats.setdefault(stage, StageTiming())
            timing.count += 1
            timing.total_s += duration_s
            timing.max_s = max(timing.max_s, duration_s)

    def log_summary(self, logger: logging.Logger, level: int = logging.INFO):
        with self._lock:
            stats = dict(self._stats)
        if not stats:
            return

        lines = []
        for stage, timing in sorted(stats.items(), key=lambda item: item[1].total_s, reverse=True):
            avg_ms = (timing.total_s / timing.count) * 1000
            lines.append(
                f"    {stage}: count={timing.count}, "
                f"total={timing.total_s:.3f}s, avg={avg_ms:.2f}ms, max={timing.max_s:.3f}s"
            )
        logger.log(level, "%s stage timings:\n%s", self.name, "\n".join(lines))
