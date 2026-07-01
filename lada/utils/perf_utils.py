# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import json
import logging
import os
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
    def __init__(self, name: str, metadata: dict[str, object] | None = None):
        self.name = name
        self.metadata = metadata or {}
        self.started_at_s = time.time()
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
        logger.log(
            level,
            "PERF_SUMMARY_JSON %s",
            json.dumps(self.to_summary_dict(stats), ensure_ascii=False, sort_keys=True, default=str),
        )

    def to_summary_dict(self, stats: dict[str, StageTiming] | None = None) -> dict[str, object]:
        if stats is None:
            with self._lock:
                stats = dict(self._stats)
        stages = []
        for stage, timing in sorted(stats.items(), key=lambda item: item[1].total_s, reverse=True):
            avg_ms = (timing.total_s / timing.count) * 1000
            stages.append(
                {
                    "stage": stage,
                    "count": timing.count,
                    "total_s": timing.total_s,
                    "avg_ms": avg_ms,
                    "max_s": timing.max_s,
                }
            )
        return {
            "event": "performance_summary",
            "timer": self.name,
            "metadata": self.metadata,
            "pid": os.getpid(),
            "elapsed_s": time.time() - self.started_at_s,
            "stages": stages,
        }
