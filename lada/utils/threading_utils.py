# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import logging
import time
import traceback
from queue import Queue, Full, Empty
import concurrent.futures as concurrent_futures
from threading import Thread
from typing import Callable

from lada import LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

class StopMarker:
    pass

class EofMarker:
    pass

class ErrorMarker(Exception):
    def __init__(self, message, stack_trace: str):
        super().__init__(message)
        self.stack_trace = stack_trace

STOP_MARKER = StopMarker()
EOF_MARKER = EofMarker()

class PipelineThread(Thread):
    def __init__(self, name: str, target: Callable[[],None], error_handler: Callable[[ErrorMarker],None]):
        super().__init__(target=target, daemon=True)
        self.error_handler = error_handler
        self.name = name

    def run(self):
        try:
            super().run()
        except Exception as e:
            logger.error(f"{self.name}: Crashed", exc_info=True)
            self.error_handler(ErrorMarker(message=str(e), stack_trace=traceback.format_exc()))

class PipelineQueue(Queue):
    def __init__(self, name: str, maxsize=0):
        logger.debug(f"Set queue size of queue {name} to {maxsize}{' (unlimited)' if maxsize == 0 else ''}")
        super().__init__(maxsize=maxsize)
        self.stats = {}
        self.name = name
        self.stats[f"{name}_wait_time_put"] = 0
        self.stats[f"{name}_wait_time_get"] = 0
        self.stats[f"{name}_max_size"] = 0

    def put(self, item, block=True, timeout=None):
        self.stats[f"{self.name}_max_size"] = max(self.qsize() + 1, self.stats[f"{self.name}_max_size"])
        s = time.time()
        super().put(item, block=block, timeout=timeout)
        self.stats[f"{self.name}_wait_time_put"] += time.time() - s

    def get(self, block=True, timeout=None):
        s = time.time()
        item = super().get(block=block, timeout=timeout)
        self.stats[f"{self.name}_wait_time_get"] += time.time() - s
        return item

def put_queue_stop_marker(queue: Queue | PipelineQueue, debug_queue_name: str | None = None, stop_marker=STOP_MARKER):
    queue_name = queue.name if isinstance(queue, PipelineQueue) else debug_queue_name
    assert queue_name is not None
    sent_out_none_success = False
    while not sent_out_none_success:
        try:
            queue.put(stop_marker, block=False)
            sent_out_none_success = True
        except Full:
            queue.get(block=False)
            queue.task_done()
    logger.debug(f"sent out None to queue {queue_name} to indicate we're stopping")


def empty_out_queue(queue: Queue | PipelineQueue, debug_queue_name: str | None = None):
    queue_name = queue.name if isinstance(queue, PipelineQueue) else debug_queue_name
    assert queue_name is not None
    while not queue.empty():
        queue.get()
        queue.task_done()
    logger.debug(f"purged all remaining elements from queue {queue_name}")

def empty_out_queue_until_producer_is_done(queue: PipelineQueue, producer_thread: Thread):
    """
    Use it only if producer we're waiting for can produce an unknown number of items before it stops and could therefore
    potentially block on put() if queue size is limited.
    """
    def consumer():
        while (producer_thread and producer_thread.is_alive()) or not queue.empty():
            try:
                queue.get(timeout=0.02)
                queue.task_done()
            except Empty:
                pass
        logger.debug(f"purged all remaining elements from queue {queue.name}")
    consumer_thread = Thread(target=consumer)
    consumer_thread.start()
    return consumer_thread

def empty_out_queue_until_futures_are_done(queue: Queue, debug_queue_name: str, futures: list[concurrent_futures.Future]):
    def consumer():
        while any([future.running() for future in futures]):
            try:
                queue.get(timeout=0.02)
                queue.task_done()
            except Empty:
                pass
        logger.debug(f"purged all remaining elements from queue {debug_queue_name}")
    consumer_thread = Thread(target=consumer)
    consumer_thread.start()
    return consumer_thread

def check_for_errors(futures: list[concurrent_futures.Future]):
    for job in concurrent_futures.as_completed(futures):
        exception = job.exception()
        if exception:
            print(f"future job failed with: {type(exception).__name__}: {exception}")
            raise exception

def wait_until_completed(futures: list[concurrent_futures.Future]):
    concurrent_futures.wait(futures, return_when=concurrent_futures.ALL_COMPLETED)
    check_for_errors(futures)

def clean_up_completed_futures(completed_futures):
    check_for_errors(completed_futures)
    for job in concurrent_futures.as_completed(completed_futures):
        completed_futures.remove(job)