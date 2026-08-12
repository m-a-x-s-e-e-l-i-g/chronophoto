from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import TypeVar

Input = TypeVar("Input")
Output = TypeVar("Output")


def export_render_workers(width: int, height: int) -> int:
    """Choose bounded render concurrency without multiplying 4K memory excessively."""

    override = os.environ.get("CHRONOPHOTO_EXPORT_WORKERS")
    if override:
        try:
            return max(1, min(8, int(override)))
        except ValueError:
            pass
    logical_cpus = os.cpu_count() or 2
    cpu_limit = max(1, min(4, logical_cpus // 2))
    pixels = max(1, width * height)
    memory_limit = 2 if pixels >= 8_000_000 else 3 if pixels >= 2_000_000 else 4
    return min(cpu_limit, memory_limit)


def export_io_workers() -> int:
    """Use a few threads for independent PNG compression and disk writes."""

    logical_cpus = os.cpu_count() or 2
    return max(1, min(4, logical_cpus // 2))


def ordered_parallel_map(
    function: Callable[[Input], Output],
    items: Iterable[Input],
    *,
    workers: int,
    pending_per_worker: int = 1,
) -> Iterator[Output]:
    """Yield ordered results while keeping only a bounded number of jobs in memory."""

    if workers <= 1:
        for item in items:
            yield function(item)
        return

    iterator = iter(items)
    maximum_pending = max(workers, workers * max(1, pending_per_worker))
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="chronophoto-export")
    pending: deque[Future[Output]] = deque()
    try:
        for _ in range(maximum_pending):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            yield pending.popleft().result()
            with suppress(StopIteration):
                pending.append(executor.submit(function, next(iterator)))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
