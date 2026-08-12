from __future__ import annotations

from collections.abc import Callable
from threading import Event
from typing import Literal

from PySide6.QtCore import QObject, Signal, Slot

TaskKind = Literal["preview", "export"]


class TaskCancelled(RuntimeError):
    pass


class TaskWorker(QObject):
    finished = Signal(str, object)
    failed = Signal(str, str)
    cancelled = Signal(str)
    progress = Signal(str, int, str)

    def __init__(
        self,
        task: Callable[[Callable[[int, str], None]], object],
        task_kind: TaskKind = "preview",
    ) -> None:
        super().__init__()
        self.task = task
        self.task_kind = task_kind
        self._cancel = Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    @Slot()
    def run(self) -> None:
        def report(value: int, message: str) -> None:
            if self._cancel.is_set():
                raise TaskCancelled
            self.progress.emit(self.task_kind, value, message)

        try:
            result = self.task(report)
            if self._cancel.is_set():
                raise TaskCancelled
        except TaskCancelled:
            self.cancelled.emit(self.task_kind)
            return
        except Exception as exc:  # noqa: BLE001 - translated into a recoverable UI state
            self.failed.emit(self.task_kind, str(exc))
            return
        self.finished.emit(self.task_kind, result)
