"""Threaded pipeline execution controller."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from PySide6 import QtCore


@dataclass
class RunResult:
    """Result emitted after a pipeline run."""

    points: Any
    elapsed: float
    executor: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class _TaskSignals(QtCore.QObject):
    result = QtCore.Signal(object)
    error = QtCore.Signal(str)
    finished = QtCore.Signal(float)


class PipelineTask(QtCore.QRunnable):
    """Background runnable executing a PEPT pipeline."""

    def __init__(self, pipeline: Any, samples: Any, *, executor: str, metadata: Mapping[str, Any]) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.samples = samples
        self.executor = executor
        self.metadata = dict(metadata)
        self.signals = _TaskSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:  # pragma: no cover - requires threading
        start = time.perf_counter()
        try:
            points = self.pipeline.fit(self.samples, executor=self.executor)
        except Exception as exc:  # noqa: BLE001 - surface pipeline failures
            self.signals.error.emit(str(exc))
            self.signals.finished.emit(0.0)
            return
        if self._cancelled:
            self.signals.finished.emit(0.0)
            return
        elapsed = time.perf_counter() - start
        self.signals.result.emit(RunResult(points, elapsed, self.executor, self.metadata))
        self.signals.finished.emit(elapsed)


class RunController(QtCore.QObject):
    """Coordinates pipeline execution in a :class:`QThreadPool`."""

    started = QtCore.Signal()
    finished = QtCore.Signal(float)
    error = QtCore.Signal(str)
    result_ready = QtCore.Signal(object)

    def __init__(self, parent: QtCore.QObject | None = None, *, thread_pool: QtCore.QThreadPool | None = None) -> None:
        super().__init__(parent)
        self._thread_pool = thread_pool or QtCore.QThreadPool.globalInstance()
        self._current_task: PipelineTask | None = None

    def run(
        self,
        pipeline: Any,
        samples: Any,
        *,
        executor: str = "joblib",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self._current_task is not None:
            raise RuntimeError("A pipeline run is already in progress")

        task = PipelineTask(pipeline, samples, executor=executor, metadata=metadata or {})
        task.signals.result.connect(self._handle_result)
        task.signals.error.connect(self.error)
        task.signals.finished.connect(self._handle_finished)

        self._current_task = task
        self.started.emit()
        self._thread_pool.start(task)

    def cancel(self) -> None:
        if self._current_task is None:
            return
        self._current_task.cancel()
        self._current_task = None

    # ----------------------------------------------------------------- handlers
    @QtCore.Slot(object)
    def _handle_result(self, result: RunResult) -> None:  # pragma: no cover - Qt slot
        self.result_ready.emit(result)

    @QtCore.Slot(float)
    def _handle_finished(self, elapsed: float) -> None:  # pragma: no cover - Qt slot
        self.finished.emit(elapsed)
        self._current_task = None
