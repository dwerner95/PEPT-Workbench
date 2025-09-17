"""Sidebar controls for dataset and pipeline configuration."""

from __future__ import annotations

from typing import Iterable

from PySide6 import QtCore, QtWidgets

from ..io import TimeMask


class ControlsPanel(QtWidgets.QWidget):
    """Provides widgets for adjusting windowing and execution settings."""

    time_mask_changed = QtCore.Signal(TimeMask)
    sample_size_changed = QtCore.Signal(int)
    overlap_changed = QtCore.Signal(int)
    executor_changed = QtCore.Signal(str)
    decimation_changed = QtCore.Signal(int)
    trajectory_colour_changed = QtCore.Signal(str)
    preview_limit_changed = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._sources_list = QtWidgets.QListWidget(self)
        self._sources_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        dataset_group = QtWidgets.QGroupBox("Dataset", self)
        dataset_layout = QtWidgets.QVBoxLayout(dataset_group)
        dataset_layout.addWidget(self._sources_list)
        layout.addWidget(dataset_group)

        window_group = QtWidgets.QGroupBox("Time Window", self)
        window_form = QtWidgets.QFormLayout(window_group)
        self._start_spin = QtWidgets.QDoubleSpinBox(self)
        self._start_spin.setDecimals(3)
        self._start_spin.setRange(0.0, 1e9)
        self._start_spin.valueChanged.connect(self._on_time_changed)

        self._end_spin = QtWidgets.QDoubleSpinBox(self)
        self._end_spin.setDecimals(3)
        self._end_spin.setRange(0.001, 1e9)
        self._end_spin.setValue(10.0)
        self._end_spin.valueChanged.connect(self._on_time_changed)
        window_form.addRow("Start [s]", self._start_spin)
        window_form.addRow("End [s]", self._end_spin)
        layout.addWidget(window_group)

        sampling_group = QtWidgets.QGroupBox("Sampling", self)
        sampling_form = QtWidgets.QFormLayout(sampling_group)
        self._sample_spin = QtWidgets.QSpinBox(self)
        self._sample_spin.setRange(0, 1_000_000)
        self._sample_spin.setSuffix(" LoRs")
        self._sample_spin.setToolTip("Number of LoRs per sample (0 keeps raw ordering)")
        self._sample_spin.valueChanged.connect(lambda value: self.sample_size_changed.emit(int(value)))
        sampling_form.addRow("Sample size", self._sample_spin)

        self._overlap_spin = QtWidgets.QSpinBox(self)
        self._overlap_spin.setRange(0, 1_000_000)
        self._overlap_spin.setSuffix(" LoRs")
        self._overlap_spin.setToolTip("Number of LoRs shared between consecutive samples")
        self._overlap_spin.valueChanged.connect(lambda value: self.overlap_changed.emit(int(value)))
        sampling_form.addRow("Overlap", self._overlap_spin)
        layout.addWidget(sampling_group)

        execution_group = QtWidgets.QGroupBox("Execution", self)
        execution_form = QtWidgets.QFormLayout(execution_group)
        self._executor_combo = QtWidgets.QComboBox(self)
        self._executor_combo.addItems(["joblib", "threaded", "sequential"])
        self._executor_combo.currentTextChanged.connect(self.executor_changed)
        execution_form.addRow("Executor", self._executor_combo)

        self._decimation_spin = QtWidgets.QSpinBox(self)
        self._decimation_spin.setRange(1, 1000)
        self._decimation_spin.setValue(10)
        self._decimation_spin.valueChanged.connect(self.decimation_changed)
        execution_form.addRow("Decimation", self._decimation_spin)

        self._preview_limit_spin = QtWidgets.QSpinBox(self)
        self._preview_limit_spin.setRange(0, 500_000)
        self._preview_limit_spin.setSingleStep(100)
        self._preview_limit_spin.setValue(1000)
        self._preview_limit_spin.setToolTip("Maximum LoR samples shown in the preview (0 disables the limit)")
        self._preview_limit_spin.valueChanged.connect(lambda value: self.preview_limit_changed.emit(int(value)))
        execution_form.addRow("LoRs in preview", self._preview_limit_spin)

        self._traj_colour_combo = QtWidgets.QComboBox(self)
        self._traj_colour_combo.addItem("t")
        self._traj_colour_combo.currentTextChanged.connect(self._on_trajectory_colour_changed)
        execution_form.addRow("Trajectory colour", self._traj_colour_combo)

        layout.addWidget(execution_group)
        layout.addStretch(1)

    # ----------------------------------------------------------------- bindings
    def set_sources(self, sources: Iterable[str]) -> None:
        self._sources_list.clear()
        for source in sources:
            self._sources_list.addItem(source)

    def set_time_mask(self, mask: TimeMask) -> None:
        self._start_spin.blockSignals(True)
        self._end_spin.blockSignals(True)
        self._start_spin.setValue(mask.start)
        self._end_spin.setValue(mask.end)
        self._start_spin.blockSignals(False)
        self._end_spin.blockSignals(False)

    def set_sampling(self, *, sample_size: int, overlap: int) -> None:
        self._sample_spin.blockSignals(True)
        self._overlap_spin.blockSignals(True)
        self._sample_spin.setValue(sample_size)
        self._overlap_spin.setValue(overlap)
        self._sample_spin.blockSignals(False)
        self._overlap_spin.blockSignals(False)

    def set_executor(self, executor: str) -> None:
        index = self._executor_combo.findText(executor)
        if index >= 0:
            self._executor_combo.setCurrentIndex(index)

    def set_decimation(self, value: int) -> None:
        self._decimation_spin.setValue(value)

    def set_preview_limit(self, value: int) -> None:
        self._preview_limit_spin.blockSignals(True)
        self._preview_limit_spin.setValue(value)
        self._preview_limit_spin.blockSignals(False)

    def set_trajectory_colour(self, value: str) -> None:
        index = self._traj_colour_combo.findText(value)
        self._traj_colour_combo.blockSignals(True)
        if index >= 0:
            self._traj_colour_combo.setCurrentIndex(index)
        else:
            self._traj_colour_combo.addItem(value)
            self._traj_colour_combo.setCurrentIndex(self._traj_colour_combo.count() - 1)
        self._traj_colour_combo.blockSignals(False)

    def current_trajectory_colour(self) -> str:
        return self._traj_colour_combo.currentText()

    def update_trajectory_colour_options(self, columns: list[str], preferred: str | None = None) -> str:
        cleaned: list[str] = []
        seen: set[str] = set()
        for column in columns or ["t"]:
            label = str(column)
            if label not in seen:
                cleaned.append(label)
                seen.add(label)
        if not cleaned:
            cleaned = ["t"]

        preferred = preferred or self.current_trajectory_colour()
        self._traj_colour_combo.blockSignals(True)
        self._traj_colour_combo.clear()
        for label in cleaned:
            self._traj_colour_combo.addItem(label)

        if preferred in cleaned:
            index = cleaned.index(preferred)
        else:
            index = 0
        self._traj_colour_combo.setCurrentIndex(index)
        self._traj_colour_combo.blockSignals(False)
        return self._traj_colour_combo.currentText()

    def current_time_mask(self) -> TimeMask:
        return TimeMask(self._start_spin.value(), self._end_spin.value())

    # ------------------------------------------------------------------- helpers
    def _on_time_changed(self, _: float) -> None:
        start = self._start_spin.value()
        end = self._end_spin.value()
        if end <= start:
            end = start + 0.001
            self._end_spin.blockSignals(True)
            self._end_spin.setValue(end)
            self._end_spin.blockSignals(False)
        mask = TimeMask(start, end)
        self.time_mask_changed.emit(mask)

    def _on_trajectory_colour_changed(self, text: str) -> None:
        self.trajectory_colour_changed.emit(text)
