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

    playback_toggled = QtCore.Signal(bool)
    playback_speed_changed = QtCore.Signal(float)

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

        self._timeline_start = 0.0
        self._timeline_end = 10.0
        self._slider_resolution = 10_000

        window_group = QtWidgets.QGroupBox("Time Window", self)
        window_layout = QtWidgets.QVBoxLayout(window_group)

        window_form = QtWidgets.QFormLayout()
        self._duration_spin = QtWidgets.QDoubleSpinBox(self)
        self._duration_spin.setDecimals(3)
        self._duration_spin.setMinimum(0.001)
        self._duration_spin.setMaximum(1e9)
        self._duration_spin.setValue(10.0)
        self._duration_spin.valueChanged.connect(self._on_duration_changed)
        window_form.addRow("Duration [s]", self._duration_spin)

        playback_row = QtWidgets.QWidget(self)
        playback_layout = QtWidgets.QHBoxLayout(playback_row)
        playback_layout.setContentsMargins(0, 0, 0, 0)

        self._play_button = QtWidgets.QToolButton(self)
        self._play_button.setText("Play")
        self._play_button.setCheckable(True)
        self._update_play_icon(False)
        self._play_button.toggled.connect(self._on_play_toggled)
        playback_layout.addWidget(self._play_button)

        self._speed_spin = QtWidgets.QDoubleSpinBox(self)
        self._speed_spin.setDecimals(3)
        self._speed_spin.setMinimum(0.01)
        self._speed_spin.setMaximum(1000.0)
        self._speed_spin.setSingleStep(0.1)
        self._speed_spin.setValue(1.0)
        self._speed_spin.setSuffix("×")
        self._speed_spin.setToolTip("Seconds of data advanced per real second")
        self._speed_spin.valueChanged.connect(self._on_speed_changed)
        playback_layout.addWidget(self._speed_spin, 1)

        window_form.addRow("Playback", playback_row)

        window_layout.addLayout(window_form)

        self._start_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self._start_slider.setRange(0, self._slider_resolution)
        self._start_slider.setPageStep(self._slider_resolution // 100 or 1)
        self._start_slider.valueChanged.connect(self._on_slider_changed)
        window_layout.addWidget(self._start_slider)

        self._range_label = QtWidgets.QLabel(self)
        window_layout.addWidget(self._range_label)

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
        self._duration_spin.blockSignals(True)
        self._duration_spin.setValue(max(mask.end - mask.start, 0.001))
        self._duration_spin.blockSignals(False)
        self._set_slider_position(mask.start)
        self._update_range_label(mask.start, mask.end)

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

    def set_playback_state(self, playing: bool) -> None:
        self._play_button.blockSignals(True)
        self._play_button.setChecked(playing)
        self._play_button.blockSignals(False)
        self._update_play_icon(playing)

    def playback_speed(self) -> float:
        return float(self._speed_spin.value())

    def set_playback_speed(self, value: float) -> None:
        clamped = max(self._speed_spin.minimum(), min(self._speed_spin.maximum(), value))
        self._speed_spin.blockSignals(True)
        self._speed_spin.setValue(clamped)
        self._speed_spin.blockSignals(False)
        self.playback_speed_changed.emit(self.playback_speed())

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
        start = self._current_start()
        duration = self._duration_spin.value()
        end = start + duration
        return TimeMask(start, end)

    def shift_time_mask(self, delta: float) -> bool:
        if not delta:
            return False
        current_start = self._current_start()
        max_start = self._max_start_value()
        proposed = max(self._timeline_start, min(max_start, current_start + delta))
        if abs(proposed - current_start) < 1e-9:
            return False
        self._set_slider_position(proposed)
        self._emit_time_mask()
        return True

    def set_time_range(self, start: float, end: float) -> None:
        self._timeline_start = float(max(0.0, start))
        self._timeline_end = float(max(self._timeline_start + 0.001, end))
        available = max(self._timeline_end - self._timeline_start, 0.001)
        self._duration_spin.blockSignals(True)
        self._duration_spin.setMaximum(available)
        if self._duration_spin.value() > available:
            self._duration_spin.setValue(available)
        self._duration_spin.blockSignals(False)
        self._ensure_slider_constraints()
        self._update_range_label(self._current_start(), self._current_start() + self._duration_spin.value())

    # ------------------------------------------------------------------- helpers
    def _on_duration_changed(self, _: float) -> None:
        self._ensure_slider_constraints()
        self._emit_time_mask()

    def _on_slider_changed(self, _: int) -> None:
        self._emit_time_mask()

    def _on_trajectory_colour_changed(self, text: str) -> None:
        self.trajectory_colour_changed.emit(text)

    def _on_play_toggled(self, active: bool) -> None:
        self._update_play_icon(active)
        self.playback_toggled.emit(active)

    def _on_speed_changed(self, value: float) -> None:
        self.playback_speed_changed.emit(float(value))

    def _emit_time_mask(self) -> None:
        start = self._current_start()
        duration = self._duration_spin.value()
        end = start + duration
        self._update_range_label(start, end)
        self.time_mask_changed.emit(TimeMask(start, end))

    def _current_start(self) -> float:
        if self._start_slider.maximum() <= 0:
            return self._timeline_start
        fraction = self._start_slider.value() / self._start_slider.maximum()
        max_start = self._max_start_value()
        if max_start <= self._timeline_start:
            return self._timeline_start
        start = self._timeline_start + fraction * (max_start - self._timeline_start)
        return min(max_start, start)

    def _set_slider_position(self, start: float) -> None:
        max_start = self._max_start_value()
        start = min(max(start, self._timeline_start), max_start)
        if max_start <= self._timeline_start:
            self._start_slider.blockSignals(True)
            self._start_slider.setValue(0)
            self._start_slider.blockSignals(False)
            self._start_slider.setEnabled(False)
            return
        fraction = (start - self._timeline_start) / (max_start - self._timeline_start)
        value = int(round(fraction * self._start_slider.maximum()))
        self._start_slider.blockSignals(True)
        self._start_slider.setValue(value)
        self._start_slider.blockSignals(False)
        self._start_slider.setEnabled(True)

    def _update_range_label(self, start: float, end: float) -> None:
        self._range_label.setText(f"Start {start:.3f} s → End {end:.3f} s")

    def _ensure_slider_constraints(self) -> None:
        max_start = self._max_start_value()
        if max_start <= self._timeline_start:
            self._start_slider.blockSignals(True)
            self._start_slider.setValue(0)
            self._start_slider.blockSignals(False)
            self._start_slider.setEnabled(False)
        else:
            self._start_slider.setEnabled(True)
            current_start = self._current_start()
            if current_start > max_start:
                self._set_slider_position(max_start)

    def _max_start_value(self) -> float:
        duration = self._duration_spin.value()
        return max(self._timeline_start, self._timeline_end - duration)

    def _update_play_icon(self, playing: bool) -> None:
        style = self.style()
        icon_enum = QtWidgets.QStyle.SP_MediaPause if playing else QtWidgets.QStyle.SP_MediaPlay
        self._play_button.setIcon(style.standardIcon(icon_enum))
        self._play_button.setText("Pause" if playing else "Play")
