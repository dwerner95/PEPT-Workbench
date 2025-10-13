"""Main window wiring for the PEPT GUI."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Optional, Sequence
from collections.abc import Iterable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from .io import (
    DatasetDescriptor,
    LoadedDataset,
    TimeMask,
    WorkspaceState,
    detect_sources,
    load_dataset,
    prepare_samples_argument,
    save_workspace,
)
from .model import (
    PipelineGraph,
    RunController,
    RunResult,
    available_transformers,
    compile_pipeline,
    registered_transformer_names,
    resolve_transformer,
)
from .model.sample_analysis import (
    SampleWindow,
    annotate_points_with_samples,
    build_sample_windows,
)
from .view import (
    ControlsPanel,
    DiagnosticsView,
    LorsView,
    NodeParameterEditor,
    PipelineGraphView,
    Trajectory2DView,
    TrajectoryView,
)

try:  # pragma: no cover - optional dependency
    import pept
except ImportError:  # pragma: no cover - allow running without dependency
    pept = None  # type: ignore[assignment]


class MainWindow(QtWidgets.QMainWindow):
    """Composes the top-level Qt UI widgets."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PEPT GUI")
        self.resize(1500, 950)

        self.pipeline_graph = PipelineGraph()
        self.run_controller = RunController(self)
        self._dataset: LoadedDataset | None = None
        self._descriptor: DatasetDescriptor | None = None
        self._time_mask = TimeMask(0.0, 10.0)
        self._timeline_start = 0.0
        self._timeline_end = 10.0
        self._decimation = 10
        self._sample_size = 0
        self._overlap = 0
        self._executor = "joblib"
        self._preview_limit = 1000
        self._transformer_registry: dict[str, Any] = {}
        self._trajectory_colour = "t"
        self._sample_windows: list[SampleWindow] = []
        self._latest_points: tuple[np.ndarray, list[str], np.ndarray | None] | None = None
        self._playback_speed = 1.0
        self._playback_pending = 0.0
        self._playback_cache: PlaybackCache | None = None
        self._playback_dirty = True
        self._play_timer = QtCore.QTimer(self)
        self._play_timer.setInterval(100)
        self._play_timer.timeout.connect(self._advance_playback)

        self._setup_actions()
        self._setup_toolbars()
        self._setup_status_bar()
        self._setup_central_layout()
        self._connect_controller()

        self.controls.set_time_mask(self._time_mask)
        self.controls.set_sampling(sample_size=self._sample_size, overlap=self._overlap)
        self.controls.set_decimation(self._decimation)
        self.controls.set_executor(self._executor)
        self.controls.set_preview_limit(self._preview_limit)
        self.controls.set_trajectory_colour(self._trajectory_colour)
        self.controls.set_playback_speed(self._playback_speed)
        self.controls.time_mask_changed.connect(self._on_time_mask_changed)
        self.controls.decimation_changed.connect(self._on_decimation_changed)
        self.controls.sample_size_changed.connect(self._on_sample_size_changed)
        self.controls.overlap_changed.connect(self._on_overlap_changed)
        self.controls.executor_changed.connect(self._on_executor_changed)
        self.controls.preview_limit_changed.connect(self._on_preview_limit_changed)
        self.controls.trajectory_colour_changed.connect(self._on_trajectory_colour_changed)
        self.controls.playback_toggled.connect(self._on_playback_toggled)
        self.controls.playback_speed_changed.connect(self._on_playback_speed_changed)

        transformer_registry = available_transformers()
        self._transformer_registry = transformer_registry
        self.graph_view.set_graph(self.pipeline_graph)
        self.graph_view.set_available_nodes(registered_transformer_names())
        self.graph_view.graph_changed.connect(self._on_graph_changed)
        self.graph_view.node_selected.connect(self._on_node_selected)

        self.node_editor.set_registry(transformer_registry)
        self.node_editor.set_transformer_resolver(resolve_transformer)
        self.node_editor.parameter_changed.connect(self._on_node_params_changed)

        self._update_code_preview()
        self._update_sample_summary()

    # ------------------------------------------------------------------ UI setup
    def _setup_actions(self) -> None:
        icon_provider = QtWidgets.QApplication.style()

        self.open_action = QtGui.QAction("Open Dataset…", self)
        self.open_action.setShortcut(QtGui.QKeySequence.Open)
        self.open_action.triggered.connect(self._open_dataset)  # type: ignore[arg-type]

        self.save_workspace_action = QtGui.QAction("Save Workspace…", self)
        self.save_workspace_action.setShortcut(QtGui.QKeySequence.Save)
        self.save_workspace_action.triggered.connect(self._save_workspace)  # type: ignore[arg-type]

        self.export_action = QtGui.QAction("Export Figure…", self)
        self.export_action.setShortcut("Ctrl+E")
        self.export_action.triggered.connect(self._export_figure)  # type: ignore[arg-type]

        self.run_action = QtGui.QAction("Run Pipeline", self)
        self.run_action.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        self.run_action.triggered.connect(self._run_pipeline)  # type: ignore[arg-type]

        self.stop_action = QtGui.QAction("Stop", self)
        self.stop_action.setShortcut(QtGui.QKeySequence("Ctrl+."))
        self.stop_action.triggered.connect(self._stop_pipeline)  # type: ignore[arg-type]

        self.open_action.setIcon(icon_provider.standardIcon(QtWidgets.QStyle.SP_DirOpenIcon))
        self.save_workspace_action.setIcon(icon_provider.standardIcon(QtWidgets.QStyle.SP_DialogSaveButton))
        self.export_action.setIcon(icon_provider.standardIcon(QtWidgets.QStyle.SP_FileIcon))
        self.run_action.setIcon(icon_provider.standardIcon(QtWidgets.QStyle.SP_MediaPlay))
        self.stop_action.setIcon(icon_provider.standardIcon(QtWidgets.QStyle.SP_BrowserStop))

    def _setup_toolbars(self) -> None:
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.addAction(self.open_action)
        toolbar.addAction(self.save_workspace_action)
        toolbar.addSeparator()
        toolbar.addAction(self.run_action)
        toolbar.addAction(self.stop_action)
        toolbar.addSeparator()
        toolbar.addAction(self.export_action)

    def _setup_status_bar(self) -> None:
        status = QtWidgets.QStatusBar(self)
        self.setStatusBar(status)
        self.progress_label = QtWidgets.QLabel("Ready")
        self.executor_label = QtWidgets.QLabel(f"Executor: {self._executor}")
        self.elapsed_label = QtWidgets.QLabel("Elapsed: 0.0 s")

        status.addPermanentWidget(self.executor_label)
        status.addPermanentWidget(self.elapsed_label)
        status.addWidget(self.progress_label, 1)

    def _setup_central_layout(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.controls = ControlsPanel(central)
        self.controls.setMinimumWidth(280)
        layout.addWidget(self.controls)
        self.controls.set_time_range(self._timeline_start, self._timeline_end)

        centre_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, central)
        layout.addWidget(centre_splitter, 1)

        self.tabs = QtWidgets.QTabWidget(centre_splitter)
        self.lors_view = LorsView(self.tabs)
        self.tabs.addTab(self.lors_view, "LoRs 3D")
        self.traj_view = TrajectoryView(self.tabs)
        self.tabs.addTab(self.traj_view, "Trajectories 3D")
        self.traj2d_view = Trajectory2DView(self.tabs)
        self.tabs.addTab(self.traj2d_view, "Trajectories 2D")
        self.diagnostics_view = DiagnosticsView(self.tabs)
        self.tabs.addTab(self.diagnostics_view, "Diagnostics")

        self.right_panel = QtWidgets.QWidget(central)
        layout.addWidget(self.right_panel)
        right_layout = QtWidgets.QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self.graph_view = PipelineGraphView(self.right_panel)
        right_layout.addWidget(self.graph_view, 3)

        self.node_editor = NodeParameterEditor(self.right_panel)
        right_layout.addWidget(self.node_editor, 2)

        self.code_preview = QtWidgets.QPlainTextEdit(self.right_panel)
        self.code_preview.setReadOnly(True)
        self.code_preview.setPlaceholderText("Pipeline code preview")
        right_layout.addWidget(self.code_preview, 2)

    def _connect_controller(self) -> None:
        self.run_controller.started.connect(lambda: self.progress_label.setText("Running pipeline…"))
        self.run_controller.error.connect(self._on_run_error)
        self.run_controller.result_ready.connect(self._on_run_result)
        self.run_controller.finished.connect(self._on_run_finished)

    # -------------------------------------------------------------- action slots
    @QtCore.Slot()
    def _open_dataset(self) -> None:  # pragma: no cover - Qt slot
        dialog = QtWidgets.QFileDialog(self)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        paths = [Path(p) for p in dialog.selectedFiles()]
        try:
            descriptor = detect_sources(paths)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Failed to detect dataset", str(exc))
            return
        self._descriptor = descriptor
        self.controls.set_sources(str(p) for p in descriptor.paths)
        self._load_dataset(descriptor)

    def _load_dataset(self, descriptor: DatasetDescriptor) -> None:
        try:
            dataset = load_dataset(
                descriptor,
                time_mask=self._time_mask,
                decimate_every=self._decimation,
                sample_size=int(self._sample_size) if self._sample_size > 0 else None,
                overlap=int(self._overlap) if self._overlap > 0 else None,
            )
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Failed to load dataset", "See console for details.")
            return
        self._dataset = dataset
        self._stop_playback()
        self._refresh_lo_rs()
        self.progress_label.setText(f"Loaded {descriptor.scanner} dataset")

        # Update default time mask to full dataset if possible
        if dataset.raw_lines.size:
            start = float(dataset.raw_lines[0, 0])
            end = float(dataset.raw_lines[-1, 0])
            if end <= start:
                end = start + 1.0
            self._timeline_start = start
            self._timeline_end = end
            self.controls.set_time_range(self._timeline_start, self._timeline_end)
            self._time_mask = TimeMask(start, min(start + 10.0, end))
            self.controls.set_time_mask(self._time_mask)
            self._apply_time_mask()

    def _save_workspace(self) -> None:  # pragma: no cover - Qt slot
        if self._descriptor is None or self._dataset is None:
            QtWidgets.QMessageBox.warning(self, "No workspace", "Load a dataset before saving a workspace.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Workspace", filter="Workspace (*.json)")
        if not path:
            return
        state = WorkspaceState(
            sources=[str(p) for p in self._descriptor.paths],
            time_mask=self._time_mask,
            pipeline=self.pipeline_graph.as_pipeline_definition(),
            sample_size=self._sample_size,
            overlap=self._overlap,
            view_state={"decimation": self._decimation, "preview_limit": self._preview_limit},
            preview_limit=self._preview_limit,
            executor=self._executor,
        )
        save_workspace(path, state)
        self.statusBar().showMessage(f"Workspace saved to {path}", 5000)

    def _export_figure(self) -> None:  # pragma: no cover - Qt slot
        widget = self._current_plot_widget()
        if widget is None:
            QtWidgets.QMessageBox.information(self, "Nothing to export", "No plot is currently selected.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Figure", filter="HTML (*.html)")
        if not path:
            return
        html = widget.last_html()
        if not html:
            QtWidgets.QMessageBox.warning(self, "No figure", "The current view has no rendered figure.")
            return
        Path(path).write_text(html, encoding="utf-8")
        self.statusBar().showMessage(f"Exported figure to {path}", 5000)

    def _run_pipeline(self) -> None:  # pragma: no cover - Qt slot
        if self._dataset is None:
            QtWidgets.QMessageBox.warning(self, "No dataset", "Please load a dataset first.")
            return
        try:
            definition = self.pipeline_graph.as_pipeline_definition()
            pipeline = compile_pipeline(definition)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Failed to compile pipeline", str(exc))
            return

        samples = self._ensure_line_data()
        metadata = {"time_mask": (self._time_mask.start, self._time_mask.end)}
        try:
            self.run_controller.run(pipeline, samples, executor=self._executor, metadata=metadata)
            self.run_action.setEnabled(False)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Failed to run pipeline", str(exc))

    def _stop_pipeline(self) -> None:  # pragma: no cover - Qt slot
        self.run_controller.cancel()
        self.progress_label.setText("Cancelled")
        self.run_action.setEnabled(True)

    # -------------------------------------------------------------- controller
    def _on_run_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Pipeline error", message)
        self.run_action.setEnabled(True)

    def _on_run_result(self, result: RunResult) -> None:
        self.run_action.setEnabled(True)
        points, columns = _to_array(result.points)
        customdata: np.ndarray | None = None
        if points.size and points.ndim == 2 and points.shape[1] >= 4 and self._sample_windows:
            customdata = self._annotate_points(points)
        self._latest_points = (points, columns, customdata)
        selected = self.controls.update_trajectory_colour_options(columns, preferred=self._trajectory_colour)
        self._trajectory_colour = selected
        self._render_points()
        speeds = _compute_speeds(points)
        if speeds.size:
            self.diagnostics_view.set_error_histogram(speeds)
        else:
            self.diagnostics_view.show_message("Not enough trajectory samples for diagnostics.")

    def _on_run_finished(self, elapsed: float) -> None:
        self.progress_label.setText("Finished")
        self.elapsed_label.setText(f"Elapsed: {elapsed:.2f} s")

    # -------------------------------------------------------------- state mgmt
    def _on_time_mask_changed(self, mask: TimeMask) -> None:
        self._time_mask = mask
        self._playback_pending = 0.0
        self._apply_time_mask()

    def _on_decimation_changed(self, value: int) -> None:
        self._decimation = value
        self._refresh_lo_rs()

    def _on_sample_size_changed(self, value: int) -> None:
        self._sample_size = int(value)
        self._rebuild_line_data()

    def _on_overlap_changed(self, value: int) -> None:
        self._overlap = int(value)
        self._rebuild_line_data()

    def _on_executor_changed(self, executor: str) -> None:
        self._executor = executor
        self.executor_label.setText(f"Executor: {executor}")

    def _on_graph_changed(self, _: object) -> None:
        self._update_code_preview()

    def _on_node_selected(self, node_id: str) -> None:
        if not node_id:
            self.node_editor.clear()
            self.statusBar().showMessage("No node selected", 3000)
            return
        self.statusBar().showMessage(f"Selected node {node_id}", 3000)
        node = next((n for n in self.pipeline_graph.nodes() if n.id == node_id), None)
        if node is None:
            self.node_editor.clear()
        else:
            self.node_editor.load_node(node)

    def _on_node_params_changed(self, node_id: str, params: dict[str, Any]) -> None:
        try:
            self.pipeline_graph.update_params(node_id, params)
        except KeyError:
            return
        self._update_code_preview()
        node = next((n for n in self.pipeline_graph.nodes() if n.id == node_id), None)
        if node is not None:
            self.node_editor.load_node(node)
        self.statusBar().showMessage(f"Parameters updated for {node_id}", 3000)

    def _apply_time_mask(self) -> None:
        if self._dataset is None:
            self._sample_windows = []
            self._update_sample_summary()
            return
        self._dataset.masked_lines = self._time_mask.apply(self._dataset.raw_lines)
        self._refresh_lo_rs()
        self._rebuild_line_data()

    def _refresh_lo_rs(self) -> None:
        if self._dataset is None:
            return
        lines = self._dataset.masked_lines
        if self._decimation > 1:
            lines = lines[:: self._decimation]
        if self._preview_limit > 0:
            lines = lines[: self._preview_limit]
        self._dataset.preview_lines = lines
        self.lors_view.set_lines(self._dataset.preview_lines, decimated=self._decimation > 1)
        count = self._dataset.masked_lines.shape[0]
        self.progress_label.setText(
            f"{count} LoRs in {self._time_mask.start:.3f}–{self._time_mask.end:.3f} s"
        )

    def _rebuild_line_data(self) -> None:
        if self._dataset is None:
            return

        lines = self._dataset.masked_lines
        if lines.size:
            timestamps = np.asarray(lines[:, 0], dtype=float)
        else:
            timestamps = np.array([], dtype=float)
        self._sample_windows = build_sample_windows(timestamps, self._sample_size, self._overlap)
        self._update_sample_summary()

        if pept is None or self._play_timer.isActive():
            self._dataset.line_data = None
            return
        kwargs: dict[str, Any] = {}
        sample_value = prepare_samples_argument(self._sample_size)
        overlap_value = prepare_samples_argument(self._overlap)
        if sample_value is not None:
            kwargs["sample_size"] = sample_value
        if overlap_value is not None:
            kwargs["overlap"] = overlap_value
        try:
            self._dataset.line_data = pept.LineData(self._dataset.masked_lines, **kwargs)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Failed to rebuild line data: {exc}")
            self._dataset.line_data = None

        if self._latest_points is not None:
            points, columns, _ = self._latest_points
            customdata = None
            if points.size and self._sample_windows:
                customdata = self._annotate_points(points)
            self._latest_points = (points, columns, customdata)
            self._render_points()

    def _ensure_line_data(self) -> Any:
        if self._dataset is None:
            raise RuntimeError("Dataset not loaded")
        if self._dataset.line_data is not None:
            return self._dataset.line_data
        if pept is None:
            raise RuntimeError("pept library is required to run the pipeline")
        self._rebuild_line_data()
        if self._dataset.line_data is None:
            raise RuntimeError("Could not create LineData from masked lines")
        return self._dataset.line_data

    def _on_preview_limit_changed(self, value: int) -> None:
        self._preview_limit = int(value)
        self._refresh_lo_rs()

    def _on_trajectory_colour_changed(self, value: str) -> None:
        self._trajectory_colour = value
        self._render_points()

    def _on_playback_toggled(self, active: bool) -> None:
        if active:
            if self._dataset is None or self._dataset.raw_lines.size == 0:
                self.controls.set_playback_state(False)
                return
            if not self._playback_can_move():
                self.statusBar().showMessage(
                    "Playback requires a time window shorter than the dataset span.",
                    5000,
                )
                self.controls.set_playback_state(False)
                return
            self._playback_pending = 0.0
            if not self._play_timer.isActive():
                self._play_timer.start()
                self.statusBar().showMessage("Playback running", 1000)
        else:
            self._stop_playback()

    def _on_playback_speed_changed(self, value: float) -> None:
        self._playback_speed = max(0.01, float(value))
        self.statusBar().showMessage(f"Playback speed set to {self._playback_speed:.2f}×", 2000)

    def _advance_playback(self) -> None:
        if self._dataset is None or self._dataset.raw_lines.size == 0:
            self._stop_playback()
            return
        if not self._playback_can_move():
            self._stop_playback()
            return
        interval_seconds = max(self._play_timer.interval(), 1) / 1000.0
        delta = self._playback_speed * interval_seconds
        if delta <= 0.0:
            return
        self._playback_pending += delta
        if self._mask_at_end():
            self._playback_pending = 0.0
            self._stop_playback()
            return
        moved = self.controls.shift_time_mask(self._playback_pending)
        if moved:
            self._playback_pending = 0.0
        elif self._mask_at_end():
            self._playback_pending = 0.0
            self._stop_playback()

    def _stop_playback(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
        self.controls.set_playback_state(False)
        self._playback_pending = 0.0
        if self._dataset is not None and self._dataset.line_data is None and pept is not None:
            self._rebuild_line_data()

    def _mask_at_end(self) -> bool:
        if self._dataset is None or self._dataset.raw_lines.size == 0:
            return True
        max_start = max(self._timeline_start, self._timeline_end - self._time_mask.duration)
        return self._time_mask.start >= max_start - 1e-9

    def _playback_can_move(self) -> bool:
        total_span = self._timeline_end - self._timeline_start
        return total_span > self._time_mask.duration + 1e-9

    def _render_points(self) -> None:
        if self._latest_points is None:
            return
        points, columns, customdata = self._latest_points
        if points.size == 0:
            self.traj_view.set_points(points)
            self.traj2d_view.set_points(points, columns, None)
            return
        colour_array, colour_label, resolved_name = _select_colour(points, columns, self._trajectory_colour)
        self.traj_view.set_points(points, colour=colour_array, colour_label=colour_label, customdata=customdata)
        self.traj2d_view.set_points(points, columns, resolved_name)

    def _current_plot_widget(self) -> Optional[object]:
        widget = self.tabs.currentWidget()
        if isinstance(widget, (LorsView, TrajectoryView, Trajectory2DView, DiagnosticsView)):
            return widget
        return None

    def _update_code_preview(self) -> None:
        try:
            nodes = self.pipeline_graph.ordered_nodes()
        except ValueError:
            self.code_preview.setPlainText("Pipeline has cycles or is invalid.")
            return
        registry = self._transformer_registry or available_transformers()
        lines = ["import pept", "", "pipeline = pept.Pipeline(["]
        for node in nodes:
            transformer = registry.get(node.type)
            module = transformer.__module__ if transformer else "pept"
            params = ", ".join(f"{key}={value!r}" for key, value in (node.params or {}).items())
            call = f"{module}.{node.type}({params})" if params else f"{module}.{node.type}()"
            lines.append(f"    {call},")
        lines.append("])")
        self.code_preview.setPlainText("\n".join(lines))

    def _update_sample_summary(self) -> None:
        summary = [
            {
                "index": window.index,
                "start": window.start_time,
                "end": window.end_time,
                "span": window.span,
                "count": window.count,
                "distance": window.distance,
            }
            for window in self._sample_windows
        ]
        self.diagnostics_view.set_sample_summary(summary)

    def _annotate_points(self, points: np.ndarray) -> np.ndarray:
        customdata, distances = annotate_points_with_samples(points, self._sample_windows)
        if distances:
            for window in self._sample_windows:
                window.distance = distances.get(window.index, float("nan"))
            self._update_sample_summary()
        return customdata


# ---------------------------------------------------------------------------
# Helpers


# ---------------------------------------------------------------------------
# Helpers

def _to_array(data: Any) -> tuple[np.ndarray, list[str]]:
    """Coerce PEPT point collections into a numeric matrix with column names."""

    collected: list[np.ndarray] = []
    column_candidates: list[list[str]] = []

    def _normalize(arr: Any, columns: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
        arr = np.asarray(arr)
        names = list(columns) if columns else []
        if arr.size == 0:
            return np.empty((0, 0)), names

        if arr.dtype.names:
            names = list(arr.dtype.names)
            arr = np.column_stack([np.asarray(arr[name], dtype=float) for name in names])

        if arr.dtype == object:
            rows: list[np.ndarray] = []
            names_found: list[str] = []
            for item in arr.flat:
                sub_arr, sub_names = _normalize(np.asarray(item))
                if sub_arr.size:
                    rows.append(sub_arr)
                    if sub_names:
                        names_found = sub_names
            if not rows:
                return np.empty((0, 0)), names
            arr = np.vstack(rows)
            if names_found:
                names = names_found

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        arr = arr.astype(float, copy=False)

        if not names or len(names) < arr.shape[1]:
            names = names[: arr.shape[1]]
            while len(names) < arr.shape[1]:
                names.append(f"col_{len(names)}")
        elif len(names) > arr.shape[1]:
            names = names[: arr.shape[1]]

        return arr, names

    def _from_point_data(obj: Any) -> tuple[np.ndarray, list[str]] | None:
        if pept is None:
            return None
        point_cls = getattr(pept, "PointData", None)
        if point_cls is not None and isinstance(obj, point_cls):
            columns_attr = getattr(obj, "columns", None)
            columns = list(columns_attr) if columns_attr is not None else None
            for attr_name in ("points", "array", "as_array", "to_numpy"):
                attr = getattr(obj, attr_name, None)
                if attr is None:
                    continue
                try:
                    value = attr() if callable(attr) else attr
                except Exception:  # pragma: no cover
                    continue
                arr, names = _normalize(value, columns)
                if not names and columns:
                    names = list(columns)
                return arr, names
        return None

    def _collect(obj: Any) -> None:
        result = _from_point_data(obj)
        if result is not None:
            arr, names = result
            if arr.size:
                collected.append(arr)
                column_candidates.append(names)
            return

        if isinstance(obj, np.ndarray):
            arr, names = _normalize(obj)
            if arr.size:
                collected.append(arr)
                column_candidates.append(names)
            return

        for attr_name in ("points", "array", "as_array", "to_numpy"):
            attr = getattr(obj, attr_name, None)
            if attr is None:
                continue
            try:
                value = attr() if callable(attr) else attr
            except Exception:
                continue
            arr, names = _normalize(value)
            if arr.size:
                collected.append(arr)
                column_candidates.append(names)
            return

        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            for item in obj:
                _collect(item)
            return

        arr, names = _normalize(np.asarray(obj))
        if arr.size:
            collected.append(arr)
            column_candidates.append(names)

    _collect(data)

    if not collected:
        return np.empty((0, 0)), []

    widths = [arr.shape[1] for arr in collected if arr.size]
    if not widths:
        return np.empty((0, 0)), []
    min_width = min(widths)
    arrays = [arr[:, :min_width] for arr in collected]
    points = np.concatenate(arrays, axis=0)

    columns: list[str] = []
    for names in column_candidates:
        if len(names) >= min_width:
            columns = names[:min_width]
            break
    if not columns:
        columns = [f"col_{i}" for i in range(min_width)]

    return points, columns


def _compute_speeds(points: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 4:
        return np.asarray([])
    dt = np.diff(points[:, 0])
    displacements = np.linalg.norm(np.diff(points[:, 1:4], axis=0), axis=1)
    dt[dt == 0] = 1e-9
    return displacements / dt


def _select_colour(points: np.ndarray, columns: list[str], selection: str) -> tuple[np.ndarray, str, str]:
    if points.ndim != 2 or points.shape[0] == 0:
        return np.array([]), "Time [ms]", columns[0] if columns else ""
    index, name = _resolve_colour_selection(columns, selection)
    if index >= points.shape[1]:
        index = min(points.shape[1] - 1, 0)
        name = columns[index] if index < len(columns) else f"col_{index}"
    colour = points[:, index]
    label = _format_colour_label(name)
    return colour, label, name


def _resolve_colour_selection(columns: list[str], selection: str) -> tuple[int, str]:
    if not columns:
        return 0, "col_0"
    selection = (selection or "").strip()
    lookup = {name.lower(): idx for idx, name in enumerate(columns)}
    lower = selection.lower()
    if lower in lookup:
        idx = lookup[lower]
        return idx, columns[idx]
    alias = {"time": 0, "t": 0}
    idx = alias.get(lower)
    if idx is not None and idx < len(columns):
        return idx, columns[idx]
    return 0, columns[0]


def _format_colour_label(name: str) -> str:
    mapping = {
        "t": "Time [ms]",
        "time": "Time [ms]",
        "x": "X [mm]",
        "y": "Y [mm]",
        "z": "Z [mm]",
        "error": "Error [mm]",
    }
    return mapping.get(name.lower(), name.capitalize())
