"""Visual pipeline builder widgets."""

from __future__ import annotations

from typing import Iterable

from PySide6 import QtCore, QtWidgets

from ..model import PipelineGraph


class PipelineGraphView(QtWidgets.QWidget):
    """Minimal graph editor managing a linear pipeline."""

    graph_changed = QtCore.Signal(object)
    node_selected = QtCore.Signal(str)

    DEFAULT_PARAMS = {
        "Condition": {"__args__": ["error < 15"]},
        "Cutpoints": {"max_distance": 0.5},
        "HDBSCAN": {"true_fraction": 0.15},
        "Centroids": {"error": True},
        "Segregate": {"window": 20, "cut_distance": 10},
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._graph = PipelineGraph()

        layout = QtWidgets.QVBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        layout.addWidget(splitter)

        self._palette = QtWidgets.QListWidget(splitter)
        self._palette.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._palette.itemDoubleClicked.connect(self._on_palette_activated)

        right_panel = QtWidgets.QWidget(splitter)
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        self._pipeline_list = QtWidgets.QListWidget(right_panel)
        self._pipeline_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._pipeline_list.itemSelectionChanged.connect(self._emit_selection)
        right_layout.addWidget(self._pipeline_list, 1)

        button_row = QtWidgets.QHBoxLayout()
        self._remove_btn = QtWidgets.QPushButton("Remove", right_panel)
        self._remove_btn.clicked.connect(self._remove_selected)
        button_row.addWidget(self._remove_btn)
        button_row.addStretch(1)
        right_layout.addLayout(button_row)

        splitter.setStretchFactor(1, 1)

    # ----------------------------------------------------------------- palette
    def set_available_nodes(self, names: Iterable[str]) -> None:
        self._palette.clear()
        for name in sorted(names):
            self._palette.addItem(name)

    def set_graph(self, graph: PipelineGraph) -> None:
        self._graph = graph
        self._refresh_pipeline_list()

    def pipeline_definition(self) -> list[dict]:
        return self._graph.as_pipeline_definition()

    # ---------------------------------------------------------------- actions
    def _on_palette_activated(self, item: QtWidgets.QListWidgetItem) -> None:
        node_type = item.text()
        params = self.DEFAULT_PARAMS.get(node_type, None)
        node = self._graph.add_node(node_type, params=dict(params) if params else None)
        ordered = self._graph.ordered_nodes()
        if len(ordered) >= 2:
            prev = ordered[-2]
            self._graph.connect(prev.id, node.id)
        self._refresh_pipeline_list(select_id=node.id)
        self.graph_changed.emit(self._graph.as_pipeline_definition())

    def _remove_selected(self) -> None:
        selected_items = self._pipeline_list.selectedItems()
        if not selected_items:
            return
        item = selected_items[0]
        node_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self._graph.remove_node(str(node_id))
        self._refresh_pipeline_list()
        self.graph_changed.emit(self._graph.as_pipeline_definition())

    def _refresh_pipeline_list(self, select_id: str | None = None) -> None:
        self._pipeline_list.clear()
        selected_item: QtWidgets.QListWidgetItem | None = None
        for node in self._graph.ordered_nodes():
            item = QtWidgets.QListWidgetItem(f"{node.type} ({node.id})")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, node.id)
            self._pipeline_list.addItem(item)
            if select_id and node.id == select_id:
                selected_item = item

        if selected_item is None and self._pipeline_list.count():
            selected_item = self._pipeline_list.item(self._pipeline_list.count() - 1)

        if selected_item is not None:
            self._pipeline_list.setCurrentItem(selected_item)
        elif self._pipeline_list.count() == 0:
            self.node_selected.emit("")

    def _emit_selection(self) -> None:
        selected_items = self._pipeline_list.selectedItems()
        if not selected_items:
            self.node_selected.emit("")
            return
        node_id = selected_items[0].data(QtCore.Qt.ItemDataRole.UserRole)
        if node_id:
            self.node_selected.emit(str(node_id))
