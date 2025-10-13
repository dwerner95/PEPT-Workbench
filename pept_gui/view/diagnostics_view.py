"""Diagnostics panel with helper plots."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from PySide6 import QtWidgets

try:  # pragma: no cover
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None  # type: ignore[assignment]

from .plotly_widget import PlotlyWidget


class DiagnosticsView(QtWidgets.QWidget):
    """Displays sample analytics and auxiliary diagnostics."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setToolTip("Diagnostics and derived statistics from the pipeline output")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._summary_browser = QtWidgets.QTextBrowser(self)
        self._summary_browser.setOpenExternalLinks(False)
        self._summary_browser.setReadOnly(True)
        self._summary_browser.setPlaceholderText("Sample analytics will appear here once data is loaded.")
        layout.addWidget(self._summary_browser, 2)

        self._plot_widget = PlotlyWidget(self)
        layout.addWidget(self._plot_widget, 3)

    def show_message(self, message: str) -> None:
        self._plot_widget.set_html(f"<p>{message}</p>")

    def set_error_histogram(self, errors: np.ndarray) -> None:
        if errors.size == 0:
            self._plot_widget.set_html("<h3>No error metrics.</h3>")
            return
        if go is None:
            self._plot_widget.set_html("<p>Plotly is required to display diagnostics.</p>")
            return
        figure = go.Figure(data=[go.Histogram(x=errors, nbinsx=40)])
        figure.update_layout(margin=dict(l=30, r=10, t=30, b=30), title="Error Histogram", bargap=0.05)
        self._plot_widget.set_figure(figure)

    def last_html(self) -> str:
        return self._plot_widget.last_html()

    def set_sample_summary(self, samples: Sequence[Mapping[str, float]]) -> None:
        if not samples:
            self._summary_browser.setHtml("<p>No samples configured. The full window will be processed.</p>")
            return
        total = len(samples)
        spans = [float(entry.get("span", 0.0)) for entry in samples]
        counts = [float(entry.get("count", 0.0)) for entry in samples]
        distances = [float(entry.get("distance", float("nan"))) for entry in samples]
        avg_span = sum(spans) / total if total else 0.0
        avg_count = sum(counts) / total if total else 0.0
        valid_distances = [d for d in distances if math.isfinite(d)]
        avg_distance = (sum(valid_distances) / len(valid_distances)) if valid_distances else float("nan")
        avg_distance_label = f"{avg_distance:.3f} mm" if math.isfinite(avg_distance) else "—"

        rows = []
        limit = min(10, total)
        header = (
            "<table style='width:100%; border-collapse:collapse;'>"
            "<tr><th align='left'>#</th><th align='right'>Span [s]</th>"
            "<th align='right'>LoRs</th><th align='right'>Distance [mm]</th></tr>"
        )
        rows.append(header)
        for entry in samples[:limit]:
            index = int(entry.get("index", 0))
            span = float(entry.get("span", 0.0))
            count = float(entry.get("count", 0.0))
            distance = float(entry.get("distance", float("nan")))
            if math.isfinite(distance):
                distance_formatted = f"{distance:.3f}"
            else:
                distance_formatted = "—"
            rows.append(
                "<tr>"
                f"<td>{index}</td>"
                f"<td align='right'>{span:.3f}</td>"
                f"<td align='right'>{count:.0f}</td>"
                f"<td align='right'>{distance_formatted}</td>"
                "</tr>"
            )
        rows.append("</table>")
        if total > limit:
            rows.append(f"<p>Showing first {limit} of {total} samples.</p>")

        summary_html = (
            "<h3>Sample Analytics</h3>"
            f"<p>Total samples: {total} &nbsp;|&nbsp; Avg span: {avg_span:.3f} s &nbsp;|&nbsp; "
            f"Avg LoRs: {avg_count:.1f} &nbsp;|&nbsp; Avg distance: {avg_distance_label}</p>"
            + "\n".join(rows)
        )
        self._summary_browser.setHtml(summary_html)
