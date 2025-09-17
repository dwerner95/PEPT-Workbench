"""Diagnostics panel with helper plots."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None  # type: ignore[assignment]

from .plotly_widget import PlotlyWidget


class DiagnosticsView(PlotlyWidget):
    """Displays auxiliary diagnostics such as error histograms."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setToolTip("Diagnostics and derived statistics from the pipeline output")

    def show_message(self, message: str) -> None:
        self.set_html(f"<p>{message}</p>")

    def set_error_histogram(self, errors: np.ndarray) -> None:
        if errors.size == 0:
            self.set_html("<h3>No error metrics.</h3>")
            return
        if go is None:
            self.set_html("<p>Plotly is required to display diagnostics.</p>")
            return
        figure = go.Figure(data=[go.Histogram(x=errors, nbinsx=40)])
        figure.update_layout(margin=dict(l=30, r=10, t=30, b=30), title="Error Histogram", bargap=0.05)
        self.set_figure(figure)
