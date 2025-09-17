"""LoR preview panel."""

from __future__ import annotations

from typing import Iterable

import numpy as np

try:  # pragma: no cover
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None  # type: ignore[assignment]

from .plotly_widget import PlotlyWidget


class LorsView(PlotlyWidget):
    """Displays Line-of-Response previews using Plotly."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setToolTip("Preview of LoRs inside the selected time window")

    def set_lines(self, lines: np.ndarray, *, decimated: bool = False) -> None:
        if lines.size == 0:
            self.set_html("<h3>No LoRs in the selected window.</h3>")
            return
        if go is None:
            self.set_html("<p>Plotly is required to preview LoRs.</p>")
            return
        figure = _build_lines_figure(lines)
        figure.update_layout(title="Decimated LoRs" if decimated else "LoRs Preview")
        self.set_figure(figure)


def _build_lines_figure(lines: np.ndarray) -> "go.Figure":
    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    for row in lines:
        if row.size < 7:
            continue
        x.extend([row[1], row[4], None])
        y.extend([row[2], row[5], None])
        z.extend([row[3], row[6], None])
    scatter = go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=1.5, color="#1f77b4"))
    layout = go.Layout(
        scene=dict(
            xaxis_title="X [mm]", yaxis_title="Y [mm]", zaxis_title="Z [mm]",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    return go.Figure(data=[scatter], layout=layout)
