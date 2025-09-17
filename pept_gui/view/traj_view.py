"""Trajectory visualisation."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    go = None  # type: ignore[assignment]

from .plotly_widget import PlotlyWidget


class TrajectoryView(PlotlyWidget):
    """Displays pipeline output trajectories as scatter plots."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setToolTip("Trajectories produced by the current pipeline")

    def set_points(
        self,
        points: np.ndarray,
        *,
        colour: np.ndarray | None = None,
        colour_label: str | None = None,
    ) -> None:
        if points.size == 0 or points.ndim != 2 or points.shape[1] < 4:
            self.set_html("<h3>No trajectory points available.</h3>")
            return
        if go is None:
            self.set_html("<p>Plotly is required to preview trajectories.</p>")
            return
        figure = _build_points_figure(points, colour, colour_label)
        self.set_figure(figure)


def _build_points_figure(points: np.ndarray, colour: np.ndarray | None, colour_label: str | None) -> "go.Figure":
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError("Trajectory points expected shape (N, >=4)")

    default_colour = points[:, 0]
    if colour is None or colour.shape[0] != points.shape[0]:
        colour = default_colour

    colourbar_title = colour_label or ("Error [mm]" if points.shape[1] >= 5 and np.allclose(colour, points[:, 4]) else "Time [ms]")

    scatter = go.Scatter3d(
        x=points[:, 1],
        y=points[:, 2],
        z=points[:, 3],
        mode="markers",
        marker=dict(
            size=4,
            color=colour,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=colourbar_title),
        ),
    )
    layout = go.Layout(
        scene=dict(xaxis_title="X [mm]", yaxis_title="Y [mm]", zaxis_title="Z [mm]", aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return go.Figure(data=[scatter], layout=layout)
