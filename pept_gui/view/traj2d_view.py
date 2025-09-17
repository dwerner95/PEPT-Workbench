"""2D trajectory projection view."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:  # pragma: no cover
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
except ImportError:  # pragma: no cover
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]

from .plotly_widget import PlotlyWidget


class Trajectory2DView(PlotlyWidget):
    """Displays trajectory positions as time series plots."""

    def __init__(self, parent: Optional[object] = None) -> None:
        super().__init__(parent)
        self.setToolTip("Time series of X/Y/Z positions with optional highlight column")

    def set_points(
        self,
        points: np.ndarray,
        columns: Sequence[str],
        highlight: Optional[str],
    ) -> None:
        if points.size == 0 or points.ndim != 2:
            self.set_html("<h3>No trajectory points available.</h3>")
            return
        if go is None or make_subplots is None:
            self.set_html("<p>Plotly is required to preview trajectories.</p>")
            return
        figure = _build_timeseries(points, list(columns), highlight)
        self.set_figure(figure)


def _build_timeseries(points: np.ndarray, columns: list[str], highlight: Optional[str]) -> "go.Figure":
    width = points.shape[1]
    if not columns:
        columns = [f"col_{i}" for i in range(width)]

    time_idx, _ = _resolve_column(columns, "t", fallback=0, width=width)
    if time_idx is None or time_idx >= width:
        time_idx = 0
    time_values = points[:, time_idx]

    axes: list[tuple[str, int]] = []
    for label, target, fallback in [
        ("X", "x", 1),
        ("Y", "y", 2),
        ("Z", "z", 3),
    ]:
        idx, resolved = _resolve_column(columns, target, fallback=fallback, width=width)
        if idx is not None and idx < width:
            axes.append((_format_label(resolved), idx))

    if not axes:
        for idx in range(min(width, 3)):
            axes.append((columns[idx], idx))

    highlight_idx = None
    highlight_label = None
    if highlight:
        highlight_idx, resolved = _resolve_column(columns, highlight, fallback=None, width=width)
        if highlight_idx is not None and highlight_idx < width:
            axis_indices = {idx for _, idx in axes}
            if highlight_idx == time_idx or highlight_idx in axis_indices:
                highlight_idx = None
            else:
                highlight_label = _format_label(resolved)
        else:
            highlight_idx = None

    rows = len(axes)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[label for label, _ in axes],
    )

    current_row = 1
    highlight_values = points[:, highlight_idx] if highlight_idx is not None else None
    for label, idx in axes:
        marker = dict(size=4)
        if highlight_values is not None:
            marker.update(
                color=highlight_values,
                colorscale="Viridis",
                showscale=current_row == 1,
                colorbar=dict(title=highlight_label or ""),
            )
        fig.add_trace(
            go.Scatter(
                x=time_values,
                y=points[:, idx],
                mode="markers+lines",
                marker=marker,
                name=label,
            ),
            row=current_row,
            col=1,
        )
        fig.update_yaxes(title_text=label, row=current_row, col=1)
        current_row += 1

    fig.update_xaxes(title_text="Time [ms]", row=rows, col=1)
    fig.update_layout(
        height=260 * rows,
        showlegend=False,
        margin=dict(l=70, r=30, t=60, b=50),
    )
    return fig


def _resolve_column(
    columns: list[str],
    selection: str,
    *,
    fallback: Optional[int],
    width: int,
) -> tuple[Optional[int], str]:
    lookup = {name.lower(): idx for idx, name in enumerate(columns)}
    lower = selection.lower()
    if lower in lookup:
        idx = lookup[lower]
        return (idx, columns[idx]) if idx < width else (None, selection)
    alias = {"time": 0, "t": 0}
    idx = alias.get(lower)
    if idx is not None and idx < len(columns) and idx < width:
        return idx, columns[idx]
    if fallback is not None and fallback < width:
        return fallback, columns[fallback] if fallback < len(columns) else f"col_{fallback}"
    return None, selection


def _format_label(name: str) -> str:
    mapping = {
        "t": "Time [ms]",
        "time": "Time [ms]",
        "x": "X [mm]",
        "y": "Y [mm]",
        "z": "Z [mm]",
        "error": "Error [mm]",
    }
    return mapping.get(name.lower(), name.capitalize())
