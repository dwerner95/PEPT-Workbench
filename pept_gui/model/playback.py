"""Precomputed playback frames for smooth animation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PlaybackFrame:
    """Represents a single playback step."""

    start: float
    end: float
    lors: np.ndarray
    traj_subset: np.ndarray
    playhead: np.ndarray | None


class PlaybackCache:
    """Caches sequential windows through LoR and trajectory data."""

    def __init__(
        self,
        *,
        timeline_start: float,
        timeline_end: float,
        window_start: float,
        window_duration: float,
        lines: np.ndarray,
        trajectories: np.ndarray | None,
        decimation: int = 1,
        preview_limit: int = 0,
        step: float | None = None,
        max_frames: int = 200,
    ) -> None:
        self._frames: list[PlaybackFrame] = []
        self._index = 0
        self._duration = max(0.0, float(window_duration))

        if self._duration <= 0.0:
            return

        start = float(max(timeline_start, window_start))
        final_start = float(max(timeline_start, timeline_end - self._duration))
        if start > final_start:
            start = final_start

        step_value = float(step) if step and step > 0 else self._duration / 10.0
        if step_value <= 0.0:
            step_value = max(self._duration / 10.0, 1e-3)

        points = trajectories if trajectories is not None else np.empty((0, 0))
        lines = np.asarray(lines)
        points = np.asarray(points)

        current = start
        frames_built = 0
        while current <= final_start + 1e-9 and frames_built < max_frames:
            frame_end = current + self._duration
            frame_lines = _slice_lines(lines, current, frame_end, decimation, preview_limit)
            frame_points = _slice_points(points, current, frame_end)
            playhead = frame_points[-1] if frame_points.size and frame_points.ndim == 2 else None
            self._frames.append(PlaybackFrame(current, frame_end, frame_lines, frame_points, playhead))
            frames_built += 1
            current += step_value

        if not self._frames:
            frame_lines = _slice_lines(lines, start, start + self._duration, decimation, preview_limit)
            frame_points = _slice_points(points, start, start + self._duration)
            playhead = frame_points[-1] if frame_points.size and frame_points.ndim == 2 else None
            self._frames.append(PlaybackFrame(start, start + self._duration, frame_lines, frame_points, playhead))

    def reset(self) -> None:
        self._index = 0

    def next_frame(self) -> PlaybackFrame | None:
        if self._index >= len(self._frames):
            return None
        frame = self._frames[self._index]
        self._index += 1
        return frame

    def frame_count(self) -> int:
        return len(self._frames)


def _slice_lines(
    lines: np.ndarray,
    start: float,
    end: float,
    decimation: int,
    preview_limit: int,
) -> np.ndarray:
    if lines.size == 0:
        return np.empty((0, 0))
    timestamps = lines[:, 0]
    mask = (timestamps >= start) & (timestamps < end)
    subset = lines[mask]
    if subset.size == 0:
        return subset
    if decimation > 1:
        subset = subset[:: int(decimation)]
    if preview_limit > 0:
        subset = subset[:preview_limit]
    return subset


def _slice_points(points: np.ndarray, start: float, end: float) -> np.ndarray:
    if points.size == 0 or points.ndim != 2 or points.shape[1] == 0:
        return np.empty((0, 0))
    times = points[:, 0]
    mask = (times >= start) & (times < end)
    return points[mask]
