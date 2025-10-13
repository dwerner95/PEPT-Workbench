"""Sample window analytics helpers."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class SampleWindow:
    """Describes a contiguous sample of LoRs on the timeline."""

    index: int
    start_time: float
    end_time: float
    count: int
    start_idx: int
    end_idx: int
    distance: float = float("nan")

    @property
    def span(self) -> float:
        return max(0.0, self.end_time - self.start_time)


def build_sample_windows(timestamps: np.ndarray, sample_size: int, overlap: int) -> List[SampleWindow]:
    """Return sample windows derived from ``timestamps``.

    ``sample_size`` and ``overlap`` are interpreted in LoR counts. A ``sample_size``
    of ``0`` yields a single window that spans the entire timeline.
    """

    times = np.asarray(timestamps, dtype=float).reshape(-1)
    if times.size == 0:
        return []

    size = int(sample_size)
    overlap = int(max(0, overlap))
    if size <= 0 or size >= times.size:
        return [SampleWindow(0, float(times[0]), float(times[-1]), int(times.size), 0, int(times.size))]

    size = max(1, size)
    overlap = min(overlap, size - 1)
    stride = max(1, size - overlap)

    windows: List[SampleWindow] = []
    start_idx = 0
    sample_index = 0
    total = times.size

    while start_idx < total:
        end_idx = min(start_idx + size, total)
        chunk = times[start_idx:end_idx]
        if chunk.size == 0:
            break
        window = SampleWindow(
            index=sample_index,
            start_time=float(chunk[0]),
            end_time=float(chunk[-1]),
            count=int(chunk.size),
            start_idx=int(start_idx),
            end_idx=int(end_idx),
        )
        windows.append(window)
        if end_idx >= total:
            break
        start_idx += stride
        sample_index += 1

    return windows


def annotate_points_with_samples(
    points: np.ndarray,
    windows: Sequence[SampleWindow],
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Return Plotly custom data and per-sample travel distances.

    ``points`` is expected to be shaped ``(N, >=4)`` where the first column is
    time and the next three columns are spatial coordinates in millimetres.
    """

    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError("points must be a 2D array with at least four columns")

    total_points = points.shape[0]
    if total_points == 0:
        return np.empty((0, 5)), {}

    if len(windows) == 0:
        times_only = np.asarray(points[:, 0], dtype=float)
        empty_custom = np.column_stack(
            (
                times_only,
                np.full(times_only.shape, -1.0, dtype=float),
                np.zeros(times_only.shape, dtype=float),
                np.zeros(times_only.shape, dtype=float),
                np.zeros(times_only.shape, dtype=float),
            )
        )
        return empty_custom, {}

    times = np.asarray(points[:, 0], dtype=float)
    sample_ids = np.full(total_points, -1.0, dtype=float)
    spans = np.zeros(total_points, dtype=float)
    counts = np.zeros(total_points, dtype=float)
    distances = np.full(total_points, float("nan"), dtype=float)

    assignments: Dict[int, List[int]] = defaultdict(list)
    current_idx = 0
    total_windows = len(windows)

    for point_idx, timestamp in enumerate(times):
        # advance while the next window starts before the current timestamp
        while current_idx < total_windows - 1 and timestamp >= windows[current_idx + 1].start_time:
            current_idx += 1
        assigned = None
        # probe surrounding windows to account for overlaps
        for candidate in range(max(0, current_idx - 1), min(total_windows, current_idx + 2)):
            window = windows[candidate]
            if timestamp < window.start_time:
                continue
            if timestamp <= window.end_time or candidate == total_windows - 1:
                assigned = window
                current_idx = candidate
                break
        if assigned is None:
            if timestamp < windows[0].start_time:
                assigned = windows[0]
                current_idx = 0
            else:
                assigned = windows[-1]
                current_idx = total_windows - 1

        sample_ids[point_idx] = float(assigned.index)
        spans[point_idx] = assigned.span
        counts[point_idx] = float(assigned.count)
        assignments[assigned.index].append(point_idx)

    distance_lookup: Dict[int, float] = {}
    coords = np.asarray(points[:, 1:4], dtype=float)
    order = np.argsort(times)
    index_position = {idx: pos for pos, idx in enumerate(order)}

    for index, indices in assignments.items():
        if not indices:
            continue
        if len(indices) >= 2:
            ordered = sorted(indices, key=lambda idx: times[idx])
            segment = coords[ordered]
            diffs = np.diff(segment, axis=0)
            travel = float(np.sum(np.linalg.norm(diffs, axis=1)))
            distance_lookup[index] = travel
            for idx in ordered:
                distances[idx] = travel
            continue

        single_idx = indices[0]
        pos = index_position.get(single_idx)
        if pos is None or pos == 0:
            travel = 0.0
        else:
            prev_idx = order[pos - 1]
            travel = float(np.linalg.norm(coords[single_idx] - coords[prev_idx]))
        distance_lookup[index] = travel
        distances[single_idx] = travel

    for window in windows:
        if window.index not in distance_lookup:
            distance_lookup[window.index] = float("nan")

    customdata = np.column_stack((times, sample_ids, spans, counts, distances))
    return customdata, distance_lookup
