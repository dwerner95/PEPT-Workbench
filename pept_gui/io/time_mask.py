"""Time masking utilities to enforce constant absolute ranges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

ArrayLike = np.ndarray


@dataclass(frozen=True)
class TimeMask:
    """Represents an immutable inclusive-exclusive time interval."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end <= self.start:
            raise ValueError("end must be greater than start")

    @property
    def duration(self) -> float:
        return self.end - self.start

    def contains(self, value: float) -> bool:
        return self.start <= value < self.end

    def mask(self, timestamps: ArrayLike) -> ArrayLike:
        """Return a boolean mask for ``timestamps`` that fall into the interval."""
        return (timestamps >= self.start) & (timestamps < self.end)

    def apply(self, array: ArrayLike, *, time_column: int = 0) -> ArrayLike:
        """Filter ``array`` rows whose timestamp is inside the window."""
        if array.ndim != 2:
            raise ValueError("array must be 2D with rows representing LoRs")
        mask = self.mask(array[:, time_column])
        return array[mask]

    def clamp(self, timestamp: float) -> float:
        """Clamp ``timestamp`` into the interval."""
        if timestamp < self.start:
            return self.start
        if timestamp >= self.end:
            return self.end
        return timestamp


def mask_by_time(array: ArrayLike, t_start: float, t_end: float, *, time_column: int = 0) -> ArrayLike:
    """Convenience helper mirroring classic behaviour."""
    return TimeMask(t_start, t_end).apply(array, time_column=time_column)


def stitch_time_axes(arrays: Sequence[ArrayLike], *, time_column: int = 0) -> list[ArrayLike]:
    """Return copies of ``arrays`` with monotonically increasing timestamps.

    Some scanners reset timestamps per file; this function shifts subsequent
    chunks only when their starting timestamp would move backwards relative to
    the last element of the previous chunk. When the time axis is already
    monotonic no adjustment is performed.
    """
    aligned: list[ArrayLike] = []
    last_time: float | None = None
    for chunk in arrays:
        if chunk.size == 0:
            aligned.append(chunk)
            continue
        adjusted = chunk.copy()
        if last_time is not None:
            start = float(adjusted[0, time_column])
            if start <= last_time:
                shift = last_time - start
                adjusted[:, time_column] = adjusted[:, time_column] + shift
        last_time = float(adjusted[-1, time_column]) if adjusted.size else last_time
        aligned.append(adjusted)
    return aligned


def concatenate_and_mask(
    arrays: Sequence[ArrayLike],
    mask: TimeMask | None,
    *,
    time_column: int = 0,
) -> ArrayLike:
    """Concatenate ``arrays`` and apply an optional ``mask``."""
    if not arrays:
        return np.empty((0, 7))
    stitched = stitch_time_axes(arrays, time_column=time_column)
    combined = np.concatenate(stitched, axis=0)
    return combined if mask is None else mask.apply(combined, time_column=time_column)


def decimate(array: ArrayLike, *, every: int) -> ArrayLike:
    """Return every ``n``-th element for quick previews."""
    if every <= 1:
        return array
    return array[::every]
