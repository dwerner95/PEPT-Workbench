"""Dataset detection and loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .time_mask import TimeMask, concatenate_and_mask, decimate

try:  # pragma: no cover - exercised in integration environments
    import pept
except ImportError:  # pragma: no cover - import guarded for testing without pept
    pept = None  # type: ignore[assignment]

ADAC_SUFFIXES = {".da", ".dac", ".daq", ".dat"}
PARALLEL_SUFFIXES = {".csv"}
MODULAR_SUFFIXES = {".bin"}


def prepare_samples_argument(value: Any) -> Any | None:
    """Convert LoR-count inputs into PEPT sample window objects when available."""

    if value is None:
        return None
    if isinstance(value, (int, float)) and value <= 0:
        return None
    if pept is None:
        return value

    if not isinstance(value, (int, float)):
        module_name = value.__class__.__module__
        if module_name.startswith("pept"):
            return value
        return value

    candidate_modules = [getattr(pept, "samples", None), pept]
    for module in candidate_modules:
        if module is None:
            continue
        for attr in ("SamplesWindow", "StaticSamples", "Samples", "FixedSamples"):
            cls = getattr(module, attr, None)
            if cls is None:
                continue
            try:
                return cls(int(value))
            except Exception:  # pragma: no cover - fallback if signature mismatch
                continue
    return int(value)


@dataclass(frozen=True)
class DatasetDescriptor:
    """Description of the detected scanner dataset."""

    scanner: str
    paths: tuple[Path, ...]
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def requires_screen_separation(self) -> bool:
        return self.scanner == "parallel_screens"


@dataclass
class LoadedDataset:
    """Represents the imported dataset and derived helper arrays."""

    descriptor: DatasetDescriptor
    raw_lines: np.ndarray
    masked_lines: np.ndarray
    preview_lines: np.ndarray
    line_data: Any | None


class DatasetDetectionError(RuntimeError):
    """Raised when the dataset type could not be determined."""


def detect_sources(paths: Sequence[str | Path]) -> DatasetDescriptor:
    """Detect the dataset scanner type from the provided ``paths``."""
    files = _collect_files(paths)
    if not files:
        raise DatasetDetectionError("No input files found")

    suffixes = {path.suffix.lower() for path in files}
    if any(suffix.startswith(".da") for suffix in suffixes) or _matches(suffixes, ADAC_SUFFIXES):
        relevant = tuple(
            sorted(
                f
                for f in files
                if f.suffix.lower().startswith(".da") or f.suffix.lower() in ADAC_SUFFIXES
            )
        )
        return DatasetDescriptor("adac_forte", relevant)
    if _matches(suffixes, PARALLEL_SUFFIXES):
        relevant = tuple(sorted(files))
        return DatasetDescriptor("parallel_screens", relevant)
    if _matches(suffixes, MODULAR_SUFFIXES):
        relevant = tuple(sorted(files))
        return DatasetDescriptor("modular_camera", relevant)

    raise DatasetDetectionError(f"Could not determine scanner type for suffixes: {sorted(suffixes)}")


def load_dataset(
    descriptor: DatasetDescriptor,
    *,
    time_mask: TimeMask | None = None,
    sample_size: Any | None = None,
    overlap: Any | None = None,
    decimate_every: int = 10,
    screen_separation: float | None = None,
) -> LoadedDataset:
    """Load the dataset described by ``descriptor`` and return filtered LoRs."""
    if pept is None:  # pragma: no cover - executed only without dependency
        raise RuntimeError("The pept package is required to load datasets")
    scanner_module = getattr(pept, "scanners", None)
    if scanner_module is None:  # pragma: no cover - handled at runtime
        raise RuntimeError("pept library does not expose scanners module")

    loader = getattr(scanner_module, descriptor.scanner)

    loader_args: list[Any] = []
    loader_kwargs: dict[str, Any] = {}
    paths = [str(path) for path in descriptor.paths]

    if descriptor.scanner == "parallel_screens":
        separation = screen_separation if screen_separation is not None else descriptor.options.get("screen_separation")
        if separation is None:
            raise ValueError("screen_separation must be provided for parallel screens datasets")
        loader_args = [paths[0] if len(paths) == 1 else paths]
        loader_kwargs["screen_separation"] = separation
    elif descriptor.scanner == "modular_camera":
        loader_args = [paths[0]]
    else:  # adac forte, default fallback
        loader_args = [paths if len(paths) > 1 else paths[0]]

    raw_line_data = loader(*loader_args, **loader_kwargs)
    raw_array = _line_data_to_array(raw_line_data)
    masked = concatenate_and_mask([raw_array], time_mask)
    preview = decimate(masked, every=decimate_every)

    kwargs: dict[str, Any] = {}
    sample_value = prepare_samples_argument(sample_size)
    overlap_value = prepare_samples_argument(overlap)
    if sample_value is not None:
        kwargs["sample_size"] = sample_value
    if overlap_value is not None:
        kwargs["overlap"] = overlap_value

    line_obj: Any | None
    try:
        line_obj = pept.LineData(masked, **kwargs) if kwargs else pept.LineData(masked)
    except Exception:  # pragma: no cover - protective fallback
        line_obj = None

    return LoadedDataset(descriptor, raw_array, masked, preview, line_obj)


# ---------------------------------------------------------------------------
# Helpers

def _matches(suffixes: set[str], candidates: set[str]) -> bool:
    return any(suffix in candidates for suffix in suffixes)


def _collect_files(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    collected: list[Path] = []
    for item in paths:
        path = Path(item).expanduser()
        if path.is_dir():
            collected.extend(sorted(p for p in path.rglob("*") if p.is_file()))
        elif path.exists():
            collected.append(path)
    return tuple(collected)


def _line_data_to_array(line_data: Any) -> np.ndarray:
    """Attempt to convert ``line_data`` to a numpy array."""

    def _coerce(obj: Any) -> np.ndarray | None:
        if isinstance(obj, np.ndarray):
            return obj

        array_attr = getattr(obj, "array", None)
        if array_attr is not None:
            try:
                array = array_attr() if callable(array_attr) else array_attr
            except Exception:  # pragma: no cover - defensive
                array = None
            if array is not None:
                if array is obj:
                    return None
                coerced = _coerce(array)
                if coerced is not None:
                    return coerced

        lines_attr = getattr(obj, "lines", None)
        if lines_attr is not None:
            try:
                lines = lines_attr() if callable(lines_attr) else lines_attr
            except Exception:  # pragma: no cover - defensive
                lines = None
            if lines is not None:
                if lines is obj:
                    return None
                coerced = _coerce(lines)
                if coerced is not None:
                    return coerced

        to_numpy = getattr(obj, "to_numpy", None)
        if callable(to_numpy):
            try:
                array = np.asarray(to_numpy())
            except Exception:  # pragma: no cover - defensive
                array = None
            if array is not None and array.ndim >= 2:
                return array

        try:
            array = np.asarray(obj)
        except Exception:  # pragma: no cover - defensive
            return None
        return array if array.ndim >= 2 else None

    array = _coerce(line_data)
    if array is not None:
        return array

    if hasattr(line_data, "__len__") and hasattr(line_data, "__getitem__"):
        try:
            length = len(line_data)
        except Exception:  # pragma: no cover - defensive
            length = 0
        if length:
            chunks: list[np.ndarray] = []
            for idx in range(length):
                sub_array = _coerce(line_data[idx])
                if sub_array is not None:
                    chunks.append(sub_array)
            if chunks:
                if len(chunks) == 1:
                    return chunks[0]
                return np.concatenate(chunks, axis=0)

    return np.asarray(line_data)
