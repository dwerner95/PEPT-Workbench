"""Workspace persistence helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .time_mask import TimeMask


@dataclass
class WorkspaceState:
    """Serializable configuration captured from the UI."""

    sources: list[str]
    time_mask: TimeMask
    pipeline: list[dict[str, Any]]
    sample_size: Any | None = None
    overlap: Any | None = None
    view_state: dict[str, Any] = field(default_factory=dict)
    executor: str = "joblib"
    preview_limit: int = 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources": self.sources,
            "time_mask": {"start": self.time_mask.start, "end": self.time_mask.end},
            "pipeline": self.pipeline,
            "sample_size": _encode_parameter(self.sample_size),
            "overlap": _encode_parameter(self.overlap),
            "view_state": _encode_parameter(self.view_state),
            "executor": self.executor,
            "preview_limit": self.preview_limit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WorkspaceState":
        mask = data.get("time_mask") or {}
        start = float(mask.get("start", 0.0))
        end = float(mask.get("end", start + 1.0))
        time_mask = TimeMask(start, end)
        view_state = _decode_parameter(data.get("view_state")) or {}
        return cls(
            sources=list(data.get("sources", [])),
            time_mask=time_mask,
            pipeline=list(data.get("pipeline", [])),
            sample_size=_decode_parameter(data.get("sample_size")),
            overlap=_decode_parameter(data.get("overlap")),
            view_state=view_state,
            executor=str(data.get("executor", "joblib")),
            preview_limit=int(data.get("preview_limit", view_state.get("preview_limit", 1000))),
        )


def save_workspace(path: str | Path, state: WorkspaceState) -> None:
    target = Path(path).expanduser()
    target.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


def load_workspace(path: str | Path) -> WorkspaceState:
    source = Path(path).expanduser()
    data = json.loads(source.read_text(encoding="utf-8"))
    return WorkspaceState.from_dict(data)


def _encode_parameter(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_encode_parameter(v) for v in value]
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode_parameter(v) for v in value]}
    if isinstance(value, dict):
        return {str(k): _encode_parameter(v) for k, v in value.items()}
    return {"__type__": value.__class__.__name__, "repr": repr(value)}


def _decode_parameter(value: Any) -> Any:
    if isinstance(value, dict) and "__type__" in value:
        if value["__type__"] == "tuple":
            return tuple(_decode_parameter(v) for v in value.get("items", []))
        return value
    if isinstance(value, list):
        return [_decode_parameter(v) for v in value]
    if isinstance(value, dict):
        return {k: _decode_parameter(v) for k, v in value.items()}
    return value
