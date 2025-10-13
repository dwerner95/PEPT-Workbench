"""PEPT GUI package exposing the QApplication launcher lazily."""

from __future__ import annotations

from typing import Any


def main(*args: Any, **kwargs: Any) -> None:
    """Entry point that imports the Qt application on demand."""

    from .app import main as _main

    _main(*args, **kwargs)


__all__ = ["main"]
