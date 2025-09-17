"""Application bootstrap utilities."""

from __future__ import annotations

import os
import sys
from typing import Iterable, Sequence

from PySide6 import QtCore, QtWidgets

from .main_window import MainWindow


def _apply_qt_defaults() -> None:
    """Configure Qt attributes before creating the application."""
    # Enable high DPI scaling when running on high resolution displays.
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)


def create_application(argv: Sequence[str] | None = None) -> QtWidgets.QApplication:
    """Construct the :class:`QApplication` with standard flags."""
    _apply_qt_defaults()
    app = QtWidgets.QApplication(list(argv or sys.argv))
    app.setOrganizationName("PEPT")
    app.setApplicationName("PEPT GUI")
    return app


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the GUI with the provided arguments."""
    app = create_application(argv)
    window = MainWindow()
    window.show()

    # On macOS we ensure the application keeps running even when the dock icon is clicked
    # without windows.
    if sys.platform == "darwin":
        app.setQuitOnLastWindowClosed(False)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
