"""Reusable Plotly embedding widget for Qt."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from PySide6 import QtCore, QtGui, QtWidgets

try:  # pragma: no cover - optional dependency
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - fallback when WebEngine unavailable
    QWebEngineView = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import plotly.io as pio
except ImportError:  # pragma: no cover - allow running without plotly
    pio = None  # type: ignore[assignment]


class PlotlyWidget(QtWidgets.QWidget):
    """Wraps a :class:`QWebEngineView` (or fallback) to display Plotly figures."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if QWebEngineView is not None:
            self._view: QtWidgets.QWidget = QWebEngineView(self)
        else:
            browser = QtWidgets.QTextBrowser(self)
            browser.setOpenExternalLinks(True)
            browser.setHtml("<h3>Plot preview requires QtWebEngine</h3>")
            self._view = browser
        layout.addWidget(self._view)
        self._last_html: str = ""
        self._temp_file: Optional[Path] = None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # type: ignore[override]
        self._cleanup_temp_file()
        super().closeEvent(event)

    def _cleanup_temp_file(self) -> None:
        if self._temp_file and self._temp_file.exists():
            try:
                os.unlink(self._temp_file)
            except OSError:
                pass
            self._temp_file = None

    def set_html(self, html: str) -> None:
        self._last_html = html
        if hasattr(self._view, "setHtml") and not isinstance(self._view, QWebEngineView):
            set_html = getattr(self._view, "setHtml")  # type: ignore[attr-defined]
            set_html(html)  # type: ignore[misc]
        elif isinstance(self._view, QWebEngineView):
            self._cleanup_temp_file()
            fd, name = tempfile.mkstemp(suffix=".html")
            os.close(fd)
            tmp = Path(name)
            tmp.write_text(html, encoding="utf-8")
            self._temp_file = tmp
            self._view.setUrl(QtCore.QUrl.fromLocalFile(str(tmp)))  # type: ignore[arg-type]

    def set_figure(self, figure: Any) -> None:
        if pio is None:
            self.set_html("<h3>Plotly is not installed.</h3>")
            return
        html = pio.to_html(figure, full_html=True, include_plotlyjs="inline")
        self.set_html(html)

    def last_html(self) -> str:
        return self._last_html

    def __del__(self) -> None:
        self._cleanup_temp_file()
