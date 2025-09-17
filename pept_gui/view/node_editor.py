"""Editor widget for adjusting pipeline node parameters."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict

from PySide6 import QtCore, QtWidgets

from ..model import PipelineNode


@dataclass
class _ParamWidget:
    widget: QtWidgets.QWidget
    default: Any
    annotation: Any
    kind: inspect._ParameterKind


class NodeParameterEditor(QtWidgets.QWidget):
    """Displays editable parameters for the selected pipeline node."""

    parameter_changed = QtCore.Signal(str, dict)

    FALLBACK_SIGNATURES: dict[str, list[inspect.Parameter]] = {
        "Condition": [inspect.Parameter("conditions", inspect.Parameter.VAR_POSITIONAL)],
    }

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._registry: Dict[str, Callable[..., Any]] = {}
        self._resolver: Callable[[str], Callable[..., Any] | None] | None = None
        self._current_node: PipelineNode | None = None
        self._widgets: dict[str, _ParamWidget] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._title = QtWidgets.QLabel("Select a node to edit parameters", self)
        layout.addWidget(self._title)

        self._form_widget = QtWidgets.QWidget(self)
        self._form_layout = QtWidgets.QFormLayout(self._form_widget)
        self._form_layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        layout.addWidget(self._form_widget)

        button_row = QtWidgets.QHBoxLayout()
        self._reset_button = QtWidgets.QPushButton("Reset", self)
        self._reset_button.clicked.connect(self._reset_defaults)
        self._apply_button = QtWidgets.QPushButton("Apply", self)
        self._apply_button.clicked.connect(self._emit_changes)
        button_row.addWidget(self._reset_button)
        button_row.addWidget(self._apply_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        layout.addStretch(1)

        self._form_widget.setVisible(False)
        self._reset_button.setEnabled(False)
        self._apply_button.setEnabled(False)

    def set_registry(self, registry: Dict[str, Callable[..., Any]]) -> None:
        self._registry = registry

    def set_transformer_resolver(
        self, resolver: Callable[[str], Callable[..., Any] | None]
    ) -> None:
        self._resolver = resolver

    def clear(self) -> None:
        self._current_node = None
        self._remove_form_widgets()
        self._title.setText("Select a node to edit parameters")
        self._form_widget.setVisible(False)
        self._reset_button.setEnabled(False)
        self._apply_button.setEnabled(False)

    def load_node(self, node: PipelineNode) -> None:
        self._current_node = node
        transformer = self._registry.get(node.type)
        if transformer is None and self._resolver is not None:
            resolved = self._resolver(node.type)
            if resolved is not None:
                transformer = resolved
                self._registry[node.type] = resolved
        fallback_parameters = self.FALLBACK_SIGNATURES.get(node.type)
        self._remove_form_widgets()

        if transformer is None and fallback_parameters is None:
            self._title.setText(f"No editor available for {node.type}")
            self._form_widget.setVisible(False)
            self._reset_button.setEnabled(False)
            self._apply_button.setEnabled(False)
            return

        if transformer is not None:
            signature = inspect.signature(transformer.__init__)
            parameters = [
                param
                for name, param in signature.parameters.items()
                if name != "self"
                and param.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                    inspect.Parameter.VAR_POSITIONAL,
                )
            ]
        else:
            parameters = fallback_parameters or []

        self._title.setText(f"Editing {node.type} ({node.id})")
        self._widgets.clear()

        for parameter in parameters:
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                widget = self._create_varargs_widget(parameter, node.params.get("__args__"))
                key = "__args__"
                label = parameter.name or "args"
            else:
                widget = self._create_widget(parameter, node.params.get(parameter.name, inspect._empty))
                key = parameter.name
                label = parameter.name
            if widget is None:
                continue
            self._widgets[key] = widget
            self._form_layout.addRow(label, widget.widget)

        has_widgets = bool(self._widgets)
        self._form_widget.setVisible(has_widgets)
        self._reset_button.setEnabled(has_widgets)
        self._apply_button.setEnabled(has_widgets)

    # ------------------------------------------------------------------ helpers
    def _remove_form_widgets(self) -> None:
        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._widgets.clear()

    def _create_widget(self, parameter: inspect.Parameter, current_value: Any) -> _ParamWidget | None:
        default = parameter.default if parameter.default is not inspect._empty else None
        annotation = parameter.annotation
        value = current_value if current_value is not inspect._empty else default

        if annotation in (bool, "bool") or isinstance(default, bool):
            checkbox = QtWidgets.QCheckBox(self)
            checkbox.setChecked(bool(value) if value is not None else bool(default))
            return _ParamWidget(checkbox, default, annotation, parameter.kind)

        if annotation in (int, "int") or isinstance(default, int):
            spin = QtWidgets.QSpinBox(self)
            spin.setRange(-10_000_000, 10_000_000)
            if value is None:
                value = default if default is not None else 0
            spin.setValue(int(value))
            return _ParamWidget(spin, default, annotation, parameter.kind)

        if annotation in (float, "float") or isinstance(default, float):
            spin = QtWidgets.QDoubleSpinBox(self)
            spin.setRange(-1e9, 1e9)
            spin.setDecimals(6)
            spin.setSingleStep(0.1)
            if value is None:
                value = default if default is not None else 0.0
            spin.setValue(float(value))
            return _ParamWidget(spin, default, annotation, parameter.kind)

        line_edit = QtWidgets.QLineEdit(self)
        if value is not None and value is not inspect._empty:
            line_edit.setText(repr(value) if not isinstance(value, str) else value)
        elif default not in (None, inspect._empty):
            line_edit.setPlaceholderText(repr(default))
        return _ParamWidget(line_edit, default, annotation, parameter.kind)

    def _create_varargs_widget(self, parameter: inspect.Parameter, current_value: Any) -> _ParamWidget:
        values = current_value if isinstance(current_value, (list, tuple)) else []
        editor = QtWidgets.QPlainTextEdit(self)
        editor.setPlaceholderText("One entry per line")
        if values:
            editor.setPlainText("\n".join(str(item) for item in values))
        return _ParamWidget(editor, list(values), parameter.annotation, parameter.kind)

    def _reset_defaults(self) -> None:
        if self._current_node is None:
            return
        for name, holder in self._widgets.items():
            widget = holder.widget
            default = holder.default
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(default))
            elif isinstance(widget, QtWidgets.QSpinBox):
                widget.setValue(int(default) if default is not None else 0)
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.setValue(float(default) if default is not None else 0.0)
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(
                    ""
                    if default in (None, inspect._empty)
                    else (default if isinstance(default, str) else repr(default))
                )
            elif isinstance(widget, QtWidgets.QPlainTextEdit):
                text = "\n".join(str(item) for item in (default or []))
                widget.setPlainText(text)

    def _emit_changes(self) -> None:
        if self._current_node is None:
            return
        updated: dict[str, Any] = dict(self._current_node.params or {})
        for name, holder in self._widgets.items():
            widget = holder.widget
            default = holder.default
            value: Any
            if isinstance(widget, QtWidgets.QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QtWidgets.QSpinBox):
                value = int(widget.value())
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                value = float(widget.value())
            elif isinstance(widget, QtWidgets.QLineEdit):
                text = widget.text().strip()
                if not text:
                    value = None
                else:
                    value = self._parse_literal(text)
            elif isinstance(widget, QtWidgets.QPlainTextEdit):
                raw = widget.toPlainText()
                lines = [line.strip() for line in raw.splitlines() if line.strip()]
                value = lines
            else:
                continue

            if name == "__args__":
                if isinstance(value, list) and value:
                    updated["__args__"] = value
                else:
                    updated.pop("__args__", None)
                continue

            if value is None or value == default or value is inspect._empty:
                updated.pop(name, None)
            else:
                updated[name] = value

        self.parameter_changed.emit(self._current_node.id, updated)

    @staticmethod
    def _parse_literal(text: str) -> Any:
        try:
            return ast.literal_eval(text)
        except Exception:
            return text
