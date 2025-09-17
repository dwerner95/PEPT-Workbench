Disclaimer: This is a AI written Program. Use carefully.

# PEPT GUI

PEPT GUI is a Qt for Python desktop application for loading raw PEPT datasets, slicing
constant time windows, and running interactive post-processing pipelines powered by the
[PEPT](https://pept.readthedocs.io) library. The interface provides immediate LoR previews,
a drag-and-drop pipeline builder, and Plotly based 3D visualisations embedded inside Qt.

## Features

- Automatic detection for ADAC Forte, modular camera, and parallel screens datasets.
- Immutable absolute time masks that guarantee deterministic LoR membership regardless of
  sampling or adaptive window adjustments.
- Configurable sample grouping in raw LoR counts, plus adjustable preview limits and colour
  selection for responsive 3D/2D visualisation (position/time series with mm/ms axes).
- Responsive LoR previews and pipeline execution using background worker threads.
- Visual pipeline canvas focused on the core PEPT-ML steps (Cutpoints → HDBSCAN → SplitLabels →
  Centroids → Segregate/Stack) with parameter editors, Python code preview, and preset management.
- Trajectory, diagnostics, and export panels powered by Plotly and `pept.plots` helpers.

## Getting Started

```bash
python -m pip install .
pept-gui
```

## Package Layout

```
pept_gui/
  app.py              # QApplication bootstrap and global wiring
  main_window.py      # QMainWindow composition of toolbars, panes, and status
  io/                 # Data import, time masking, workspace persistence
  model/              # Pipeline graph, PEPT bridge, background execution
  view/               # Qt widgets for graph canvas, visualisations, and controls
  resources/          # Icons, optional QML snippets
  tests/              # Unit test suite
```

## Development

Install dependencies in editable mode:

```bash
python -m pip install -e .[export]
python -m pip install -r requirements-dev.txt  # optional linting/type checking
```

Run the suite:

```bash
pytest
```

## License

GPL v.3
