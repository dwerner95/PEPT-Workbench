# PEPT GUI Coding Agent

Author: Dominik’s coding agent

Target stack: Python 3.10+, Qt for Python (PySide6), Plotly for embedded 3D, PEPT library

Status: Technical design and implementation brief

## 1. Problem statement

We need a desktop graphical user interface that lets a user:

1. Load raw PEPT datasets and immediately visualise Lines of Response as a 3D plot.
2. Build a post‑processing pipeline with drag and drop using the official PEPT pipeline concepts.
3. Execute the pipeline on a smart, constant time window subset of the data, independent of sample size or overlap adjustments.
4. Inspect outputs as trajectories and common post‑processing visualisations, with responsive interaction and without blocking the UI.

The app must be engineered for computationally expensive workloads. Users must be able to restrict processing to a fixed wall‑time interval, for example 60 s to 70 s, regardless of how samples are chunked. The UI will guarantee that a constant time slice is honoured even if the user changes adaptive sampling, overlap, or sample size.

## 2. Canonical terms from the PEPT library

The PEPT library provides:

* `pept.LineData` and `pept.PointData` base classes for LoRs and tracked points.
* `pept.Pipeline` chaining Filters and Reducers, with joblib or other executors.
* `pept.TimeWindow` and `pept.AdaptiveWindow` for time based sampling.
* Tracking transformers such as `BirminghamMethod`, `Centroids`, `HDBSCAN`, `Segregate`, `Stack`, `Interpolate`, `Velocity`, and more.
* Plotly based visualisation helpers via `pept.plots.PlotlyGrapher` for interactive 3D rendering.

The GUI must lean on these core ideas and expose them visually.

## 3. User stories

* As a user, I drag a file or folder into the app. The app detects format and initialises a `LineData` provider using the relevant `pept.scanners.*` converter. Instantly I see a 3D LoR preview.
* As a user, I choose a time window, for example \[60.0 s, 70.0 s]. The preview and any pipeline runs operate only on LoRs whose timestamps fall within this exact interval.
* As a user, I compose a pipeline by dragging transformers into a canvas. The app executes the chain on the chosen time window and shows trajectories.
* As a user, I change sample size, overlap or switch to adaptive windows. The analysed interval stays the same absolute time span.
* As a user, I save a workspace that captures source files, time window, pipeline graph, parameters and figure layouts.
* As a user, I export figures to HTML or static images.

## 4. Key behaviours and requirements

### 4.1 Constant time window slicing

* Maintain an immutable time filter defined in absolute seconds from the dataset’s start time.
* Implement a `TimeMask` that filters raw LoRs by the pair `[t_start, t_end)` before constructing any `LineData` samples.
* If the dataset has multiple files stitched together, compute a unified time axis during import.
* When the user edits sample size, overlap, or switches to `TimeWindow` or `AdaptiveWindow`, the upstream `TimeMask` remains active so the processed population cannot leak outside the chosen interval.

### 4.2 Live preview that scales

* Show a decimated LoR preview immediately on load to keep the UI responsive.
* Allow the user to toggle between decimated preview and full resolution for the selected time slice.
* Defer heavy computations to worker threads or processes. Use PEPT’s `Pipeline.fit(executor="joblib")` by default.
* Always keep the main Qt thread free. Use signals to stream incremental results into plots.

### 4.3 Pipeline builder

* Visual canvas with nodes for Filters and Reducers. Edges represent flow.
* Nodes expose their parameters with forms.
* Permit two ways to build:

  1. Visual graph that compiles to `pept.Pipeline([...])`.
  2. A code view showing the equivalent Python for copy and reproducibility.
* Support saving and loading pipeline presets.

### 4.4 Visualisation

* Panel A: LoR preview from `LineData` for the selected time window.
* Panel B: Trajectory view from pipeline output `PointData`.
* Optional subplots for error histograms, labels, and velocity maps.
* Use Plotly embedded in Qt via `QWebEngineView` to display interactive figures created by `pept.plots.PlotlyGrapher`.
* Provide export as HTML or PNG using Plotly’s built in image export.

### 4.5 Performance

* Default to `executor="joblib"`, with max workers equal to CPU count.
* Allow switching to sequential for debugging or to an MPI executor when available.
* Provide options to decimate LoRs for preview, and to cache intermediate samples.
* Persevere with zero copy hand‑offs wherever possible, and ensure reducers are only invoked when necessary.
* Keep a simple profiling overlay in the UI to display wall time per stage.

## 5. Data import

1. **Automatic source detection**

   * ADAC Forte binary: `pept.scanners.adac_forte("*.da*")`
   * Parallel screens CSV: `pept.scanners.parallel_screens(csv, screen_separation)`
   * Modular camera: `pept.scanners.modular_camera(path)`
2. **Unified model**

   * Always convert to `pept.LineData` with supplied `sample_size` and `overlap` parameters, but only after applying the constant time mask.
3. **Large files**

   * Provide chunked reading and memory mapped access for previews.

## 6. Draft UI layout

* **Top bar**: Open dataset, Save workspace, Export figure, Run pipeline, Stop.
* **Left sidebar**: Data inspector and time window selector.
* **Centre**: Tabbed views: LoRs 3D, Trajectories 3D, Diagnostics.
* **Right sidebar**: Pipeline graph and parameters.
* **Status bar**: Progress, executor policy, elapsed time.

## 7. Example interactions with PEPT

### 7.1 Plotting LoRs and points

```python
import pept
from pept.plots import PlotlyGrapher
import numpy as np

# Example LoRs preview
lines_raw = np.arange(70).reshape((10, 7))
lines = pept.LineData(lines_raw)

PlotlyGrapher().add_lines(lines).show()
```

### 7.2 Initialising scanner data

```python
import pept

# ADAC Forte binary files, globbed
lines = pept.scanners.adac_forte("/data/run_01/data.da*")

# Parallel screens CSV, with known screen separation in mm
lines = pept.scanners.parallel_screens(csv_or_array, screen_separation=500)

# Modular camera
lines = pept.scanners.modular_camera("/data/camera.bin")
```

### 7.3 Adaptive sampling and pipeline

```python
import pept
import pept.tracking as pt

lors = pept.LineData(...)

pipeline = pept.Pipeline([
    pt.OptimizeWindow(ideal_elems=200),
    pt.BirminghamMethod(fopt=0.5),
    pt.Stack(),
])

locations = pipeline.fit(lors)  # executor defaults to "joblib"
```

### 7.4 Birmingham Method with separation

```python
import pept
from pept.tracking import BirminghamMethod, Segregate, Stack

pipeline = pept.Pipeline([
    BirminghamMethod(fopt=0.5),
    Segregate(window=20, cut_distance=10),
    Stack(),
])
locations = pipeline.fit(lors)
```

### 7.5 Fixed time window samples

```python
import pept

# Use a fixed time window of 12 ms with 6 ms overlap
lors = pept.LineData(
    ...,
    sample_size=pept.TimeWindow(12.0),
    overlap=pept.TimeWindow(6.0),
)
```

## 8. Making constant time windows first‑class in the GUI

We will not rely solely on `sample_size` and `overlap` to scope the analysis window because those can shift which elements are included. Instead:

* The importer reads raw timestamps and builds a monotonic time axis.
* The UI always applies a top level `[t_start, t_end]` filter to form a **masked LineData**. This yields deterministic membership independent of subsequent adaptive sampling.
* Changes to sample size and overlap only affect how elements within the mask are grouped into samples, not which elements are included.
* If the current adaptive strategy would yield empty samples inside the mask, we show a non‑blocking warning and suggest alternatives.

## 9. Threading and processes

* Use a `QThreadPool` or `QFuture` based runner for pipeline execution.
* Each run constructs a fresh `pept.Pipeline` from the graph model and calls `fit(masked_lors, executor="joblib")`.
* Stream partial results to the UI through Qt signals.
* Provide a hard cancel that aborts workers and clears queues.

## 10. Package layout

```
pept_gui/
  __init__.py
  app.py                 # QApplication bootstrap
  main_window.py         # QMainWindow, wiring of panes
  io/
    __init__.py
    detectors.py         # ADAC Forte, modular, parallel screens detection and loading
    time_mask.py         # Constant time range slicing helpers
    workspace.py         # Save and load workspace state
  model/
    __init__.py
    pipeline_graph.py    # Node and edge models, serialisation
    pept_bridge.py       # Compile graph to pept.Pipeline
    run_controller.py    # Threaded runs, progress and cancellation
  view/
    __init__.py
    graph_view.py        # Drag and drop pipeline canvas
    lors_view.py         # LoR 3D view using Plotly in QWebEngineView
    traj_view.py         # Trajectory 3D view
    diagnostics_view.py  # Error histograms, velocity, labels
    controls.py          # Forms for node parameters and time window picker
  resources/
    qml/                 # If using QML for graph editor
    icons/
  tests/
    test_time_mask.py
    test_pipeline_compile.py
    test_imports.py
  pyproject.toml         # PEP 621 build config
  README.md
```

## 11. Node palette and parameters

* **Sources**: ADAC Forte, Modular Camera, Parallel Screens.
* **Windowing**: TimeWindow, AdaptiveWindow, Overlap control.
* **Tracking**: BirminghamMethod, Centroids, LinesCentroids, HDBSCAN, FPI, Cutpoints, Minpoints, Segregate, Reconnect, Interpolate, Velocity, OutOfViewFilter, RemoveStatic.
* **Reducers**: Stack, GroupBy, SplitLabels, SplitAll.
* **Post processing**: DynamicProbability2D/3D, ResidenceDistribution2D/3D, VectorField2D/3D, RelativeDeviations, AutoCorrelation, SpatialProjections.
* **Utilities**: Condition, SamplesCondition, Debug, Reorient, Voxelize.

Each node binds to the corresponding PEPT class, exposing init parameters, with validation and sensible defaults.

## 12. Embedding Plotly in Qt

* Use `QWebEngineView` to render the Plotly figure produced by `pept.plots.PlotlyGrapher`.
* Provide an export helper that writes `figure.write_html()` and `figure.write_image()` when `kaleido` is installed.

## 13. Example code snippets used by the GUI backend

### 13.1 Compiling a node graph to a pipeline

```python
# model/pept_bridge.py
from __future__ import annotations
import pept
import pept.tracking as pt

TRANSFORMERS = {
    "BirminghamMethod": pt.BirminghamMethod,
    "Centroids": pt.Centroids,
    "HDBSCAN": pt.HDBSCAN,
    "Segregate": pt.Segregate,
    "Stack": pt.Stack,
    "Interpolate": pt.Interpolate,
    "Velocity": pt.Velocity,
    # ... extend as needed
}

def compile_pipeline(nodes: list[dict]) -> pept.Pipeline:
    steps = []
    for n in nodes:
        cls = TRANSFORMERS[n["type"]]
        steps.append(cls(**n.get("params", {})))
    return pept.Pipeline(steps)
```

### 13.2 Enforcing a constant time mask

```python
# io/time_mask.py
import numpy as np

def mask_by_time(lors_array: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
    t = lors_array[:, 0]
    keep = (t >= t_start) & (t < t_end)
    return lors_array[keep]
```

### 13.3 Running the pipeline in a worker

```python
# model/run_controller.py
from PySide6.QtCore import QObject, Signal, Slot, QRunnable
import pept

class PipelineTask(QRunnable):
    def __init__(self, pipeline: pept.Pipeline, samples):
        super().__init__()
        self.pipeline = pipeline
        self.samples = samples

    def run(self):
        points = self.pipeline.fit(self.samples, executor="joblib")
        # emit a signal with results
```

## 14. File formats and persistence

* **Workspace**: JSON containing source paths, time mask, pipeline nodes, and view state.
* **Figures**: HTML and PNG via Plotly.
* **Checkpoints**: pickle via `pept.save()` at key steps if the user asks to cache.

## 15. Testing

* Unit tests for time masking and pipeline compilation.
* Golden image tests for a few demo pipelines using small synthetic datasets.
* Performance tests asserting wall time bounds on moderate inputs.

## 16. Distribution and packaging

* `pyproject.toml` with `hatchling` or `setuptools`.
* Entry point `pept-gui = pept_gui.app:main`.
* Optional `conda` environment file including `pept`, `PySide6`, `plotly`, `kaleido`, `numpy`, `joblib`.

## 17. Milestones

1. Skeleton app with data import, constant time window preview, and LoR plot.
2. Visual pipeline canvas that compiles to `pept.Pipeline`.
3. Trajectory rendering and basic reducers.
4. Performance tuning and cancellation.
5. Export, workspaces, and tests.

## 18. Acceptance criteria

* Load a dataset, select 60 s to 70 s, see LoRs immediately.
* Compose a pipeline with BirminghamMethod and Stack, run, and see trajectories for the same fixed interval.
* Changing sample size and overlap does not change which elements are included within the chosen window.
* Exports work and UI remains responsive during processing.

## 19. Notes and references

The following library features and examples inform this design:

* `pept.plots.PlotlyGrapher` for interactive 3D LoR and point plots, including subplots and colourbars.
* Scanner conversion helpers `pept.scanners.adac_forte`, `parallel_screens`, and `modular_camera` for initialising `LineData`.
* Adaptive sampling strategy using `pept.AdaptiveWindow` and `pept.tracking.OptimizeWindow` with example pipelines.
* Birmingham Method recipes and separation with `Segregate` and `Stack`.
* `pept.Pipeline` execution model and executors, with examples of composition using `+` or explicit lists.
* `pept.TimeWindow` for fixed windows applied to `sample_size` and `overlap` when desired.

Please see the docs you shared for exact examples and API signatures.
