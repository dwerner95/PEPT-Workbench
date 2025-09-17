from __future__ import annotations

from pathlib import Path
import types

import numpy as np
import pytest

from pept_gui.io import TimeMask, detect_sources, load_dataset, prepare_samples_argument
from pept_gui.io.detectors import _line_data_to_array, pept as DETECTOR_PEPT


def test_detect_sources_handles_da_variants(tmp_path):
    files = [
        tmp_path / "example.da01",
        tmp_path / "example.da02",
        tmp_path / "example.da03",
        tmp_path / "ignore.txt",
    ]
    for path in files:
        path.write_bytes(b"")

    descriptor = detect_sources([tmp_path])
    assert descriptor.scanner == "adac_forte"
    expected = tuple(sorted(path for path in files[:3]))
    assert descriptor.paths == expected


def test_detect_sources_with_real_dataset():
    dataset = Path(__file__).with_name("closed_400.da01")
    descriptor = detect_sources([dataset])
    assert descriptor.scanner == "adac_forte"
    assert descriptor.paths[0] == dataset


def test_line_data_to_array_handles_sequences():
    class FakeLineData:
        def __init__(self, chunks):
            self._chunks = chunks

        def __len__(self):
            return len(self._chunks)

        def __getitem__(self, idx):
            return self._chunks[idx]

    chunks = [np.ones((2, 7)), np.zeros((3, 7))]
    raw = FakeLineData(chunks)
    array = _line_data_to_array(raw)
    assert array.shape == (5, 7)
    assert np.allclose(array[:2], 1.0)
    assert np.allclose(array[2:], 0.0)


def test_prepare_samples_argument_prefers_pept_samples(monkeypatch):
    fake_samples_module = types.SimpleNamespace(SamplesWindow=lambda count: ("samples", count))
    fake_pept = types.SimpleNamespace(samples=fake_samples_module)
    monkeypatch.setattr("pept_gui.io.detectors.pept", fake_pept, raising=False)

    result = prepare_samples_argument(25)
    assert result == ("samples", 25)

    assert prepare_samples_argument(0) is None

    monkeypatch.setattr("pept_gui.io.detectors.pept", DETECTOR_PEPT, raising=False)


@pytest.mark.integration
def test_load_dataset_and_time_mask_behaviour():
    pept = pytest.importorskip("pept")
    dataset = Path(__file__).with_name("closed_400.da01")
    descriptor = detect_sources([dataset])

    full = load_dataset(descriptor, time_mask=None, decimate_every=1)
    assert full.raw_lines.shape[1] == 7
    assert full.raw_lines.shape[0] >= full.masked_lines.shape[0] > 0
    assert full.preview_lines.shape == full.masked_lines.shape
    assert isinstance(full.line_data, pept.LineData)

    times = full.raw_lines[:, 0]
    duration = float(times[-1] - times[0])
    window = max(1e-3, min(duration / 4.0 if duration > 0 else 1e-3, 0.1))
    mask = TimeMask(float(times[0]), float(times[0] + window))

    subset = load_dataset(descriptor, time_mask=mask, decimate_every=1)
    assert subset.masked_lines.shape[0] >= 1
    assert np.all(subset.masked_lines[:, 0] >= mask.start)
    assert np.all(subset.masked_lines[:, 0] < mask.end)
    assert subset.preview_lines.shape == subset.masked_lines.shape
    assert isinstance(subset.line_data, pept.LineData)

    decimated = load_dataset(descriptor, time_mask=mask, decimate_every=5)
    assert decimated.preview_lines.shape[0] <= subset.masked_lines.shape[0]
