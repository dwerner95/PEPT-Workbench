import numpy as np

from pept_gui.model.sample_analysis import annotate_points_with_samples, build_sample_windows


def test_build_sample_windows_with_overlap():
    timestamps = np.array([0.0, 0.5, 1.0, 1.6, 2.4, 3.0, 3.4])
    windows = build_sample_windows(timestamps, sample_size=3, overlap=1)
    assert len(windows) == 3
    assert windows[0].start_time == 0.0
    assert windows[0].end_time == 1.0
    assert windows[1].start_time == 1.0
    assert windows[1].count == 3
    assert windows[2].count == 2  # final partial window


def test_build_sample_windows_full_span_when_disabled():
    timestamps = np.array([10.0, 11.0, 12.0])
    windows = build_sample_windows(timestamps, sample_size=0, overlap=0)
    assert len(windows) == 1
    first = windows[0]
    assert first.start_time == 10.0
    assert first.end_time == 12.0
    assert first.count == 3


def test_annotate_points_with_samples_computes_distances():
    timestamps = np.linspace(0.0, 2.0, 5)
    windows = build_sample_windows(timestamps, sample_size=3, overlap=0)
    points = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.5, 1.0, 2.0, 0.0],
        [2.0, 1.0, 3.0, 0.0],
    ])
    custom, distances = annotate_points_with_samples(points, windows)
    assert custom.shape == (5, 5)
    # first three points belong to first sample, last two to second
    assert np.allclose(custom[:3, 1], 0)  # sample index
    assert np.allclose(custom[3:, 1], 1)
    # distance travelled in first sample is sum of two unit steps
    assert distances[0] == 2.0
    assert np.allclose(custom[:3, 4], 2.0)
    # distance travelled within second sample is length between points (1 unit)
    assert distances[1] == 1.0
    # customdata stores distance in column 4 (0-based index 4)
    assert np.allclose(custom[3:, 4], 1.0)
