import numpy as np

from pept_gui.main_window import _to_array, _compute_speeds, _select_colour


class FakePointData:
    def __init__(self, array):
        self._array = array

    def array(self):
        return self._array


class FakePointDataPoints:
    def __init__(self, array):
        self._points = array
        self.columns = ["t", "x", "y", "z", "error"]

    def points(self):
        return self._points


def test_to_array_stacks_sequence_of_point_data():
    samples = [
        FakePointData(np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])),
        FakePointData(np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])),
    ]
    stacked, columns = _to_array(samples)
    assert stacked.shape == (2, 5)
    assert len(columns) == 5
    assert np.allclose(stacked[0], np.array([0.0, 1.0, 2.0, 3.0, 4.0]))


def test_to_array_handles_structured_arrays():
    structured = np.array(
        [(0.0, 1.0, 2.0, 3.0, 4.0)],
        dtype=[("t", float), ("x", float), ("y", float), ("z", float), ("error", float)],
    )
    array, columns = _to_array(structured)
    assert array.shape == (1, 5)
    assert columns == ["t", "x", "y", "z", "error"]
    assert np.allclose(array[0], [0.0, 1.0, 2.0, 3.0, 4.0])


def test_to_array_handles_object_rows():
    obj_array = np.array([
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    ], dtype=object)
    array, columns = _to_array(obj_array)
    assert array.shape == (2, 5)
    assert len(columns) == 5
    assert np.isclose(array[1, 0], 1.0)


def test_to_array_handles_pointdata_with_points_attr():
    samples = [FakePointDataPoints(np.array([[0.0, 1.0, 2.0, 3.0, 4.0]])) for _ in range(2)]
    array, columns = _to_array(samples)
    assert array.shape == (2, 5)
    assert columns == ["t", "x", "y", "z", "error"]


def test_compute_speeds_handles_small_input():
    empty = _compute_speeds(np.empty((0, 0)))
    assert empty.size == 0

    single = _compute_speeds(np.array([[0.0, 1.0, 2.0, 3.0]]))
    assert single.size == 0

    valid = _compute_speeds(np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]))
    assert np.isclose(valid[0], 1.0)


def test_select_colour_prefers_error_when_available():
    points = np.array(
        [
            [0.0, 1.0, 2.0, 3.0, 0.5],
            [1.0, 2.0, 3.0, 4.0, 0.6],
        ]
    )
    columns = ["t", "x", "y", "z", "error"]
    error_colour, label, name = _select_colour(points, columns, "error")
    assert np.allclose(error_colour, [0.5, 0.6])
    assert label == "Error [mm]"
    assert name == "error"
    time_colour, label, name = _select_colour(points, columns, "time")
    assert np.allclose(time_colour, [0.0, 1.0])
    assert label == "Time [ms]"
    assert name in {"t", "time"}
