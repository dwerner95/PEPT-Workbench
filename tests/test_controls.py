import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets

from pept_gui.io import TimeMask
from pept_gui.view.controls import ControlsPanel


@pytest.fixture
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_shift_time_mask_advances_within_range(qapp):
    controls = ControlsPanel()
    controls.set_time_range(0.0, 5.0)
    controls.set_time_mask(TimeMask(0.0, 1.0))
    assert controls.shift_time_mask(0.5) is True
    mask = controls.current_time_mask()
    assert mask.start == pytest.approx(0.5)
    assert mask.end == pytest.approx(1.5)

    assert controls.shift_time_mask(10.0) is True
    mask = controls.current_time_mask()
    assert mask.start == pytest.approx(4.0)
    assert controls.shift_time_mask(0.5) is False


def test_set_playback_state_silent(qapp):
    controls = ControlsPanel()
    triggered: list[bool] = []
    controls.playback_toggled.connect(triggered.append)
    controls.set_playback_state(True)
    controls.set_playback_state(False)
    assert triggered == []


def test_set_playback_speed_clamps(qapp):
    controls = ControlsPanel()
    captured: list[float] = []
    controls.playback_speed_changed.connect(captured.append)
    controls.set_playback_speed(0.001)
    controls.set_playback_speed(2000.0)
    assert pytest.approx(controls.playback_speed()) == 1000.0
    assert captured[0] == pytest.approx(0.01)
    assert captured[-1] == pytest.approx(1000.0)

