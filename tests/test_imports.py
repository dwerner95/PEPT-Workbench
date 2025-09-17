import pept_gui
from pept_gui import app
from pept_gui.model import available_transformers


def test_app_main_accessible():
    assert callable(pept_gui.main)
    assert callable(app.main)


def test_available_transformers_returns_dict():
    registry = available_transformers()
    assert isinstance(registry, dict)
