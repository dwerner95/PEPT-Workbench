import types

import numpy as np
import pytest

from pept_gui.model import PipelineGraph, compile_pipeline


@pytest.fixture()
def fake_pept_bridge(monkeypatch):
    import pept_gui.model.pept_bridge as bridge

    class FakeTransformer:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs
            self.args = args

    class FakePipeline:
        def __init__(self, steps):
            self.steps = steps

        def fit(self, samples, executor="joblib"):
            return {
                "executor": executor,
                "steps": self.steps,
                "sample_count": int(np.asarray(samples).shape[0]),
            }

    relevant = [
        "BirminghamMethod",
        "Stack",
        "Cutpoints",
        "HDBSCAN",
        "SplitLabels",
        "Centroids",
        "Segregate",
        "Condition",
    ]
    for name in relevant:
        bridge.TRANSFORMERS[name] = FakeTransformer
    monkeypatch.setattr(bridge, "pept", types.SimpleNamespace(Pipeline=FakePipeline), raising=False)
    yield


def test_pipeline_graph_serialisation_roundtrip():
    graph = PipelineGraph()
    first = graph.add_node("BirminghamMethod")
    second = graph.add_node("Stack")
    graph.connect(first.id, second.id)

    payload = graph.to_dict()
    restored = PipelineGraph.from_dict(payload)

    ordered = restored.ordered_nodes()
    assert [node.type for node in ordered] == ["BirminghamMethod", "Stack"]


@pytest.mark.usefixtures("fake_pept_bridge")
def test_compile_pipeline_builds_pipeline():
    graph = PipelineGraph()
    node_a = graph.add_node("BirminghamMethod", params={"fopt": 0.5})
    pipeline = compile_pipeline(graph.as_pipeline_definition())
    result = pipeline.fit(np.zeros((3, 7)), executor="threaded")

    assert result["executor"] == "threaded"
    assert result["sample_count"] == 3
    assert len(result["steps"]) == 2


@pytest.mark.usefixtures("fake_pept_bridge")
def test_compile_pipeline_respects_existing_reducer():
    graph = PipelineGraph()
    node_a = graph.add_node("BirminghamMethod")
    node_b = graph.add_node("Stack")
    graph.connect(node_a.id, node_b.id)

    pipeline = compile_pipeline(graph.as_pipeline_definition())
    result = pipeline.fit(np.zeros((2, 7)))

    assert len(result["steps"]) == 2  # no implicit Stack appended


@pytest.mark.usefixtures("fake_pept_bridge")
def test_compile_pipeline_unknown_transformer():
    graph = PipelineGraph()
    graph.add_node("Nonexistent")
    with pytest.raises(KeyError):
        compile_pipeline(graph.as_pipeline_definition())


@pytest.mark.usefixtures("fake_pept_bridge")
def test_compile_pipeline_handles_condition_varargs():
    graph = PipelineGraph()
    graph.add_node("Condition", params={"__args__": ["error < 15"]})
    graph.add_node("Stack")
    pipeline = compile_pipeline(graph.as_pipeline_definition())
    result = pipeline.fit(np.zeros((1, 7)))
    assert len(result["steps"]) == 2
