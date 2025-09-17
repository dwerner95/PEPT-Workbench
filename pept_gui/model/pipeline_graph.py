"""In-memory representation of the pipeline graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable


@dataclass
class PipelineNode:
    """A single pipeline transformer/reducer node."""

    id: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)
    position: tuple[float, float] | None = None


class PipelineGraph:
    """Directed acyclic graph describing the user pipeline."""

    def __init__(self) -> None:
        self._nodes: Dict[str, PipelineNode] = {}
        self._forward: Dict[str, set[str]] = {}
        self._reverse: Dict[str, set[str]] = {}
        self._counter = 0

    # ------------------------------------------------------------------ mutation
    def add_node(
        self,
        node_type: str,
        *,
        node_id: str | None = None,
        params: dict[str, Any] | None = None,
        position: tuple[float, float] | None = None,
    ) -> PipelineNode:
        node_id = node_id or self._next_id()
        if node_id in self._nodes:
            raise ValueError(f"Node id {node_id!r} already exists")
        node = PipelineNode(node_id, node_type, params or {}, position)
        self._nodes[node_id] = node
        self._forward.setdefault(node_id, set())
        self._reverse.setdefault(node_id, set())
        return node

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            return
        for source in list(self._reverse.get(node_id, set())):
            self.disconnect(source, node_id)
        for target in list(self._forward.get(node_id, set())):
            self.disconnect(node_id, target)
        self._nodes.pop(node_id, None)
        self._forward.pop(node_id, None)
        self._reverse.pop(node_id, None)

    def connect(self, source: str, target: str) -> None:
        if source == target:
            raise ValueError("Cannot connect a node to itself")
        if source not in self._nodes or target not in self._nodes:
            raise KeyError("Both nodes must be present before connecting")
        self._forward.setdefault(source, set()).add(target)
        self._reverse.setdefault(target, set()).add(source)
        self._ensure_acyclic()

    def disconnect(self, source: str, target: str) -> None:
        self._forward.get(source, set()).discard(target)
        self._reverse.get(target, set()).discard(source)

    # ---------------------------------------------------------------- traversal
    def ordered_nodes(self) -> list[PipelineNode]:
        order: list[str] = []
        indegree = {node_id: len(self._reverse.get(node_id, set())) for node_id in self._nodes}
        queue = [node_id for node_id, deg in indegree.items() if deg == 0]

        while queue:
            node_id = queue.pop(0)
            order.append(node_id)
            for neighbour in self._forward.get(node_id, set()):
                indegree[neighbour] -= 1
                if indegree[neighbour] == 0:
                    queue.append(neighbour)

        if len(order) != len(self._nodes):
            raise ValueError("Pipeline graph contains cycles")
        return [self._nodes[node_id] for node_id in order]

    def as_pipeline_definition(self) -> list[dict[str, Any]]:
        ordered = self.ordered_nodes()
        definition: list[dict[str, Any]] = []
        for node in ordered:
            definition.append(
                {
                    "id": node.id,
                    "type": node.type,
                    "params": node.params,
                    "inputs": sorted(self._reverse.get(node.id, set())),
                    "outputs": sorted(self._forward.get(node.id, set())),
                    "position": node.position,
                }
            )
        return definition

    def update_params(self, node_id: str, params: dict[str, Any]) -> None:
        if node_id not in self._nodes:
            raise KeyError(f"Unknown node id {node_id}")
        self._nodes[node_id].params = params

    def clear(self) -> None:
        self._nodes.clear()
        self._forward.clear()
        self._reverse.clear()
        self._counter = 0

    # ------------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        return {"nodes": self.as_pipeline_definition()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineGraph":
        graph = cls()
        for spec in payload.get("nodes", []):
            graph.add_node(spec["type"], node_id=spec.get("id"), params=spec.get("params"), position=spec.get("position"))
        for spec in payload.get("nodes", []):
            for successor in spec.get("outputs", []):
                graph.connect(spec["id"], successor)
        return graph

    # ---------------------------------------------------------------- Utilities
    def _next_id(self) -> str:
        self._counter += 1
        return f"node_{self._counter}"

    def _ensure_acyclic(self) -> None:
        self.ordered_nodes()  # Will raise if cycles present

    # Read-only views ---------------------------------------------------------
    def nodes(self) -> Iterable[PipelineNode]:
        return self._nodes.values()

    def successors(self, node_id: str) -> set[str]:
        return set(self._forward.get(node_id, set()))

    def predecessors(self, node_id: str) -> set[str]:
        return set(self._reverse.get(node_id, set()))
