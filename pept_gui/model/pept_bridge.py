"""Bridge between pipeline graphs and the PEPT pipeline implementation."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Dict, Iterable

try:  # pragma: no cover - optional dependency
    import pept
except ImportError:  # pragma: no cover - used in tests without dependency
    pept = None  # type: ignore[assignment]

TRANSFORMER_PATHS: dict[str, tuple[str, ...]] = {
    "BirminghamMethod": ("pept.tracking.BirminghamMethod",),
    "Condition": ("pept.tracking.Condition", "pept.utilities.Condition"),
    "Stack": ("pept.tracking.Stack", "pept.reducers.Stack"),
    "Cutpoints": ("pept.tracking.Cutpoints",),
    "HDBSCAN": ("pept.tracking.HDBSCAN",),
    "SplitLabels": ("pept.tracking.SplitLabels", "pept.reducers.SplitLabels"),
    "Centroids": ("pept.tracking.Centroids",),
    "Segregate": ("pept.tracking.Segregate",),
}


REDUCER_NAMES: set[str] = {
    "Stack",
    "SplitLabels",
    "GroupBy",
    "SplitAll",
}


def _resolve_transformers() -> Dict[str, Callable[..., Any]]:
    registry: Dict[str, Callable[..., Any]] = {}
    for name, candidates in TRANSFORMER_PATHS.items():
        for dotted_path in candidates:
            module_path, attr = dotted_path.rsplit(".", 1)
            try:
                module = import_module(module_path)
                transformer = getattr(module, attr)
            except Exception:  # pragma: no cover - missing optional transformer
                continue
            registry[name] = transformer
            break
    return registry


TRANSFORMERS: Dict[str, Callable[..., Any]] = _resolve_transformers()


def resolve_transformer(name: str) -> Callable[..., Any] | None:
    """Return the transformer callable for ``name``, attempting lazy import."""

    transformer = TRANSFORMERS.get(name)
    if transformer is not None:
        return transformer

    candidates = TRANSFORMER_PATHS.get(name, ())
    for dotted_path in candidates:
        module_path, attr = dotted_path.rsplit(".", 1)
        try:
            module = import_module(module_path)
            transformer = getattr(module, attr)
        except Exception:  # pragma: no cover - optional dependency failure
            continue
        TRANSFORMERS[name] = transformer
        return transformer
    return None


def compile_pipeline(nodes: Iterable[dict[str, Any]]) -> "pept.Pipeline":
    """Compile ``nodes`` into a :class:`pept.Pipeline`.``"""
    if pept is None:  # pragma: no cover - triggered when dependency missing
        raise RuntimeError("The pept package is required to compile pipelines")

    steps = []
    reducers_present = False
    for node in nodes:
        node_type = node.get("type")
        params = dict(node.get("params") or {})
        args = params.pop("__args__", None)
        if isinstance(args, (list, tuple)):
            positional = list(args)
        elif args is None:
            positional = []
        else:
            positional = [args]
        transformer = TRANSFORMERS.get(node_type) or resolve_transformer(node_type)
        if transformer is None:
            raise KeyError(f"Unknown transformer {node_type!r}")
        instance = transformer(*positional, **params)
        if node_type in REDUCER_NAMES or instance.__class__.__module__.startswith("pept.reducers"):
            reducers_present = True
        steps.append(instance)

    if steps and not reducers_present:
        stack_cls = TRANSFORMERS.get("Stack")
        if stack_cls is not None:
            steps.append(stack_cls())

    return pept.Pipeline(steps)


def available_transformers() -> Dict[str, Callable[..., Any]]:
    """Return the registry of resolved transformer callables."""
    return dict(TRANSFORMERS)


def registered_transformer_names() -> list[str]:
    """Return all declared transformer names, even if not resolved."""
    return sorted(TRANSFORMER_PATHS.keys())
