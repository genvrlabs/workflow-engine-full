"""
Node registry — auto-discovers all node modules under nodes/*/
and exposes them by their node_type key.

A node module is any Python file (not __init__.py, base.py, registry.py) that
exports `inputs`, `outputs`, `metadata`, and `execute`.
"""

import importlib
import pkgutil
import sys
from pathlib import Path
from typing import Any


def _load_node_modules() -> dict[str, Any]:
    registry: dict[str, Any] = {}
    nodes_dir = Path(__file__).parent

    for finder, pkg_name, is_pkg in pkgutil.walk_packages(
        path=[str(nodes_dir)],
        prefix="nodes.",
        onerror=lambda name: None,
    ):
        # Skip internal modules
        leaf = pkg_name.split(".")[-1]
        if leaf in ("__init__", "base", "registry"):
            continue

        try:
            module = importlib.import_module(pkg_name)
        except Exception as exc:
            print(f"[registry] failed to import {pkg_name}: {exc}")
            continue

        # Validate required exports
        required = ("inputs", "outputs", "metadata", "execute")
        if not all(hasattr(module, attr) for attr in required):
            continue

        # node_type is the dotted path under nodes, e.g. arithmetic.add_two
        node_type = pkg_name.removeprefix("nodes.")
        registry[node_type] = module

    return registry


# Module-level singleton — loaded once on first import
registry: dict[str, Any] = _load_node_modules()


def get_node(node_type: str):
    """Return the node module for the given type, or None."""
    return registry.get(node_type)


def list_nodes() -> list[dict]:
    """Return serialisable metadata for all registered nodes."""
    result = []
    for node_type, module in registry.items():
        result.append({
            "node_type": node_type,
            "metadata": module.metadata,
            "inputs": [p if isinstance(p, dict) else p.to_dict() for p in module.inputs],
            "outputs": [p if isinstance(p, dict) else p.to_dict() for p in module.outputs],
        })
    return result
