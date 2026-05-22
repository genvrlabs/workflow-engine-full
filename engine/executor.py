"""
Workflow executor — resolves a graph of nodes in topological order and
executes each node using its registered execute() function.

Graph format mirrors the GenVR workflow designer (ReactFlow compatible):
  nodes: [{"id": str, "type": str, "data": dict}, ...]
  edges: [{"source": str, "target": str, "sourceHandle": str, "targetHandle": str}, ...]

For each node, inputs are resolved from:
  1. The node's own `data` dict (static / default values)
  2. Outputs of upstream nodes, wired via edges

Edge wiring:
  - sourceHandle → the output var_name on the source node
  - targetHandle → the input var_name on the target node
  If handles are omitted, the first output/input var is used.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import Any

from nodes.registry import get_node


# ── Types ────────────────────────────────────────────────────────────────────

class NodeSpec:
    __slots__ = ("id", "type", "data")

    def __init__(self, id: str, type: str, data: dict):
        self.id = id
        self.type = type
        self.data = data or {}


class EdgeSpec:
    __slots__ = ("source", "target", "source_handle", "target_handle")

    def __init__(self, source: str, target: str, source_handle: str | None, target_handle: str | None):
        self.source = source
        self.target = target
        self.source_handle = source_handle
        self.target_handle = target_handle


class ExecutionResult:
    def __init__(self):
        # node_id → {var_name: value}
        self.node_outputs: dict[str, dict[str, Any]] = {}
        self.log: list[dict] = []

    def record(self, node_id: str, outputs: dict, status: str = "done", error: str | None = None):
        self.node_outputs[node_id] = outputs
        entry: dict[str, Any] = {"node_id": node_id, "status": status}
        if error:
            entry["error"] = error
        self.log.append(entry)

    @property
    def final_outputs(self) -> dict[str, Any]:
        """Return the merged outputs dict of all output_node nodes (or last node)."""
        return self.node_outputs


# ── Topological sort ─────────────────────────────────────────────────────────

def _topological_order(nodes: list[NodeSpec], edges: list[EdgeSpec]) -> list[str]:
    in_degree: dict[str, int] = {n.id: 0 for n in nodes}
    adj: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        adj[edge.source].append(edge.target)
        in_degree[edge.target] += 1

    queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
    order: list[str] = []

    while queue:
        nid = queue.popleft()
        order.append(nid)
        for child in adj[nid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(nodes):
        raise ValueError("Workflow graph contains a cycle — cannot execute.")

    return order


# ── Input resolution ─────────────────────────────────────────────────────────

def _resolve_inputs(
    node: NodeSpec,
    edges: list[EdgeSpec],
    completed: dict[str, dict[str, Any]],
    node_module,
) -> dict[str, Any]:
    """Merge static data with upstream edge outputs into a single inputs dict."""
    resolved: dict[str, Any] = dict(node.data)

    # Determine first output var of each source (fallback when handle is absent)
    for edge in edges:
        if edge.target != node.id:
            continue

        source_outputs = completed.get(edge.source, {})

        # Which output var from the source?
        if edge.source_handle:
            value = source_outputs.get(edge.source_handle)
        else:
            value = next(iter(source_outputs.values()), None) if source_outputs else None

        # Which input var on this node?
        if edge.target_handle:
            resolved[edge.target_handle] = value
        else:
            # Fall back to first declared input
            declared_inputs = getattr(node_module, "inputs", [])
            if declared_inputs:
                first_input = declared_inputs[0]
                key = first_input["var_name"] if isinstance(first_input, dict) else first_input.var_name
                resolved[key] = value

    return resolved


# ── Main executor ─────────────────────────────────────────────────────────────

async def execute_workflow(
    uid: str,
    token: str,
    raw_nodes: list[dict],
    raw_edges: list[dict],
) -> ExecutionResult:
    nodes = [NodeSpec(n["id"], n["type"], n.get("data", {})) for n in raw_nodes]
    edges = [
        EdgeSpec(
            e["source"],
            e["target"],
            e.get("sourceHandle"),
            e.get("targetHandle"),
        )
        for e in raw_edges
    ]

    order = _topological_order(nodes, edges)
    node_by_id = {n.id: n for n in nodes}
    completed: dict[str, dict[str, Any]] = {}
    result = ExecutionResult()

    for node_id in order:
        node = node_by_id[node_id]
        module = get_node(node.type)

        if module is None:
            error_msg = f"Unknown node type: '{node.type}'"
            result.record(node_id, {}, status="error", error=error_msg)
            raise ValueError(error_msg)

        resolved_inputs = _resolve_inputs(node, edges, completed, module)

        try:
            outputs = await module.execute(uid, token, resolved_inputs)
        except Exception as exc:
            error_msg = str(exc)
            result.record(node_id, {}, status="error", error=error_msg)
            raise RuntimeError(f"Node '{node_id}' ({node.type}) failed: {error_msg}") from exc

        if not isinstance(outputs, dict):
            outputs = {"value": outputs}

        completed[node_id] = outputs
        result.record(node_id, outputs, status="done")

    return result
