"""
FastAPI route definitions for the GenVR Workflow Engine.

Routes
------
GET  /nodes
    List all registered nodes with their metadata, inputs, and outputs.

GET  /nodes/{node_type}
    Get metadata for a single node type.

POST /nodes/{node_type}/execute
    Execute a single node directly.
    Body: { uid, token, inputs: {var_name: value, ...} }

POST /workflow/execute
    Execute a full workflow graph.
    Body: { uid, token, nodes: [...], edges: [...] }
"""

import traceback

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any

from comfyui_custom_nodes.comfy_debug import comfy_log
from comfyui_custom_nodes.workflow_bridge import is_comfy_node_type

from nodes.registry import list_nodes, get_node
from engine.executor import execute_workflow
from engine.node_runner import run_node

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────────────────

class NodeExecuteRequest(BaseModel):
    uid: str
    token: str
    inputs: dict[str, Any] = {}


class WorkflowNode(BaseModel):
    id: str
    type: str
    data: dict[str, Any] = {}


class WorkflowEdge(BaseModel):
    source: str
    target: str
    sourceHandle: str | None = None
    targetHandle: str | None = None


class WorkflowExecuteRequest(BaseModel):
    uid: str
    token: str
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge] = []


# ── Node routes ───────────────────────────────────────────────────────────────

@router.get("/nodes", summary="List all registered nodes")
def list_all_nodes():
    return {"nodes": list_nodes()}


@router.get("/nodes/{node_type:path}", summary="Get a single node's definition")
def get_node_definition(node_type: str):
    module = get_node(node_type)
    if module is None:
        raise HTTPException(status_code=404, detail=f"Node type '{node_type}' not found")
    return {
        "node_type": node_type,
        "metadata": module.metadata,
        "inputs": module.inputs,
        "outputs": module.outputs,
    }


@router.post("/nodes/{node_type:path}/execute", summary="Execute a single node")
async def execute_node(node_type: str, body: NodeExecuteRequest):
    module = get_node(node_type)
    if module is None:
        raise HTTPException(status_code=404, detail=f"Node type '{node_type}' not found")

    if is_comfy_node_type(node_type):
        comfy_log(node_type.split(".")[-1], "api.execute.request", {
            "node_type": node_type,
            "inputs": body.inputs,
        })

    try:
        outputs = await run_node(module, body.uid, body.token, body.inputs)
    except ValueError as exc:
        if is_comfy_node_type(node_type):
            comfy_log(node_type.split(".")[-1], "api.execute.422", {
                "detail": str(exc),
                "traceback": traceback.format_exc(),
            })
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        if is_comfy_node_type(node_type):
            comfy_log(node_type.split(".")[-1], "api.execute.500", {
                "detail": str(exc),
                "traceback": traceback.format_exc(),
            })
        raise HTTPException(status_code=500, detail=str(exc))

    return {"outputs": outputs}


# ── Workflow route ─────────────────────────────────────────────────────────────

@router.post("/workflow/execute", summary="Execute a full workflow graph")
async def execute_workflow_route(body: WorkflowExecuteRequest):
    raw_nodes = [n.model_dump() for n in body.nodes]
    raw_edges = [e.model_dump() for e in body.edges]

    try:
        result = await execute_workflow(body.uid, body.token, raw_nodes, raw_edges)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "node_outputs": result.node_outputs,
        "log": result.log,
    }
