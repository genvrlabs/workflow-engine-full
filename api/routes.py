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
import json
import traceback
import os
import requests
import uuid
from nodes.diffusion._project_folder import get_project_folder
from api.log_buffer import log_buffer



from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any
from nodes.diffusion._pick_image_file import pick_image_file
from nodes.diffusion.genvr_api import get_available_models, get_model_schema

from comfyui_custom_nodes.comfy_debug import comfy_log
from comfyui_custom_nodes.workflow_bridge import is_comfy_node_type

from nodes.registry import list_nodes, get_node
from engine.executor import execute_workflow
from services.workflow_agent import generate_workflow_text
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

@router.get("/local-file", summary="Serve a local file (image/tensor) so the browser can display it")
def get_local_file(path: str):
    import os
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    return FileResponse(path)


@router.get("/pick-image", summary="Open a native file picker for selecting an image")
def get_pick_image():
    path = pick_image_file()
    return {"path": path}



@router.get("/genvr-models", summary="Get the list of available GenVR models")
def get_genvr_models(api_key: str):
    return {"models": get_available_models(api_key)}

@router.get("/genvr-schema", summary="Get parameter schema for a specific GenVR model")
def get_genvr_schema(category: str, subcategory: str):
    return {"schema": get_model_schema(category, subcategory)}

@router.post("/upload-image", summary="Upload an image, save it locally, return its path")
async def upload_image(file: UploadFile = File(...)):
    folder = get_project_folder()
    ext = os.path.splitext(file.filename or "")[1] or ".png"
    file_path = os.path.join(folder, f"upload_{uuid.uuid4().hex}{ext}")

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    return {"path": file_path}

WORKFLOWS_DIR_NAME = "saved_workflows"


def _workflows_dir() -> str:
    folder = os.path.join(get_project_folder(), WORKFLOWS_DIR_NAME)
    os.makedirs(folder, exist_ok=True)
    return folder


class SaveWorkflowRequest(BaseModel):
    name: str
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge] = []
    positions: dict[str, dict[str, float]] = {}


@router.post("/workflows", summary="Save a workflow to disk")
def save_workflow(body: SaveWorkflowRequest):
    safe_name = "".join(c for c in body.name if c.isalnum() or c in (" ", "-", "_")).strip() or "untitled"
    file_path = os.path.join(_workflows_dir(), f"{safe_name}.json")
    with open(file_path, "w") as f:
        json.dump(body.model_dump(), f, indent=2)
    return {"name": safe_name}
class GenerateWorkflowRequest(BaseModel):
    prompt: str
    uid: str
    token: str
    model: str


@router.post("/agent/generate-workflow", summary="Generate a workflow from a plain-English request")
def generate_workflow_route(body: GenerateWorkflowRequest):
    try:
        raw_text = generate_workflow_text(body.prompt, body.uid, body.token, body.model)
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}")

    try:
        workflow_json = json.loads(raw_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail=f"LLM did not return valid JSON: {raw_text[:300]}")

    return workflow_json


@router.get("/workflows", summary="List saved workflows")
def list_workflows():
    folder = _workflows_dir()
    names = [os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith(".json")]
    return {"workflows": sorted(names)}


@router.get("/workflows/{name}", summary="Load a saved workflow")
def load_workflow(name: str):
    file_path = os.path.join(_workflows_dir(), f"{name}.json")
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")
    with open(file_path) as f:
        return json.load(f)


@router.delete("/workflows/{name}", summary="Delete a saved workflow")
def delete_workflow(name: str):
    file_path = os.path.join(_workflows_dir(), f"{name}.json")
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")
    os.remove(file_path)
    return {"deleted": name}

@router.get("/assets", summary="List generated image files in the project folder")
def list_assets():
    folder = get_project_folder()
    exts = (".png", ".jpg", ".jpeg", ".webp", ".gif")
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))
    ]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
    return {"assets": [{"name": f, "path": os.path.join(folder, f)} for f in files]}


@router.get("/logs", summary="Get recent backend log lines")
def get_logs():
    return {"logs": list(log_buffer)}

@router.delete("/logs", summary="Clear the backend log buffer")
def clear_logs():
    log_buffer.clear()
    return {"cleared": True}

@router.delete("/logs", summary="Clear the backend log buffer")
def clear_logs():
    log_buffer.clear()
    return {"cleared": True}