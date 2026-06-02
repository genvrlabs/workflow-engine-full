"""HTTP API for the ComfyUI custom-node installer."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from comfyui_custom_nodes.config import CUSTOM_NODES_DIR, INSTALLED_DIR, PROJECT_ROOT
from comfyui_custom_nodes.installer import install_from_source, list_installed, uninstall

router = APIRouter(prefix="/comfyui", tags=["ComfyUI Custom Nodes"])


class InstallRequest(BaseModel):
    source: str = Field(
        ...,
        description="GitHub URL (repo, tree, blob, raw) or local file/directory path",
        examples=[
            "https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet/blob/master/PoseNode/pose_node.py"
        ],
    )
    branch: str | None = Field(None, description="Override git branch (default: from URL or master)")


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": "ComfyUI Custom Node Installer",
        "installed_dir": str(INSTALLED_DIR.resolve()),
        "custom_nodes_dir": str(CUSTOM_NODES_DIR.resolve()),
        "project_root": str(PROJECT_ROOT),
    }


@router.get("/installed")
def get_installed():
    return {"installed": list_installed()}


@router.post("/install")
def install_node(body: InstallRequest):
    try:
        result = install_from_source(body.source, branch=body.branch)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result.to_dict()


@router.delete("/installed/{slug}")
def remove_installed(slug: str):
    try:
        return uninstall(slug)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
