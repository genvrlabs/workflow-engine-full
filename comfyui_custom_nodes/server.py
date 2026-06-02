"""
ComfyUI custom-node installer — standalone FastAPI app.

Run:
    uvicorn comfyui_custom_nodes.server:app --host 0.0.0.0 --port 8001
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from comfyui_custom_nodes.routes import router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="GenVR ComfyUI Custom Node Installer",
    description=(
        "Install ComfyUI custom nodes from GitHub URLs or local paths. "
        "Packages are stored under comfyui/installed and linked into comfyui/custom_nodes. "
        "ComfyUI loads custom_nodes on startup via NODE_CLASS_MAPPINGS in each package __init__.py."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def ui():
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {
        "status": "ok",
        "service": "ComfyUI Custom Node Installer",
        "docs": "/docs",
    }


@app.get("/api", include_in_schema=False)
def api_info():
    return {
        "status": "ok",
        "service": "ComfyUI Custom Node Installer",
        "ui": "/",
        "docs": "/docs",
    }
