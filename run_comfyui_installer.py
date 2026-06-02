"""
Start the ComfyUI custom-node installer API (default port 8001).

    python run_comfyui_installer.py
    python run_comfyui_installer.py --port 8002
"""

import argparse

import uvicorn

from comfyui_custom_nodes.config import DEFAULT_INSTALLER_PORT


def main():
    parser = argparse.ArgumentParser(description="GenVR ComfyUI custom-node installer")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_INSTALLER_PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "comfyui_custom_nodes.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
