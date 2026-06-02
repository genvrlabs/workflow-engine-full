"""Paths and settings for the ComfyUI custom-node installer."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMFYUI_ROOT = PROJECT_ROOT / "comfyui"
INSTALLED_DIR = COMFYUI_ROOT / "installed"
CUSTOM_NODES_DIR = COMFYUI_ROOT / "custom_nodes"

DEFAULT_INSTALLER_PORT = 8001
