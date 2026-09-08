"""
Empty Latent Image node for SDXL.

Creates a random latent tensor that serves as the starting point for
the diffusion process.
"""

from __future__ import annotations

import os
import uuid
import torch
from nodes.diffusion._project_folder import get_project_folder

metadata = {
    "display_name": "Empty Latent Image",
    "description": "Create an empty latent tensor for SDXL image generation.",
    "category": "diffusion",
    "color": "purple",
}

inputs = [
    {
        "var_name": "width",
        "display_name": "Width",
        "type": "number",
    },
    {
        "var_name": "height",
        "display_name": "Height",
        "type": "number",
    },
    {
        "var_name": "batch_size",
        "display_name": "Batch Size",
        "type": "number",
    },
]

outputs = [
    {
        "var_name": "latents",
        "display_name": "Latents",
        "type": "latent",
    }
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    width = int(inputs.get("width", 1024))
    height = int(inputs.get("height", 1024))
    batch_size = int(inputs.get("batch_size", 1))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    latents = torch.randn(
        batch_size,
        4,
        height // 8,
        width // 8,
        device=device,
        dtype=dtype,
    )

    folder = get_project_folder()
    file_name = f"latents_{uuid.uuid4().hex}.pt"
    file_path = os.path.join(folder, file_name)
    torch.save(latents.cpu(), file_path)

    return {"latents": file_path}