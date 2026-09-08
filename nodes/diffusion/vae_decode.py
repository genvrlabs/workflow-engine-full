"""
VAE Decode node — uses either the checkpoint's built-in VAE or an override.
"""

from __future__ import annotations

import asyncio
import os
import uuid
import threading
from datetime import datetime

import torch
from diffusers import AutoencoderKL
from PIL import Image

from nodes.diffusion._model_choices import MODEL_CHOICES, VAE_CHOICES
from nodes.diffusion._project_folder import get_project_folder
from nodes.diffusion._models_folder import require_model_path

_cache: dict = {}
_lock = threading.Lock()


def _get_vae(model_id: str, vae_id: str):
    cache_key = f"{model_id}::{vae_id}"
    if cache_key in _cache:
        return _cache[cache_key]
    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        override_repo = VAE_CHOICES.get(vae_id, vae_id if "/" in (vae_id or "") else None)

        if override_repo:
            override_repo = require_model_path(override_repo)
            vae = AutoencoderKL.from_pretrained(override_repo, torch_dtype=dtype).to(device).eval()
        else:
            repo_id = MODEL_CHOICES.get(model_id, model_id)
            repo_id = require_model_path(repo_id)
            vae = AutoencoderKL.from_pretrained(
                repo_id, subfolder="vae", torch_dtype=dtype,
            ).to(device).eval()

        _cache[cache_key] = dict(vae=vae, device=device)
    return _cache[cache_key]


metadata = {
    "display_name": "VAE Decode",
    "description": "Decodes SDXL latents into a final image.",
    "category": "diffusion",
    "color": "purple",
}

inputs = [
    {"var_name": "model_id", "display_name": "Model ID", "type": "text"},
    {"var_name": "vae_id", "display_name": "VAE ID (default = checkpoint's VAE)", "type": "text"},
    {"var_name": "latents", "display_name": "Latents", "type": "latent"},
]
outputs = [{"var_name": "image_url", "display_name": "Image URL", "type": "text"}]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    model_id = inputs.get("model_id", "balanced_general_purpose")
    vae_id = inputs.get("vae_id", "default")

    models = await asyncio.to_thread(_get_vae, model_id, vae_id)
    vae = models["vae"]
    device = models["device"]
    dtype = torch.float32

    latents = torch.load(inputs["latents"], map_location=device).to(dtype=dtype)

    scaling_factor = getattr(vae.config, "scaling_factor", 0.13025)
    latents = latents / scaling_factor

    with torch.no_grad():
        decoded = vae.decode(latents).sample

    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
    image_array = (decoded[0] * 255).round().astype("uint8")
    image = Image.fromarray(image_array)

    folder = get_project_folder()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    image_file_name = f"generated_{timestamp}_{uuid.uuid4().hex[:8]}.png"
    image_path = os.path.join(folder, image_file_name)
    image.save(image_path)

    return {"image_url": image_path}
