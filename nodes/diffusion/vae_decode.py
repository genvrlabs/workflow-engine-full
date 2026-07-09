"""
VAE Decode node for SDXL.

Converts denoised latents into a final PNG image.
"""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Any, Dict

import torch
from diffusers import AutoencoderKL
from PIL import Image

from nodes.ffmpeg._utils import download_to_tempfile, upload_file

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
_cache = {}
_cache_lock = threading.Lock()


def _get_vae() -> Dict[str, Any]:
    if _cache:
        return _cache

    with _cache_lock:
        if _cache:
            return _cache

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        vae = AutoencoderKL.from_pretrained(
            MODEL_ID,
            subfolder="vae",
            torch_dtype=dtype,
        ).to(device).eval()

        _cache.update(vae=vae, device=device)

    return _cache


metadata = {
    "display_name": "VAE Decode",
    "description": "Decodes SDXL latents into a final image.",
    "category": "diffusion",
    "color": "purple",
}

inputs = [
    {
        "var_name": "latents",
        "display_name": "Latents",
        "type": "latent",
    },
]

outputs = [
    {
        "var_name": "image_url",
        "display_name": "Image URL",
        "type": "text",
    },
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    import asyncio

    models = await asyncio.to_thread(_get_vae)
    vae = models["vae"]
    device = models["device"]
    dtype = torch.float16 if device == "cuda" else torch.float32

    latents_url = inputs["latents"]
    latents_path = download_to_tempfile(latents_url, suffix=".pt")

    try:
        latents = torch.load(latents_path, map_location=device).to(dtype=dtype)
    finally:
        os.unlink(latents_path)

    # SDXL VAE scaling factor
    scaling_factor = getattr(vae.config, "scaling_factor", 0.13025)
    latents = latents / scaling_factor

    with torch.no_grad():
        decoded = vae.decode(latents).sample

    # Convert tensor [-1, 1] -> PIL image
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
    image_array = (decoded[0] * 255).round().astype("uint8")
    image = Image.fromarray(image_array)

    image_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    image_tmp.close()
    image.save(image_tmp.name)
    image_url = upload_file(uid, token, image_tmp.name)
    os.unlink(image_tmp.name)

    return {
        "image_url": image_url,
    }