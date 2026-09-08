"""
Image to Latent — loads a real image from disk, encodes it into SDXL's
latent space via the VAE encoder, and adds partial noise based on a
'strength' setting. Wire this into KSampler's `latents` input INSTEAD OF
Empty Latent Image, when you want to start from a real picture rather
than pure random noise.
"""

from __future__ import annotations

import asyncio
import os
import threading

import torch
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from PIL import Image
import numpy as np

from nodes.diffusion._model_choices import MODEL_CHOICES

_cache: dict = {}
_lock = threading.Lock()


def _get_vae(model_id: str):
    if model_id in _cache:
        return _cache[model_id]
    with _lock:
        if model_id in _cache:
            return _cache[model_id]

        repo_id = MODEL_CHOICES.get(model_id, model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32  # VAE stays float32, same reasoning as VAE Decode

        vae = AutoencoderKL.from_pretrained(repo_id, subfolder="vae", torch_dtype=dtype).to(device).eval()
        scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")

        _cache[model_id] = dict(vae=vae, device=device, scheduler=scheduler)
    return _cache[model_id]


metadata = {
    "display_name": "Image to Latent",
    "description": "Loads a real image and converts it into latents with partial noise, for image-to-image generation.",
    "category": "diffusion",
    "color": "purple",
}

inputs = [
    {"var_name": "model_id", "display_name": "Model ID", "type": "text"},
    {"var_name": "image_path", "display_name": "Image File Path", "type": "text"},
    {"var_name": "strength", "display_name": "Strength (0.0-1.0)", "type": "number"},
    {"var_name": "steps", "display_name": "Steps (MUST match KSampler's Steps)", "type": "number"},
]

outputs = [
    {"var_name": "latents", "display_name": "Latents", "type": "latent"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    model_id = inputs.get("model_id", "balanced_general_purpose")
    image_path = inputs.get("image_path", "").strip().strip('"').strip("'")
    strength = float(inputs.get("strength", 0.75))
    steps = int(inputs.get("steps", 30))

    if not image_path:
        raise ValueError("image_path is required")
    print(f"DEBUG - repr of image_path: {repr(image_path)}")
    if not os.path.exists(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    if not (0.0 < strength <= 1.0):
        raise ValueError("strength must be between 0.0 and 1.0")

    models = await asyncio.to_thread(_get_vae, model_id)
    vae = models["vae"]
    device = models["device"]
    scheduler = models["scheduler"]
    dtype = torch.float32

    def _process():
        image = Image.open(image_path).convert("RGB")
        image = image.resize((1024, 1024))

        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = (image_array * 2.0) - 1.0  # scale to [-1, 1], matching VAE's expected range
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            latents = vae.encode(image_tensor).latent_dist.sample()

        scaling_factor = getattr(vae.config, "scaling_factor", 0.13025)
        latents = latents * scaling_factor

        # Add partial noise based on strength — higher strength = more noise = more creative freedom
        scheduler.set_timesteps(steps)
        step_index = int((1.0 - strength) * len(scheduler.timesteps))
        step_index = max(0, min(step_index, len(scheduler.timesteps) - 1))
        start_timestep = scheduler.timesteps[step_index]

        noise = torch.randn_like(latents)
        noised_latents = scheduler.add_noise(latents, noise, start_timestep.unsqueeze(0))

        return noised_latents

    latents = await asyncio.to_thread(_process)

    import tempfile
    import uuid as uuid_module
    from nodes.diffusion._project_folder import get_project_folder

    folder = get_project_folder()
    file_path = os.path.join(folder, f"img2latent_{uuid_module.uuid4().hex}.pt")
    torch.save(latents.cpu(), file_path)

    return {"latents": file_path}