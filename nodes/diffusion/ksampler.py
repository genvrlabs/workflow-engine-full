"""
KSampler node for SDXL.
"""

from __future__ import annotations

import asyncio
import threading

from typing import Any, Dict

import torch
import os
import tempfile
from nodes.ffmpeg._utils import download_to_tempfile
from nodes.ffmpeg._utils import download_to_tempfile, upload_file
from diffusers import EulerDiscreteScheduler, UNet2DConditionModel

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
_cache = {}
_cache_lock = threading.Lock()

def _get_models() -> Dict[str, Any]:
    """Load (or retrieve cached) UNet + scheduler."""

    if _cache:
        return _cache

    with _cache_lock:
        if _cache:
            return _cache

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        unet = UNet2DConditionModel.from_pretrained(
            MODEL_ID,
            subfolder="unet",
            torch_dtype=dtype,
        ).to(device).eval()

        scheduler = EulerDiscreteScheduler.from_pretrained(
            MODEL_ID,
            subfolder="scheduler",
        )

        _cache.update(
            unet=unet,
            scheduler=scheduler,
            device=device,
        )

    return _cache

def _build_added_cond_kwargs(
    pooled_embeds: torch.Tensor,
    height: int,
    width: int,
    device: str,
    dtype: torch.dtype,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Construct SDXL's `added_cond_kwargs` (pooled text embeds + time ids)."""
    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]],
        dtype=dtype,
        device=device,
    ).repeat(batch_size, 1)

    return {
        "text_embeds": pooled_embeds.to(device=device, dtype=dtype),
        "time_ids": add_time_ids,
    }

metadata = {
    "display_name": "KSampler",
    "description": "Performs SDXL denoising.",
    "category": "diffusion",
    "color": "purple",
}

inputs = [
    {
        "var_name": "embeds",
        "display_name": "Embeds",
        "type": "text",
    },
    {
        "var_name": "pooled_embeds",
        "display_name": "Pooled Embeds",
        "type": "text",
    },
    {
        "var_name": "latents",
        "display_name": "Latents",
        "type": "latent",
    },
    {
        "var_name": "steps",
        "display_name": "Steps",
        "type": "number",
    },
    {
        "var_name": "cfg_scale",
        "display_name": "CFG Scale",
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

    models = await asyncio.to_thread(_get_models)

    unet = models["unet"]
    scheduler = models["scheduler"]
    device = models["device"]

    dtype = torch.float16 if device == "cuda" else torch.float32

    # Read inputs from previous nodes — these arrive as URLs, not tensors
    embeds_url = inputs["embeds"]
    pooled_url = inputs["pooled_embeds"]
    latents_url = inputs["latents"]

    embeds_path = download_to_tempfile(embeds_url, suffix=".pt")
    pooled_path = download_to_tempfile(pooled_url, suffix=".pt")
    latents_path = download_to_tempfile(latents_url, suffix=".pt")

    try:
        embeds = torch.load(embeds_path, map_location=device).to(dtype=dtype)
        pooled_embeds = torch.load(pooled_path, map_location=device).to(dtype=dtype)
        latents = torch.load(latents_path, map_location=device).to(dtype=dtype)
    finally:
        os.unlink(embeds_path)
        os.unlink(pooled_path)
        os.unlink(latents_path)

    batch_size = latents.shape[0]
    height = latents.shape[-2] * 8
    width = latents.shape[-1] * 8

    # Build added conditioning kwargs
    added_cond_kwargs = _build_added_cond_kwargs(
        pooled_embeds=pooled_embeds,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )

    steps = int(inputs.get("steps", 30))
    cfg_scale = float(inputs.get("cfg_scale", 7))

    scheduler.set_timesteps(steps, device=device)

    with torch.no_grad():
        for t in scheduler.timesteps:
            noise_pred = unet(
                latents,
                t,
                encoder_hidden_states=embeds,
                added_cond_kwargs=added_cond_kwargs,
            ).sample

            latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    latents_tmp.close()
    torch.save(latents.cpu(), latents_tmp.name)
    latents_url = upload_file(uid, token, latents_tmp.name)
    os.unlink(latents_tmp.name)

    return {
        "latents": latents_url,
    }