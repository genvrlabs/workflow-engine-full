"""
KSampler node for SDXL — supports choosing model and sampler algorithm.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading

import torch
from diffusers import (
    UNet2DConditionModel,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
)

from nodes.ffmpeg._utils import download_to_tempfile, upload_file
from nodes.diffusion._model_choices import MODEL_CHOICES

SAMPLER_CHOICES = {
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "ddim": DDIMScheduler,
    "dpmpp_2m": DPMSolverMultistepScheduler,
}

_unet_cache: dict = {}
_sched_config_cache: dict = {}
_lock = threading.Lock()


def _get_unet(model_id: str):
    if model_id in _unet_cache:
        return _unet_cache[model_id]
    with _lock:
        if model_id in _unet_cache:
            return _unet_cache[model_id]
        repo_id = MODEL_CHOICES.get(model_id, model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        unet = UNet2DConditionModel.from_pretrained(
            repo_id, subfolder="unet", torch_dtype=dtype,
        ).to(device).eval()
        base_scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
        _unet_cache[model_id] = dict(unet=unet, device=device)
        _sched_config_cache[model_id] = base_scheduler.config
    return _unet_cache[model_id]


def _build_scheduler(model_id: str, sampler_name: str):
    config = _sched_config_cache[model_id]
    scheduler_cls = SAMPLER_CHOICES.get(sampler_name, EulerDiscreteScheduler)
    return scheduler_cls.from_config(config)


def _build_added_cond_kwargs(pooled_embeds, height, width, device, dtype, batch_size):
    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], dtype=dtype, device=device,
    ).repeat(batch_size, 1)
    return {"text_embeds": pooled_embeds.to(device=device, dtype=dtype), "time_ids": add_time_ids}


metadata = {
    "display_name": "KSampler",
    "description": "Performs SDXL denoising with CFG and a choice of sampler.",
    "category": "diffusion",
    "color": "purple",
}

inputs = [
    {"var_name": "model_id", "display_name": "Model ID", "type": "text"},
    {"var_name": "embeds", "display_name": "Embeds", "type": "text"},
    {"var_name": "pooled_embeds", "display_name": "Pooled Embeds", "type": "text"},
    {"var_name": "negative_embeds", "display_name": "Negative Embeds", "type": "text"},
    {"var_name": "negative_pooled_embeds", "display_name": "Negative Pooled Embeds", "type": "text"},
    {"var_name": "latents", "display_name": "Latents", "type": "latent"},
    {"var_name": "steps", "display_name": "Steps", "type": "number"},
    {"var_name": "cfg_scale", "display_name": "CFG Scale", "type": "number"},
    {
        "var_name": "sampler",
        "display_name": "Sampler",
        "type": "dropdown",
        "options": list(SAMPLER_CHOICES.keys()),
    },
]

outputs = [{"var_name": "latents", "display_name": "Latents", "type": "latent"}]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    model_id = inputs.get("model_id", "balanced_general_purpose")
    sampler_name = (inputs.get("sampler") or "euler").strip()

    models = await asyncio.to_thread(_get_unet, model_id)
    unet = models["unet"]
    device = models["device"]
    dtype = torch.float16 if device == "cuda" else torch.float32
    scheduler = _build_scheduler(model_id, sampler_name)

    embeds_path = download_to_tempfile(inputs["embeds"], suffix=".pt")
    pooled_path = download_to_tempfile(inputs["pooled_embeds"], suffix=".pt")
    neg_embeds_path = download_to_tempfile(inputs["negative_embeds"], suffix=".pt")
    neg_pooled_path = download_to_tempfile(inputs["negative_pooled_embeds"], suffix=".pt")
    latents_path = download_to_tempfile(inputs["latents"], suffix=".pt")

    try:
        embeds = torch.load(embeds_path, map_location=device).to(dtype=dtype)
        pooled_embeds = torch.load(pooled_path, map_location=device).to(dtype=dtype)
        neg_embeds = torch.load(neg_embeds_path, map_location=device).to(dtype=dtype)
        neg_pooled_embeds = torch.load(neg_pooled_path, map_location=device).to(dtype=dtype)
        latents = torch.load(latents_path, map_location=device).to(dtype=dtype)
    finally:
        os.unlink(embeds_path)
        os.unlink(pooled_path)
        os.unlink(neg_embeds_path)
        os.unlink(neg_pooled_path)
        os.unlink(latents_path)

    batch_size = latents.shape[0]
    height = latents.shape[-2] * 8
    width = latents.shape[-1] * 8

    cond_kwargs_cond = _build_added_cond_kwargs(pooled_embeds, height, width, device, dtype, batch_size)
    cond_kwargs_uncond = _build_added_cond_kwargs(neg_pooled_embeds, height, width, device, dtype, batch_size)

    steps = int(inputs.get("steps", 30))
    cfg_scale = float(inputs.get("cfg_scale", 7))

    scheduler.set_timesteps(steps, device=device)
    latents = latents * scheduler.init_noise_sigma

    with torch.no_grad():
        for t in scheduler.timesteps:
            scaled_latents = scheduler.scale_model_input(latents, t)

            noise_pred_cond = unet(
                scaled_latents, t, encoder_hidden_states=embeds, added_cond_kwargs=cond_kwargs_cond,
            ).sample
            noise_pred_uncond = unet(
                scaled_latents, t, encoder_hidden_states=neg_embeds, added_cond_kwargs=cond_kwargs_uncond,
            ).sample

            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    latents_tmp.close()
    torch.save(latents.cpu(), latents_tmp.name)
    latents_url = upload_file(uid, token, latents_tmp.name)
    os.unlink(latents_tmp.name)

    return {"latents": latents_url}
