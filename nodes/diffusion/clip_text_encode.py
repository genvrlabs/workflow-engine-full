"""
CLIP / Text Encoder node for SDXL.

Encodes a text prompt using both SDXL text encoders and returns
a single conditioning object for downstream UNet/KSampler nodes.
"""

from __future__ import annotations

import asyncio
import threading

import torch
import os
import tempfile
from nodes.ffmpeg._utils import upload_file

from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

metadata = {
    "display_name": "CLIP Text Encode (SDXL)",
    "description": "Encode a prompt into SDXL conditioning embeddings.",
    "category": "diffusion",
    "color": "purple",
}

inputs = [
    {
        "var_name": "prompt",
        "display_name": "Prompt",
        "type": "text",
    }
]

outputs = [
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
]

_cache = {}
_cache_lock = threading.Lock()


def _get_models():
    if _cache:
        return _cache

    with _cache_lock:
        if _cache:
            return _cache

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_ID,
            subfolder="tokenizer",
        )

        tokenizer_2 = CLIPTokenizer.from_pretrained(
            MODEL_ID,
            subfolder="tokenizer_2",
        )

        text_encoder = CLIPTextModel.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder",
            torch_dtype=dtype,
        ).to(device).eval()

        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            MODEL_ID,
            subfolder="text_encoder_2",
            torch_dtype=dtype,
        ).to(device).eval()

        _cache.update(
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            device=device,
        )

    return _cache


def _encode(prompt: str, models: dict):
    device = models["device"]

    ids_1 = models["tokenizer"](
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    ids_2 = models["tokenizer_2"](
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        out_1 = models["text_encoder"](
            ids_1,
            output_hidden_states=True,
        )

        out_2 = models["text_encoder_2"](
            ids_2,
            output_hidden_states=True,
        )

    hidden_1 = out_1.hidden_states[-2]
    hidden_2 = out_2.hidden_states[-2]

    pooled = out_2.text_embeds

    embeds = torch.cat(
        [hidden_1, hidden_2],
        dim=-1,
    )

    return embeds, pooled


async def execute(uid: str, token: str, inputs: dict) -> dict:
    prompt = inputs.get("prompt", "")

    if not prompt:
        raise ValueError("prompt is required")

    models = await asyncio.to_thread(_get_models)

    embeds, pooled_embeds = await asyncio.to_thread(
        _encode,
        prompt,
        models,
    )

    embeds_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    embeds_tmp.close()
    torch.save(embeds.cpu(), embeds_tmp.name)
    embeds_url = upload_file(uid, token, embeds_tmp.name)
    os.unlink(embeds_tmp.name)

    pooled_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    pooled_tmp.close()
    torch.save(pooled_embeds.cpu(), pooled_tmp.name)
    pooled_url = upload_file(uid, token, pooled_tmp.name)
    os.unlink(pooled_tmp.name)

    return {
        "embeds": embeds_url,
        "pooled_embeds": pooled_url,
    }