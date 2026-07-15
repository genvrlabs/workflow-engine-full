"""
CLIP Text Encode — generic text encoder node.

Drag this node onto the canvas TWICE: once for your positive prompt,
once for your negative prompt. Wire each instance's outputs to the
matching KSampler input slots (embeds/pooled_embeds for positive,
negative_embeds/negative_pooled_embeds for negative).
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import threading

import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from nodes.ffmpeg._utils import upload_file
from nodes.diffusion._model_choices import MODEL_CHOICES

metadata = {
    "display_name": "CLIP Text Encode",
    "description": "Encodes a prompt into SDXL conditioning. Use two of these — one for positive, one for negative.",
    "category": "diffusion",
    "color": "purple",
}

inputs = [
    {"var_name": "model_id", "display_name": "Model ID", "type": "text"},
    {"var_name": "prompt", "display_name": "Prompt", "type": "text"},
]

outputs = [
    {"var_name": "embeds", "display_name": "Embeds", "type": "text"},
    {"var_name": "pooled_embeds", "display_name": "Pooled Embeds", "type": "text"},
]

_cache: dict = {}
_cache_lock = threading.Lock()


def _resolve_repo(model_id: str) -> str:
    return MODEL_CHOICES.get(model_id, model_id)


def _get_models(model_id: str):
    if model_id in _cache:
        return _cache[model_id]
    with _cache_lock:
        if model_id in _cache:
            return _cache[model_id]
        repo_id = _resolve_repo(model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        tokenizer = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(repo_id, subfolder="tokenizer_2")
        text_encoder = CLIPTextModel.from_pretrained(
            repo_id, subfolder="text_encoder", torch_dtype=dtype,
        ).to(device).eval()
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            repo_id, subfolder="text_encoder_2", torch_dtype=dtype,
        ).to(device).eval()

        _cache[model_id] = dict(
            tokenizer=tokenizer, tokenizer_2=tokenizer_2,
            text_encoder=text_encoder, text_encoder_2=text_encoder_2, device=device,
        )
    return _cache[model_id]


def _encode(prompt: str, models: dict):
    device = models["device"]
    ids_1 = models["tokenizer"](
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    ids_2 = models["tokenizer_2"](
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    with torch.no_grad():
        out_1 = models["text_encoder"](ids_1, output_hidden_states=True)
        out_2 = models["text_encoder_2"](ids_2, output_hidden_states=True)
    embeds = torch.cat([out_1.hidden_states[-2], out_2.hidden_states[-2]], dim=-1)
    return embeds, out_2.text_embeds


def _save_and_upload(uid, token, tensor):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    tmp.close()
    torch.save(tensor.cpu(), tmp.name)
    url = upload_file(uid, token, tmp.name)
    os.unlink(tmp.name)
    return url


async def execute(uid: str, token: str, inputs: dict) -> dict:
    model_id = inputs.get("model_id", "balanced_general_purpose")
    prompt = inputs.get("prompt", "") or ""

    models = await asyncio.to_thread(_get_models, model_id)
    embeds, pooled = await asyncio.to_thread(_encode, prompt, models)

    embeds_url = await asyncio.to_thread(_save_and_upload, uid, token, embeds)
    pooled_url = await asyncio.to_thread(_save_and_upload, uid, token, pooled)

    return {"embeds": embeds_url, "pooled_embeds": pooled_url}
