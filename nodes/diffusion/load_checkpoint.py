"""
Load Checkpoint node — choose a model by style, or type an exact HuggingFace
repo id directly if you know the specific SDXL-compatible model you want.
"""

from __future__ import annotations

from nodes.diffusion._model_choices import MODEL_CHOICES

metadata = {
    "display_name": "Load Checkpoint",
    "description": "Pick a model style, or type any SDXL-compatible HuggingFace repo id (advanced).",
    "category": "diffusion",
    "color": "purple",
}

inputs = [
    {
        "var_name": "model_name",
        "display_name": "Model style",
        "type": "dropdown",
        "options": list(MODEL_CHOICES.keys()),
    },
]

outputs = [
    {"var_name": "model_id", "display_name": "Model ID", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    model_name = (inputs.get("model_name") or "balanced_general_purpose").strip()

    if model_name in MODEL_CHOICES:
        return {"model_id": model_name}

    if "/" in model_name:
        return {"model_id": model_name}

    raise ValueError(
        f"\'{model_name}\' isn\'t a recognized model style, and doesn\'t look like a "
        f"HuggingFace repo id either. Pick one of: {list(MODEL_CHOICES.keys())}, "
        f"or type an exact model id like \'org/model-name\' for a specific SDXL-architecture model."
    )
