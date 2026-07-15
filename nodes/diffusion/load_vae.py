"""
Load VAE node — optionally overrides the checkpoint's built-in VAE.
"""

from __future__ import annotations

from nodes.diffusion._model_choices import VAE_CHOICES

metadata = {
    "display_name": "Load VAE",
    "description": "Choose a VAE — \'default\' uses the checkpoint\'s built-in VAE.",
    "category": "diffusion",
    "color": "purple",
}

inputs = [
    {
        "var_name": "vae_name",
        "display_name": "VAE",
        "type": "dropdown",
        "options": list(VAE_CHOICES.keys()),
    },
]

outputs = [
    {"var_name": "vae_id", "display_name": "VAE ID", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    vae_name = (inputs.get("vae_name") or "default").strip()

    if vae_name not in VAE_CHOICES and "/" not in vae_name:
        raise ValueError(
            f"\'{vae_name}\' isn\'t a known VAE. Available: {list(VAE_CHOICES.keys())} "
            f"(or type a raw HuggingFace repo id)"
        )

    return {"vae_id": vae_name}
