"""Mix blend: linear interpolation between two layers."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Blend Mix",
    "description": "Linear mix between two layers using cv2.addWeighted. mix=0: A, mix=1: B.",
    "category": "composite",
    "color": "blue",
}

inputs = [
    {"var_name": "layer_a_url", "display_name": "Layer A URL", "type": "text"},
    {"var_name": "layer_b_url", "display_name": "Layer B URL", "type": "text"},
    {"var_name": "mix", "display_name": "Mix (0=A, 1=B)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    a_url = inputs.get("layer_a_url", "")
    b_url = inputs.get("layer_b_url", "")
    if not a_url or not b_url:
        raise ValueError("layer_a_url and layer_b_url are required")
    mix = float(inputs.get("mix", 0.5))
    mix = max(0.0, min(1.0, mix))

    a = load_image_from_url(a_url, cv2.IMREAD_COLOR)
    b = load_image_from_url(b_url, cv2.IMREAD_COLOR)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    result = cv2.addWeighted(a, 1.0 - mix, b, mix, 0)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
