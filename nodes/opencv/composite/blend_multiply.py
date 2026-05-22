"""Multiply blend mode."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Blend Multiply",
    "description": "Multiply blend mode: (A * B) / 255, mixed by opacity.",
    "category": "composite",
    "color": "blue",
}

inputs = [
    {"var_name": "layer_a_url", "display_name": "Layer A URL", "type": "text"},
    {"var_name": "layer_b_url", "display_name": "Layer B URL", "type": "text"},
    {"var_name": "opacity", "display_name": "Opacity (0-1)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    a_url = inputs.get("layer_a_url", "")
    b_url = inputs.get("layer_b_url", "")
    if not a_url or not b_url:
        raise ValueError("layer_a_url and layer_b_url are required")
    opacity = float(inputs.get("opacity", 1.0))

    a = load_image_from_url(a_url, cv2.IMREAD_COLOR)
    b = load_image_from_url(b_url, cv2.IMREAD_COLOR)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    a_f = a.astype(np.float32) / 255.0
    b_f = b.astype(np.float32) / 255.0
    multiplied = np.clip(a_f * b_f * 255.0, 0, 255).astype(np.uint8)
    result = cv2.addWeighted(a, 1.0 - opacity, multiplied, opacity, 0)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
