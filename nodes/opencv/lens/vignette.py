"""Vignette effect: darken image edges with an elliptical gradient."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Vignette",
    "description": "Apply vignette effect by darkening the image edges with an elliptical gradient.",
    "category": "lens",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "strength", "display_name": "Strength (0-1)", "type": "number"},
    {"var_name": "radius", "display_name": "Radius (0-1 relative)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    strength = float(inputs.get("strength", 0.5))
    radius = float(inputs.get("radius", 0.8))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # Create elliptical gradient mask
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2.0, h / 2.0
    # Normalized distance from center (elliptical)
    dist = np.sqrt(((X - cx) / (w / 2.0)) ** 2 + ((Y - cy) / (h / 2.0)) ** 2)
    # Create vignette: 1 at center, falling off toward edges
    mask = 1.0 - np.clip((dist - radius) / (1.0 - radius + 1e-6), 0, 1) * strength
    mask = np.clip(mask, 0, 1).astype(np.float32)

    img_f = img.astype(np.float32)
    if img.ndim == 3:
        mask3 = mask[:, :, np.newaxis]
    else:
        mask3 = mask
    result = np.clip(img_f * mask3, 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
