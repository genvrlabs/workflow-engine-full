"""Add film grain noise to an image."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Add Film Grain",
    "description": "Add realistic film grain noise, with optional size scaling for coarser grain.",
    "category": "noise",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "intensity", "display_name": "Intensity (noise std dev)", "type": "number"},
    {"var_name": "size", "display_name": "Grain Size (1.0=fine)", "type": "number"},
    {"var_name": "monochrome", "display_name": "Monochrome Grain", "type": "boolean"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    intensity = float(inputs.get("intensity", 20))
    size = float(inputs.get("size", 1.0))
    monochrome = bool(inputs.get("monochrome", True))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    if size > 1.0:
        small_h = max(1, int(h / size))
        small_w = max(1, int(w / size))
    else:
        small_h, small_w = h, w

    if monochrome:
        grain_small = np.random.normal(0, intensity, (small_h, small_w)).astype(np.float32)
        if size > 1.0:
            grain = cv2.resize(grain_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            grain = grain_small
        grain_3ch = np.stack([grain, grain, grain], axis=2)
    else:
        grain_small = np.random.normal(0, intensity, (small_h, small_w, 3)).astype(np.float32)
        if size > 1.0:
            grain_3ch = cv2.resize(grain_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            grain_3ch = grain_small

    result = np.clip(img.astype(np.float32) + grain_3ch, 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
