"""Posterize: reduce each channel to N discrete levels."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Posterize",
    "description": "Reduce each color channel to N discrete levels for a poster-like effect.",
    "category": "color",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "levels", "display_name": "Levels", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    levels = int(inputs.get("levels", 4))
    if levels < 2:
        raise ValueError("levels must be at least 2")

    step = 255 // (levels - 1)
    lut = np.array([round(i / step) * step for i in range(256)], dtype=np.uint8)
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    img = load_image_from_url(url)
    result = cv2.LUT(img, lut)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
