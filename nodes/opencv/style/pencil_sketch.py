"""Pencil sketch effect using cv2.pencilSketch."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Pencil Sketch",
    "description": "Apply pencil sketch effect using cv2.pencilSketch (requires opencv-contrib).",
    "category": "style",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "sigma_s", "display_name": "Sigma S (spatial)", "type": "number"},
    {"var_name": "sigma_r", "display_name": "Sigma R (range)", "type": "number"},
    {"var_name": "shade_factor", "display_name": "Shade Factor", "type": "number"},
    {"var_name": "mode", "display_name": "Mode (gray/color)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    sigma_s = float(inputs.get("sigma_s", 60))
    sigma_r = float(inputs.get("sigma_r", 0.07))
    shade_factor = float(inputs.get("shade_factor", 0.05))
    mode = inputs.get("mode", "gray").lower()

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray_sketch, color_sketch = cv2.pencilSketch(img, sigma_s=sigma_s, sigma_r=sigma_r,
                                                  shade_factor=shade_factor)
    if mode == "color":
        result = color_sketch
    else:
        result = gray_sketch

    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
