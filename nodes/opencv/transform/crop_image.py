"""Crop a rectangular region from an image."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Crop Image",
    "description": "Crop a rectangular region from the image.",
    "category": "transform",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "x", "display_name": "X (left edge)", "type": "number"},
    {"var_name": "y", "display_name": "Y (top edge)", "type": "number"},
    {"var_name": "width", "display_name": "Width", "type": "number"},
    {"var_name": "height", "display_name": "Height", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    x = int(inputs.get("x", 0))
    y = int(inputs.get("y", 0))
    w = int(inputs.get("width", 0))
    h = int(inputs.get("height", 0))
    if w <= 0 or h <= 0:
        raise ValueError("width and height must be positive")

    img = load_image_from_url(url)
    ih, iw = img.shape[:2]
    x = max(0, min(x, iw - 1))
    y = max(0, min(y, ih - 1))
    x2 = min(x + w, iw)
    y2 = min(y + h, ih)
    result = img[y:y2, x:x2]
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
