"""Morphological gradient: difference between dilation and erosion."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Morph Gradient",
    "description": "Morphological gradient: the difference between dilation and erosion, highlighting edges.",
    "category": "morph",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "radius", "display_name": "Radius", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    radius = int(inputs.get("radius", 3))

    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    img = load_image_from_url(url)
    result = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
