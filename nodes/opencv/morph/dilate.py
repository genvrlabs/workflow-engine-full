"""Morphological dilation."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Dilate",
    "description": "Apply morphological dilation to expand bright regions.",
    "category": "morph",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "radius", "display_name": "Radius", "type": "number"},
    {"var_name": "iterations", "display_name": "Iterations", "type": "number"},
    {"var_name": "shape", "display_name": "Shape (rect/ellipse/cross)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]

_SHAPE_MAP = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    radius = int(inputs.get("radius", 3))
    iterations = int(inputs.get("iterations", 1))
    shape_name = inputs.get("shape", "rect").lower()
    morph_shape = _SHAPE_MAP.get(shape_name, cv2.MORPH_RECT)

    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(morph_shape, (k, k))
    img = load_image_from_url(url)
    result = cv2.dilate(img, kernel, iterations=iterations)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
