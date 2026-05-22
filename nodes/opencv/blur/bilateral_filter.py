"""Bilateral filter: edge-preserving smoothing."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Bilateral Filter",
    "description": "Apply edge-preserving bilateral filter.",
    "category": "blur",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "diameter", "display_name": "Diameter", "type": "number"},
    {"var_name": "sigma_color", "display_name": "Sigma Color", "type": "number"},
    {"var_name": "sigma_space", "display_name": "Sigma Space", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    diameter = int(inputs.get("diameter", 9))
    sigma_color = float(inputs.get("sigma_color", 75))
    sigma_space = float(inputs.get("sigma_space", 75))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    result = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
