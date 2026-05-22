"""Bilateral denoising: apply bilateral filter multiple times."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Denoise Bilateral",
    "description": "Reduce noise by applying bilateral filter multiple times iteratively.",
    "category": "noise",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "iterations", "display_name": "Iterations", "type": "number"},
    {"var_name": "diameter", "display_name": "Diameter", "type": "number"},
    {"var_name": "sigma", "display_name": "Sigma", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    iterations = int(inputs.get("iterations", 3))
    diameter = int(inputs.get("diameter", 9))
    sigma = float(inputs.get("sigma", 75))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    result = img
    for _ in range(max(1, iterations)):
        result = cv2.bilateralFilter(result, diameter, sigma, sigma)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
