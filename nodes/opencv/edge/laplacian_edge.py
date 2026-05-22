"""Laplacian edge detection."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Laplacian Edge",
    "description": "Detect edges using the Laplacian operator.",
    "category": "edge",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "ksize", "display_name": "Kernel Size (odd)", "type": "number"},
    {"var_name": "scale", "display_name": "Scale", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    ksize = int(inputs.get("ksize", 3))
    scale = float(inputs.get("scale", 1))
    if ksize % 2 == 0:
        ksize += 1

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize, scale=scale)
    result = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
