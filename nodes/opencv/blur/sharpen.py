"""Sharpen image using a sharpening kernel."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Sharpen",
    "description": "Sharpen image using a convolution kernel with adjustable strength.",
    "category": "blur",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "strength", "display_name": "Strength", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    strength = float(inputs.get("strength", 1.0))

    s = strength
    kernel = np.array([[-s, -s, -s],
                       [-s, 8 * s + 1, -s],
                       [-s, -s, -s]], dtype=np.float32)
    img = load_image_from_url(url)
    result = cv2.filter2D(img, -1, kernel)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
