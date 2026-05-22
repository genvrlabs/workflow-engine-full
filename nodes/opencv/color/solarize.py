"""Solarize: invert pixels above a threshold."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Solarize",
    "description": "Invert pixel values that exceed a threshold, creating a solarize effect.",
    "category": "color",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "threshold", "display_name": "Threshold (0-255)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    threshold = int(inputs.get("threshold", 128))

    lut = np.array([i if i < threshold else 255 - i for i in range(256)], dtype=np.uint8)
    img = load_image_from_url(url)
    result = cv2.LUT(img, lut)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
