"""Apply gamma correction via a lookup table."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Gamma Correction",
    "description": "Apply gamma correction to an image using a LUT.",
    "category": "color",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "gamma", "display_name": "Gamma", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    gamma = float(inputs.get("gamma", 1.0))
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    img = load_image_from_url(url)
    inv_gamma = 1.0 / gamma
    lut = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    result = cv2.LUT(img, lut)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
