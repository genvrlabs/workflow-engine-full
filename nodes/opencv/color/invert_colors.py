"""Invert all pixel values in the image."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Invert Colors",
    "description": "Invert all pixel values using cv2.bitwise_not.",
    "category": "color",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")

    img = load_image_from_url(url)
    result = cv2.bitwise_not(img)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
