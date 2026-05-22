"""Flip image horizontally, vertically, or both."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Flip Image",
    "description": "Flip image horizontally, vertically, or in both directions.",
    "category": "transform",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "direction", "display_name": "Direction (horizontal/vertical/both)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    direction = inputs.get("direction", "horizontal").lower()

    flip_code = {"horizontal": 1, "vertical": 0, "both": -1}.get(direction)
    if flip_code is None:
        raise ValueError(f"Unknown direction: {direction}. Use horizontal/vertical/both.")

    img = load_image_from_url(url)
    result = cv2.flip(img, flip_code)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
