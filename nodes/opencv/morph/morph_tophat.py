"""Morphological top hat and black hat transforms."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Morph Top Hat",
    "description": "Apply morphological top hat (highlights bright features) or black hat (dark features).",
    "category": "morph",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "radius", "display_name": "Radius", "type": "number"},
    {"var_name": "hat_type", "display_name": "Hat Type (tophat/blackhat)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    radius = int(inputs.get("radius", 15))
    hat_type = inputs.get("hat_type", "tophat").lower()

    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    img = load_image_from_url(url)

    if hat_type == "blackhat":
        morph_type = cv2.MORPH_BLACKHAT
    else:
        morph_type = cv2.MORPH_TOPHAT

    result = cv2.morphologyEx(img, morph_type, kernel)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
