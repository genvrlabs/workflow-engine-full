"""Desaturate: blend original with grayscale version."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Desaturate",
    "description": "Blend image with its grayscale version. Amount 0=original, 1=full grayscale.",
    "category": "color",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "amount", "display_name": "Amount (0=original, 1=grayscale)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    amount = float(inputs.get("amount", 1.0))
    amount = max(0.0, min(1.0, amount))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = cv2.addWeighted(img, 1.0 - amount, gray_bgr, amount, 0)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
