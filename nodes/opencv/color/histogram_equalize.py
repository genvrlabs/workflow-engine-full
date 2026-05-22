"""Histogram equalization via YCrCb color space."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Histogram Equalize",
    "description": "Equalize the Y (luminance) channel of a YCrCb image to improve contrast.",
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

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
