"""Pixelate: downsample then upsample for a blocky pixel effect."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Pixelate",
    "description": "Create a pixelation effect by downsampling and upsampling with nearest-neighbor.",
    "category": "style",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "pixel_size", "display_name": "Pixel Size", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    pixel_size = max(int(inputs.get("pixel_size", 10)), 1)

    img = load_image_from_url(url)
    h, w = img.shape[:2]
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    result = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
