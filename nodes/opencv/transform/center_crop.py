"""Center crop: crop a region from the center of the image."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Center Crop",
    "description": "Crop a specified region from the center of the image.",
    "category": "transform",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "width", "display_name": "Crop Width", "type": "number"},
    {"var_name": "height", "display_name": "Crop Height", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    crop_w = int(inputs.get("width", 0))
    crop_h = int(inputs.get("height", 0))
    if crop_w <= 0 or crop_h <= 0:
        raise ValueError("width and height must be positive")

    img = load_image_from_url(url)
    ih, iw = img.shape[:2]
    x = max(0, (iw - crop_w) // 2)
    y = max(0, (ih - crop_h) // 2)
    x2 = min(x + crop_w, iw)
    y2 = min(y + crop_h, ih)
    result = img[y:y2, x:x2]
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
