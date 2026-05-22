"""Feather matte: soften matte edges with Gaussian blur."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Feather Matte",
    "description": "Soften matte edges by applying Gaussian blur to the mask.",
    "category": "key",
    "color": "blue",
}

inputs = [
    {"var_name": "mask_url", "display_name": "Mask URL", "type": "text"},
    {"var_name": "radius", "display_name": "Feather Radius", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Mask URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("mask_url", "")
    if not url:
        raise ValueError("mask_url is required")
    radius = int(inputs.get("radius", 5))

    k = 2 * radius + 1
    mask = load_image_from_url(url, cv2.IMREAD_GRAYSCALE)
    result = cv2.GaussianBlur(mask, (k, k), 0)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
