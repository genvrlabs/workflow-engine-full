"""Threshold key: binary threshold to create a matte."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Threshold Key",
    "description": "Create a binary matte using cv2.threshold.",
    "category": "key",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "threshold", "display_name": "Threshold (0-255)", "type": "number"},
    {"var_name": "invert", "display_name": "Invert", "type": "boolean"},
]

outputs = [
    {"var_name": "mask_url", "display_name": "Mask URL", "type": "text"},
    {"var_name": "output_url", "display_name": "Output Image URL (BGRA)", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    threshold = int(inputs.get("threshold", 128))
    invert = bool(inputs.get("invert", False))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, threshold, 255, thresh_type)

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask

    mask_url = save_and_upload(mask, uid, token, ".png")
    output_url = save_and_upload(bgra, uid, token, ".png")
    return {"mask_url": mask_url, "output_url": output_url}
