"""Erode/dilate matte: morphological operations to clean a matte."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Erode/Dilate Matte",
    "description": "Apply morphological operations to erode, dilate, open, or close a matte.",
    "category": "key",
    "color": "blue",
}

inputs = [
    {"var_name": "mask_url", "display_name": "Mask URL", "type": "text"},
    {"var_name": "operation", "display_name": "Operation (erode/dilate/open/close)", "type": "text"},
    {"var_name": "radius", "display_name": "Radius", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Mask URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("mask_url", "")
    if not url:
        raise ValueError("mask_url is required")
    operation = inputs.get("operation", "erode").lower()
    radius = int(inputs.get("radius", 3))

    k = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask = load_image_from_url(url, cv2.IMREAD_GRAYSCALE)

    if operation == "erode":
        result = cv2.erode(mask, kernel)
    elif operation == "dilate":
        result = cv2.dilate(mask, kernel)
    elif operation == "open":
        result = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        result = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    else:
        raise ValueError(f"Unknown operation: {operation}. Use erode/dilate/open/close.")

    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
