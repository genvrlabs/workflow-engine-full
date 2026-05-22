"""Adaptive threshold: locally adaptive binary threshold."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Adaptive Threshold",
    "description": "Apply adaptive thresholding using local neighborhood statistics.",
    "category": "key",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "block_size", "display_name": "Block Size (odd number)", "type": "number"},
    {"var_name": "C", "display_name": "C (subtracted from mean)", "type": "number"},
    {"var_name": "method", "display_name": "Method (gaussian/mean)", "type": "text"},
]

outputs = [
    {"var_name": "mask_url", "display_name": "Mask URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    block_size = int(inputs.get("block_size", 11))
    C = int(inputs.get("C", 2))
    method = inputs.get("method", "gaussian").lower()

    # block_size must be odd and >= 3
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(block_size, 3)

    adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(gray, 255, adaptive_method, cv2.THRESH_BINARY, block_size, C)

    mask_url = save_and_upload(mask, uid, token, ".png")
    return {"mask_url": mask_url}
