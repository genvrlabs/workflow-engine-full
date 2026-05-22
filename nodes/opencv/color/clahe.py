"""CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "CLAHE",
    "description": "Apply Contrast Limited Adaptive Histogram Equalization to the L channel of a LAB image.",
    "category": "color",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "clip_limit", "display_name": "Clip Limit", "type": "number"},
    {"var_name": "tile_size", "display_name": "Tile Size", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    clip_limit = float(inputs.get("clip_limit", 2.0))
    tile_size = int(inputs.get("tile_size", 8))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
