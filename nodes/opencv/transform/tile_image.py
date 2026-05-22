"""Tile image in a grid pattern."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Tile Image",
    "description": "Tile the image in a grid of specified columns and rows.",
    "category": "transform",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "cols", "display_name": "Columns", "type": "number"},
    {"var_name": "rows", "display_name": "Rows", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    cols = int(inputs.get("cols", 2))
    rows = int(inputs.get("rows", 2))
    if cols <= 0 or rows <= 0:
        raise ValueError("cols and rows must be positive")

    img = load_image_from_url(url)
    if img.ndim == 2:
        result = np.tile(img, (rows, cols))
    else:
        result = np.tile(img, (rows, cols, 1))
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
