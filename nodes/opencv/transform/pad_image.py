"""Pad image with solid color border."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Pad Image",
    "description": "Add padding around the image with a specified fill color.",
    "category": "transform",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "top", "display_name": "Top Padding", "type": "number"},
    {"var_name": "bottom", "display_name": "Bottom Padding", "type": "number"},
    {"var_name": "left", "display_name": "Left Padding", "type": "number"},
    {"var_name": "right", "display_name": "Right Padding", "type": "number"},
    {"var_name": "fill_color", "display_name": "Fill Color (R,G,B)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    top = int(inputs.get("top", 0))
    bottom = int(inputs.get("bottom", 0))
    left = int(inputs.get("left", 0))
    right = int(inputs.get("right", 0))
    fill_color_str = inputs.get("fill_color", "0,0,0")
    fill_rgb = tuple(int(x) for x in fill_color_str.split(","))
    if len(fill_rgb) < 3:
        fill_rgb = (0, 0, 0)
    fill_bgr = (fill_rgb[2], fill_rgb[1], fill_rgb[0])

    img = load_image_from_url(url)
    result = cv2.copyMakeBorder(img, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=fill_bgr)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
