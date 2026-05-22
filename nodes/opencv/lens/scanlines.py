"""Scanlines: draw horizontal lines over image for CRT effect."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Scanlines",
    "description": "Overlay horizontal scanlines on the image for a CRT/retro effect.",
    "category": "lens",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "line_gap", "display_name": "Line Gap (pixels between lines)", "type": "number"},
    {"var_name": "line_width", "display_name": "Line Width (pixels)", "type": "number"},
    {"var_name": "opacity", "display_name": "Opacity (0-1)", "type": "number"},
    {"var_name": "color", "display_name": "Line Color (R,G,B)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    line_gap = int(inputs.get("line_gap", 4))
    line_width = int(inputs.get("line_width", 1))
    opacity = float(inputs.get("opacity", 0.3))
    color_str = inputs.get("color", "0,0,0")
    color_rgb = tuple(int(x) for x in color_str.split(","))
    color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0]) if len(color_rgb) == 3 else (0, 0, 0)

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    overlay = img.copy()

    y = 0
    while y < h:
        for dy in range(line_width):
            row = y + dy
            if row < h:
                overlay[row, :] = color_bgr
        y += line_gap + line_width

    result = cv2.addWeighted(img, 1.0 - opacity, overlay, opacity, 0)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
