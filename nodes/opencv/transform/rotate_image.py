"""Rotate image by an arbitrary angle with optional canvas expansion."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Rotate Image",
    "description": "Rotate an image by an arbitrary angle with optional canvas expansion.",
    "category": "transform",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "angle", "display_name": "Angle (degrees, CCW)", "type": "number"},
    {"var_name": "expand", "display_name": "Expand Canvas", "type": "boolean"},
    {"var_name": "fill_color", "display_name": "Fill Color (R,G,B)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    angle = float(inputs.get("angle", 0))
    expand = bool(inputs.get("expand", True))
    fill_color_str = inputs.get("fill_color", "0,0,0")
    fill_color = tuple(int(x) for x in fill_color_str.split(","))
    if len(fill_color) < 3:
        fill_color = (0, 0, 0)
    fill_bgr = (fill_color[2], fill_color[1], fill_color[0]) if len(fill_color) == 3 else fill_color

    img = load_image_from_url(url)
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    if expand:
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2.0
        M[1, 2] += (new_h - h) / 2.0
        out_size = (new_w, new_h)
    else:
        out_size = (w, h)

    border_val = fill_bgr[:3] if img.ndim == 3 else fill_bgr[0]
    result = cv2.warpAffine(img, M, out_size, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
