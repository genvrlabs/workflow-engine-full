"""Resize image with optional aspect ratio preservation."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Resize Image",
    "description": "Resize an image to specified dimensions with optional aspect ratio preservation.",
    "category": "transform",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "width", "display_name": "Width (-1=auto)", "type": "number"},
    {"var_name": "height", "display_name": "Height (-1=auto)", "type": "number"},
    {"var_name": "keep_aspect", "display_name": "Keep Aspect Ratio", "type": "boolean"},
    {"var_name": "interpolation", "display_name": "Interpolation (lanczos/linear/cubic/nearest)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]

_INTERP_MAP = {
    "lanczos": cv2.INTER_LANCZOS4,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
}


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    target_w = int(inputs.get("width", -1))
    target_h = int(inputs.get("height", -1))
    keep_aspect = bool(inputs.get("keep_aspect", True))
    interp_name = inputs.get("interpolation", "lanczos").lower()
    interp = _INTERP_MAP.get(interp_name, cv2.INTER_LANCZOS4)

    img = load_image_from_url(url)
    h, w = img.shape[:2]

    if target_w <= 0 and target_h <= 0:
        raise ValueError("At least one of width or height must be specified (> 0)")

    if keep_aspect:
        if target_w <= 0:
            scale = target_h / h
            target_w = int(w * scale)
        elif target_h <= 0:
            scale = target_w / w
            target_h = int(h * scale)
        else:
            scale = min(target_w / w, target_h / h)
            target_w = int(w * scale)
            target_h = int(h * scale)
    else:
        if target_w <= 0:
            target_w = w
        if target_h <= 0:
            target_h = h

    result = cv2.resize(img, (target_w, target_h), interpolation=interp)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
