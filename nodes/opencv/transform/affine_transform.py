"""Affine transform: translate, scale, and shear an image."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Affine Transform",
    "description": "Apply affine transformation: translate, scale, and shear.",
    "category": "transform",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "translate_x", "display_name": "Translate X (pixels)", "type": "number"},
    {"var_name": "translate_y", "display_name": "Translate Y (pixels)", "type": "number"},
    {"var_name": "scale_x", "display_name": "Scale X", "type": "number"},
    {"var_name": "scale_y", "display_name": "Scale Y", "type": "number"},
    {"var_name": "shear", "display_name": "Shear", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    tx = float(inputs.get("translate_x", 0))
    ty = float(inputs.get("translate_y", 0))
    sx = float(inputs.get("scale_x", 1.0))
    sy = float(inputs.get("scale_y", 1.0))
    shear = float(inputs.get("shear", 0))

    # Build 2x3 affine matrix
    M = np.array([
        [sx, shear, tx],
        [0,  sy,    ty],
    ], dtype=np.float32)

    img = load_image_from_url(url)
    h, w = img.shape[:2]
    result = cv2.warpAffine(img, M, (w, h))
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
