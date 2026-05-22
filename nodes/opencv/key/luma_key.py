"""Luma keying: key based on luminance range."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Luma Key",
    "description": "Remove pixels based on their luminance range.",
    "category": "key",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "luma_min", "display_name": "Luma Min (0-255)", "type": "number"},
    {"var_name": "luma_max", "display_name": "Luma Max (0-255)", "type": "number"},
    {"var_name": "invert", "display_name": "Invert Mask", "type": "boolean"},
]

outputs = [
    {"var_name": "mask_url", "display_name": "Mask URL", "type": "text"},
    {"var_name": "output_url", "display_name": "Output Image URL (BGRA)", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    luma_min = int(inputs.get("luma_min", 0))
    luma_max = int(inputs.get("luma_max", 50))
    invert = bool(inputs.get("invert", False))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, luma_min, luma_max)
    # mask is 255 where in range (keyed out area)
    alpha = cv2.bitwise_not(mask)  # 255 where kept
    if invert:
        alpha = cv2.bitwise_not(alpha)

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    mask_url = save_and_upload(alpha, uid, token, ".png")
    output_url = save_and_upload(bgra, uid, token, ".png")
    return {"mask_url": mask_url, "output_url": output_url}
