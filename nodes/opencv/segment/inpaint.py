"""Inpainting: fill masked regions using surrounding pixels."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Inpaint",
    "description": "Fill masked regions using cv2.inpaint (Telea or Navier-Stokes method).",
    "category": "segment",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "mask_url", "display_name": "Mask URL (white=fill area)", "type": "text"},
    {"var_name": "radius", "display_name": "Inpaint Radius", "type": "number"},
    {"var_name": "method", "display_name": "Method (telea/ns)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    img_url = inputs.get("input_url", "")
    mask_url = inputs.get("mask_url", "")
    if not img_url or not mask_url:
        raise ValueError("input_url and mask_url are required")
    radius = int(inputs.get("radius", 3))
    method = inputs.get("method", "telea").lower()

    img = load_image_from_url(img_url, cv2.IMREAD_COLOR)
    mask = load_image_from_url(mask_url, cv2.IMREAD_GRAYSCALE)
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    inpaint_method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    result = cv2.inpaint(img, mask, radius, inpaint_method)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
