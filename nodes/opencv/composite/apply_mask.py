"""Apply a grayscale mask as alpha to an input image."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Apply Mask",
    "description": "Apply a grayscale mask as the alpha channel to produce a BGRA image.",
    "category": "composite",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "mask_url", "display_name": "Mask URL (grayscale)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL (BGRA)", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    img_url = inputs.get("input_url", "")
    mask_url = inputs.get("mask_url", "")
    if not img_url or not mask_url:
        raise ValueError("input_url and mask_url are required")

    img = load_image_from_url(img_url, cv2.IMREAD_COLOR)
    mask = load_image_from_url(mask_url, cv2.IMREAD_GRAYSCALE)
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    output_url = save_and_upload(bgra, uid, token, ".png")
    return {"output_url": output_url}
