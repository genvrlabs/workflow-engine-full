"""Set alpha: combine BGR image with a grayscale alpha channel into BGRA."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Set Alpha",
    "description": "Combine a BGR image with a grayscale alpha mask to produce a BGRA image.",
    "category": "channel",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL (BGR)", "type": "text"},
    {"var_name": "alpha_url", "display_name": "Alpha Channel URL (grayscale)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL (BGRA)", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    img_url = inputs.get("input_url", "")
    alpha_url = inputs.get("alpha_url", "")
    if not img_url or not alpha_url:
        raise ValueError("input_url and alpha_url are required")

    img = load_image_from_url(img_url, cv2.IMREAD_COLOR)
    alpha = load_image_from_url(alpha_url, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h))

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    output_url = save_and_upload(bgra, uid, token, ".png")
    return {"output_url": output_url}
