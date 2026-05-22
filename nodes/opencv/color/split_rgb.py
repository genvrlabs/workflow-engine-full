"""Split an image into its R, G, and B channels as grayscale images."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Split RGB",
    "description": "Split a color image into three grayscale images, one per channel.",
    "category": "color",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
]

outputs = [
    {"var_name": "red_url", "display_name": "Red Channel URL", "type": "text"},
    {"var_name": "green_url", "display_name": "Green Channel URL", "type": "text"},
    {"var_name": "blue_url", "display_name": "Blue Channel URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    red_url = save_and_upload(r, uid, token, ".png")
    green_url = save_and_upload(g, uid, token, ".png")
    blue_url = save_and_upload(b, uid, token, ".png")
    return {"red_url": red_url, "green_url": green_url, "blue_url": blue_url}
