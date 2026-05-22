"""Sobel edge detection."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Sobel Edge",
    "description": "Detect edges using Sobel operator in X, Y, or both directions.",
    "category": "edge",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "direction", "display_name": "Direction (x/y/both)", "type": "text"},
    {"var_name": "ksize", "display_name": "Kernel Size (odd)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    direction = inputs.get("direction", "both").lower()
    ksize = int(inputs.get("ksize", 3))
    if ksize % 2 == 0:
        ksize += 1

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if direction == "x":
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    elif direction == "y":
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    else:
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        sobel = np.sqrt(sx ** 2 + sy ** 2)

    result = np.clip(np.abs(sobel), 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
