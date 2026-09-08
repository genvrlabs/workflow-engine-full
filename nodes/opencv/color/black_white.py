"""Converting image to black and white"""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Black & White",
    "description": "Convert a color image to a grayscale (black and white) image.",
    "category": "color",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
   

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = gray_bgr
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
