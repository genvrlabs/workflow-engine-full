"""Canny edge detection."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Canny Edge",
    "description": "Detect edges using the Canny edge detection algorithm.",
    "category": "edge",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "threshold1", "display_name": "Lower Threshold", "type": "number"},
    {"var_name": "threshold2", "display_name": "Upper Threshold", "type": "number"},
    {"var_name": "invert", "display_name": "Invert Output", "type": "boolean"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    threshold1 = float(inputs.get("threshold1", 50))
    threshold2 = float(inputs.get("threshold2", 150))
    invert = bool(inputs.get("invert", False))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    if invert:
        edges = cv2.bitwise_not(edges)
    output_url = save_and_upload(edges, uid, token, ".png")
    return {"output_url": output_url}
