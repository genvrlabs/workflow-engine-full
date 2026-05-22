"""Merge grayscale images into a BGR or BGRA image."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Merge Channels",
    "description": "Merge separate grayscale images for R, G, B (and optionally A) into a single image.",
    "category": "channel",
    "color": "blue",
}

inputs = [
    {"var_name": "red_url", "display_name": "Red Channel URL", "type": "text"},
    {"var_name": "green_url", "display_name": "Green Channel URL", "type": "text"},
    {"var_name": "blue_url", "display_name": "Blue Channel URL", "type": "text"},
    {"var_name": "alpha_url", "display_name": "Alpha Channel URL (optional)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    r_url = inputs.get("red_url", "")
    g_url = inputs.get("green_url", "")
    b_url = inputs.get("blue_url", "")
    a_url = inputs.get("alpha_url", "")
    if not r_url or not g_url or not b_url:
        raise ValueError("red_url, green_url, and blue_url are required")

    r = load_image_from_url(r_url, cv2.IMREAD_GRAYSCALE)
    g = load_image_from_url(g_url, cv2.IMREAD_GRAYSCALE)
    b = load_image_from_url(b_url, cv2.IMREAD_GRAYSCALE)

    # Ensure all same size
    h, w = r.shape[:2]
    if g.shape[:2] != (h, w):
        g = cv2.resize(g, (w, h))
    if b.shape[:2] != (h, w):
        b = cv2.resize(b, (w, h))

    if a_url:
        a = load_image_from_url(a_url, cv2.IMREAD_GRAYSCALE)
        if a.shape[:2] != (h, w):
            a = cv2.resize(a, (w, h))
        result = cv2.merge([b, g, r, a])
    else:
        result = cv2.merge([b, g, r])

    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
