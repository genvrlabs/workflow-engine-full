"""Compare two images: compute PSNR, MSE, and amplified difference image."""
import cv2
import numpy as np
import math
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Compare Images",
    "description": "Compare two images by computing PSNR, MSE, and generating an amplified difference image.",
    "category": "analysis",
    "color": "blue",
}

inputs = [
    {"var_name": "image_a_url", "display_name": "Image A URL", "type": "text"},
    {"var_name": "image_b_url", "display_name": "Image B URL", "type": "text"},
]

outputs = [
    {"var_name": "psnr", "display_name": "PSNR (dB)", "type": "number"},
    {"var_name": "mse", "display_name": "MSE", "type": "number"},
    {"var_name": "diff_url", "display_name": "Amplified Difference Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    a_url = inputs.get("image_a_url", "")
    b_url = inputs.get("image_b_url", "")
    if not a_url or not b_url:
        raise ValueError("image_a_url and image_b_url are required")

    a = load_image_from_url(a_url, cv2.IMREAD_COLOR)
    b = load_image_from_url(b_url, cv2.IMREAD_COLOR)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    diff = a_f - b_f
    mse = float(np.mean(diff ** 2))

    if mse == 0:
        psnr = float('inf')
    else:
        psnr = float(20 * math.log10(255.0 / math.sqrt(mse)))

    # Amplified absolute difference
    abs_diff = np.abs(diff)
    amp_diff = np.clip(abs_diff * 10, 0, 255).astype(np.uint8)
    diff_url = save_and_upload(amp_diff, uid, token, ".png")

    return {
        "psnr": psnr,
        "mse": mse,
        "diff_url": diff_url,
    }
