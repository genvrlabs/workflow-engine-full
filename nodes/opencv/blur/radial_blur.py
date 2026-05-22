"""Radial (zoom) blur: average multiple scaled copies around a center point."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Radial Blur",
    "description": "Apply radial zoom-blur effect by averaging scaled copies around a center point.",
    "category": "blur",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "amount", "display_name": "Amount", "type": "number"},
    {"var_name": "center_x", "display_name": "Center X (-1=image center)", "type": "number"},
    {"var_name": "center_y", "display_name": "Center Y (-1=image center)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    amount = int(inputs.get("amount", 10))
    center_x = float(inputs.get("center_x", -1))
    center_y = float(inputs.get("center_y", -1))

    img = load_image_from_url(url)
    h, w = img.shape[:2]
    cx = w / 2.0 if center_x < 0 else center_x
    cy = h / 2.0 if center_y < 0 else center_y

    num_steps = max(amount, 1)
    accum = img.astype(np.float32)
    for i in range(1, num_steps + 1):
        scale = 1.0 + (i / num_steps) * 0.02 * amount
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        scaled = cv2.warpAffine(img, M, (w, h))
        accum += scaled.astype(np.float32)
    result = np.clip(accum / (num_steps + 1), 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
