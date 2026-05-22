"""Adjust color temperature (warm/cool) and tint (green/magenta)."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Temperature & Tint",
    "description": "Shift image color temperature (warm/cool) and tint (green/magenta).",
    "category": "color",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "temperature", "display_name": "Temperature (-100 warm to +100 cool)", "type": "number"},
    {"var_name": "tint", "display_name": "Tint (-100 to +100)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    temperature = float(inputs.get("temperature", 0))
    tint = float(inputs.get("tint", 0))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    img_f = img.astype(np.float32)
    # Temperature: positive = cool (boost B, reduce R), negative = warm (boost R, reduce B)
    img_f[:, :, 2] = np.clip(img_f[:, :, 2] - temperature, 0, 255)  # R
    img_f[:, :, 0] = np.clip(img_f[:, :, 0] + temperature, 0, 255)  # B
    # Tint: positive = boost G (green shift), negative = reduce G (magenta shift)
    img_f[:, :, 1] = np.clip(img_f[:, :, 1] + tint, 0, 255)  # G
    result = img_f.astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
