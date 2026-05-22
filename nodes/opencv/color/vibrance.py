"""Vibrance: boost saturation of less-saturated pixels more than already-saturated ones."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Vibrance",
    "description": "Intelligently boost saturation, affecting less-saturated pixels more.",
    "category": "color",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "vibrance", "display_name": "Vibrance (-100 to 100)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    vibrance = float(inputs.get("vibrance", 0))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    weight = 1.0 - s  # less saturated pixels get boosted more
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + vibrance * weight * 2.55, 0, 255)
    hsv = hsv.astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
