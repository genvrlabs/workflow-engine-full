"""Auto levels: stretch each channel so min->0 and max->255."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Auto Levels",
    "description": "Stretch each channel independently so the minimum maps to 0 and maximum to 255.",
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
    channels = cv2.split(img)
    normalized = []
    for ch in channels:
        norm = np.zeros_like(ch)
        cv2.normalize(ch, norm, 0, 255, cv2.NORM_MINMAX)
        normalized.append(norm)
    result = cv2.merge(normalized)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
