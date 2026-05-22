"""Adjust hue, saturation, and value of an image."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Adjust Hue & Saturation",
    "description": "Shift hue and scale saturation and value channels of an image in HSV space.",
    "category": "color",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "hue_shift", "display_name": "Hue Shift (-180 to 180)", "type": "number"},
    {"var_name": "saturation_scale", "display_name": "Saturation Scale", "type": "number"},
    {"var_name": "value_scale", "display_name": "Value Scale", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    hue_shift = float(inputs.get("hue_shift", 0))
    saturation_scale = float(inputs.get("saturation_scale", 1.0))
    value_scale = float(inputs.get("value_scale", 1.0))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * value_scale, 0, 255)
    hsv = hsv.astype(np.uint8)
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
