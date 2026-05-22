"""Range key: key based on HSV range using cv2.inRange."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Range Key",
    "description": "Key pixels within a specified HSV range using cv2.inRange.",
    "category": "key",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "hue_min", "display_name": "Hue Min (0-179)", "type": "number"},
    {"var_name": "hue_max", "display_name": "Hue Max (0-179)", "type": "number"},
    {"var_name": "sat_min", "display_name": "Saturation Min", "type": "number"},
    {"var_name": "sat_max", "display_name": "Saturation Max", "type": "number"},
    {"var_name": "val_min", "display_name": "Value Min", "type": "number"},
    {"var_name": "val_max", "display_name": "Value Max", "type": "number"},
]

outputs = [
    {"var_name": "mask_url", "display_name": "Mask URL", "type": "text"},
    {"var_name": "output_url", "display_name": "Output Image URL (BGRA)", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    hue_min = int(inputs.get("hue_min", 0))
    hue_max = int(inputs.get("hue_max", 179))
    sat_min = int(inputs.get("sat_min", 0))
    sat_max = int(inputs.get("sat_max", 255))
    val_min = int(inputs.get("val_min", 0))
    val_max = int(inputs.get("val_max", 255))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_min, sat_min, val_min], dtype=np.uint8)
    upper = np.array([hue_max, sat_max, val_max], dtype=np.uint8)
    key_mask = cv2.inRange(hsv, lower, upper)
    alpha = cv2.bitwise_not(key_mask)

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    mask_url = save_and_upload(alpha, uid, token, ".png")
    output_url = save_and_upload(bgra, uid, token, ".png")
    return {"mask_url": mask_url, "output_url": output_url}
