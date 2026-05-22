"""Chroma keying: remove green, blue, or custom color background."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Chroma Key",
    "description": "Remove a colored background (green/blue/custom) using HSV range keying.",
    "category": "key",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "key_color", "display_name": "Key Color (green/blue/custom)", "type": "text"},
    {"var_name": "custom_hue", "display_name": "Custom Hue (0-179)", "type": "number"},
    {"var_name": "hue_range", "display_name": "Hue Range", "type": "number"},
    {"var_name": "saturation_min", "display_name": "Saturation Min", "type": "number"},
    {"var_name": "spill_suppress", "display_name": "Spill Suppress", "type": "boolean"},
]

outputs = [
    {"var_name": "mask_url", "display_name": "Mask URL (grayscale)", "type": "text"},
    {"var_name": "output_url", "display_name": "Output Image URL (BGRA)", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    key_color = inputs.get("key_color", "green").lower()
    custom_hue = int(inputs.get("custom_hue", 60))
    hue_range = int(inputs.get("hue_range", 30))
    saturation_min = int(inputs.get("saturation_min", 50))
    spill_suppress = bool(inputs.get("spill_suppress", True))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if key_color == "green":
        hue_center = 60
    elif key_color == "blue":
        hue_center = 110
    else:
        hue_center = custom_hue

    h_min = max(0, hue_center - hue_range)
    h_max = min(179, hue_center + hue_range)
    lower = np.array([h_min, saturation_min, 0], dtype=np.uint8)
    upper = np.array([h_max, 255, 255], dtype=np.uint8)
    key_mask = cv2.inRange(hsv, lower, upper)

    # alpha: 0 where keyed, 255 where kept
    alpha = cv2.bitwise_not(key_mask)

    if spill_suppress:
        # Desaturate keyed channel areas slightly
        img_hsv = hsv.copy().astype(np.float32)
        spill_area = key_mask.astype(bool)
        img_hsv[spill_area, 1] *= 0.3
        img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    mask_url = save_and_upload(alpha, uid, token, ".png")
    output_url = save_and_upload(bgra, uid, token, ".png")
    return {"mask_url": mask_url, "output_url": output_url}
