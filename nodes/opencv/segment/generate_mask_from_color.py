"""Generate a mask by selecting pixels in an HSV color range."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Generate Mask From Color",
    "description": "Generate a binary mask selecting pixels within a specified HSV color range.",
    "category": "segment",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "hue_center", "display_name": "Hue Center (0-179)", "type": "number"},
    {"var_name": "hue_range", "display_name": "Hue Range (±degrees)", "type": "number"},
    {"var_name": "sat_min", "display_name": "Saturation Min", "type": "number"},
    {"var_name": "val_min", "display_name": "Value Min", "type": "number"},
    {"var_name": "blur", "display_name": "Blur Radius (0=no blur)", "type": "number"},
]

outputs = [
    {"var_name": "mask_url", "display_name": "Mask URL (grayscale)", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    hue_center = int(inputs.get("hue_center", 0))
    hue_range = int(inputs.get("hue_range", 15))
    sat_min = int(inputs.get("sat_min", 80))
    val_min = int(inputs.get("val_min", 50))
    blur_radius = int(inputs.get("blur", 3))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = max(0, hue_center - hue_range)
    h_max = min(179, hue_center + hue_range)
    lower = np.array([h_min, sat_min, val_min], dtype=np.uint8)
    upper = np.array([h_max, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    if blur_radius > 0:
        k = 2 * blur_radius + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    mask_url = save_and_upload(mask, uid, token, ".png")
    return {"mask_url": mask_url}
