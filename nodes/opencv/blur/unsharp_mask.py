"""Unsharp mask: classic sharpening by subtracting blurred version."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Unsharp Mask",
    "description": "Sharpen image using unsharp masking technique.",
    "category": "blur",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "radius", "display_name": "Blur Radius", "type": "number"},
    {"var_name": "amount", "display_name": "Amount", "type": "number"},
    {"var_name": "threshold", "display_name": "Threshold", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    radius = int(inputs.get("radius", 3))
    amount = float(inputs.get("amount", 1.5))
    threshold = int(inputs.get("threshold", 0))

    k = 2 * radius + 1
    img = load_image_from_url(url)
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    img_f = img.astype(np.float32)
    blur_f = blurred.astype(np.float32)
    diff = img_f - blur_f
    mask = np.abs(diff) > threshold
    result = img_f + amount * diff * mask
    result = np.clip(result, 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
