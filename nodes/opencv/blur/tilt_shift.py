"""Tilt-shift effect: blur above and below a focus band."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Tilt Shift",
    "description": "Simulate tilt-shift lens effect with a focused band and blurred regions.",
    "category": "blur",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "focus_y", "display_name": "Focus Y (0-1 relative)", "type": "number"},
    {"var_name": "focus_width", "display_name": "Focus Width (0-1 relative)", "type": "number"},
    {"var_name": "blur_radius", "display_name": "Blur Radius", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    focus_y = float(inputs.get("focus_y", 0.5))
    focus_width = float(inputs.get("focus_width", 0.2))
    blur_radius = int(inputs.get("blur_radius", 15))

    img = load_image_from_url(url)
    h, w = img.shape[:2]
    k = 2 * blur_radius + 1
    blurred = cv2.GaussianBlur(img, (k, k), 0)

    # Build a mask: 1 in focus band, 0 outside, gradient in between
    focus_center = int(focus_y * h)
    half_width = int(focus_width * h / 2)
    mask = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        dist = abs(y - focus_center)
        if dist <= half_width:
            mask[y, :] = 1.0
        elif dist <= half_width + blur_radius:
            mask[y, :] = 1.0 - (dist - half_width) / blur_radius
        else:
            mask[y, :] = 0.0

    if img.ndim == 3:
        mask3 = mask[:, :, np.newaxis]
    else:
        mask3 = mask

    img_f = img.astype(np.float32)
    blur_f = blurred.astype(np.float32)
    result = img_f * mask3 + blur_f * (1.0 - mask3)
    result = np.clip(result, 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
