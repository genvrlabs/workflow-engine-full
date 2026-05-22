"""Anamorphic lens flare: horizontal streaks from highlight points."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Anamorphic Flare",
    "description": "Add anamorphic lens flare streaks to bright highlights.",
    "category": "lens",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "threshold", "display_name": "Highlight Threshold (0-255)", "type": "number"},
    {"var_name": "length", "display_name": "Flare Length (pixels)", "type": "number"},
    {"var_name": "color", "display_name": "Flare Color (R,G,B)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    threshold = int(inputs.get("threshold", 220))
    length = int(inputs.get("length", 200))
    color_str = inputs.get("color", "180,220,255")
    flare_rgb = tuple(int(x) for x in color_str.split(","))
    flare_bgr = (flare_rgb[2], flare_rgb[1], flare_rgb[0]) if len(flare_rgb) == 3 else (255, 220, 180)

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # Extract highlight mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Create flare layer: tinted highlights
    flare_layer = np.zeros_like(img, dtype=np.float32)
    flare_layer[bright_mask > 0] = flare_bgr

    # Apply horizontal motion blur for streak
    k = max(length | 1, 3)  # ensure odd
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    flare_blurred = cv2.filter2D(flare_layer, -1, kernel)

    result = np.clip(img.astype(np.float32) + flare_blurred, 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
