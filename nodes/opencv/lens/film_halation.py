"""Film halation: red-biased highlight bloom from bright areas."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Film Halation",
    "description": "Simulate film halation by creating a red-biased bloom from bright highlights.",
    "category": "lens",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "threshold", "display_name": "Highlight Threshold (0-255)", "type": "number"},
    {"var_name": "radius", "display_name": "Halation Radius", "type": "number"},
    {"var_name": "intensity", "display_name": "Intensity (0-1)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    threshold = int(inputs.get("threshold", 220))
    radius = int(inputs.get("radius", 30))
    intensity = float(inputs.get("intensity", 0.4))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    k = 2 * radius + 1

    # Extract highlights and tint red
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    halation = np.zeros_like(img, dtype=np.float32)
    halation[:, :, 2] = bright_mask.astype(np.float32)  # red channel only

    blurred = cv2.GaussianBlur(halation, (k, k), 0)

    # Screen blend back
    img_f = img.astype(np.float32) / 255.0
    bloom_f = blurred / 255.0 * intensity
    result = 1.0 - (1.0 - img_f) * (1.0 - bloom_f)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
