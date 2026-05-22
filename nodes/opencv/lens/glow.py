"""Glow effect: extract bright areas, blur, add back."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Glow",
    "description": "Add a cinematic glow effect by extracting and blurring bright highlights.",
    "category": "lens",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "radius", "display_name": "Glow Radius", "type": "number"},
    {"var_name": "intensity", "display_name": "Glow Intensity (0-1)", "type": "number"},
    {"var_name": "threshold", "display_name": "Highlight Threshold (0-255)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    radius = int(inputs.get("radius", 21))
    intensity = float(inputs.get("intensity", 0.6))
    threshold = int(inputs.get("threshold", 200))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    k = 2 * radius + 1

    # Extract highlights
    highlights = img.copy().astype(np.float32)
    highlights[highlights < threshold] = 0
    highlights = np.clip(highlights, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(highlights, (k, k), 0)
    result = cv2.addWeighted(img, 1.0, blurred, intensity, 0)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
