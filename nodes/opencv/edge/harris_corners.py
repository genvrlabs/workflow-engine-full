"""Harris corner detection."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Harris Corners",
    "description": "Detect corners using the Harris corner detector and mark them on the image.",
    "category": "edge",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "block_size", "display_name": "Block Size", "type": "number"},
    {"var_name": "k_size", "display_name": "Aperture Size (ksize)", "type": "number"},
    {"var_name": "k", "display_name": "Harris Detector Free Parameter (k)", "type": "number"},
    {"var_name": "threshold", "display_name": "Threshold (relative, 0-1)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL with corners marked", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    block_size = int(inputs.get("block_size", 2))
    k_size = int(inputs.get("k_size", 3))
    k = float(inputs.get("k", 0.04))
    threshold = float(inputs.get("threshold", 0.01))

    if k_size % 2 == 0:
        k_size += 1

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    harris = cv2.cornerHarris(gray, block_size, k_size, k)
    harris = cv2.dilate(harris, None)

    result = img.copy()
    result[harris > threshold * harris.max()] = [0, 0, 255]
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
