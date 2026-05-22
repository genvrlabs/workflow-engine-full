"""Add Gaussian noise to an image."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Add Gaussian Noise",
    "description": "Add Gaussian noise with specified mean and standard deviation.",
    "category": "noise",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "mean", "display_name": "Mean", "type": "number"},
    {"var_name": "std", "display_name": "Standard Deviation", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    mean = float(inputs.get("mean", 0))
    std = float(inputs.get("std", 25))

    img = load_image_from_url(url)
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    result = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
