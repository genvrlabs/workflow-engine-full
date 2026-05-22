"""Curves adjustment: lift blacks, compress highlights, apply midtone gamma."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Curves",
    "description": "Adjust tonal curve: lift shadows, compress highlights, and apply midtone gamma.",
    "category": "color",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "shadows_lift", "display_name": "Shadows Lift (0-50)", "type": "number"},
    {"var_name": "highlights_compress", "display_name": "Highlights Compress (0-50)", "type": "number"},
    {"var_name": "midtone_gamma", "display_name": "Midtone Gamma", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    shadows_lift = float(inputs.get("shadows_lift", 0))
    highlights_compress = float(inputs.get("highlights_compress", 0))
    midtone_gamma = float(inputs.get("midtone_gamma", 1.0))
    if midtone_gamma <= 0:
        raise ValueError("midtone_gamma must be positive")

    low = shadows_lift
    high = 255 - highlights_compress
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        # Map i from [0,255] to [low, high]
        mapped = low + (i / 255.0) * (high - low)
        # Apply gamma on normalized value in [0,1]
        normalized = mapped / 255.0
        if normalized < 0:
            normalized = 0.0
        gamma_applied = normalized ** (1.0 / midtone_gamma)
        lut[i] = np.clip(gamma_applied * 255.0, 0, 255)
    lut = lut.astype(np.uint8)

    img = load_image_from_url(url)
    result = cv2.LUT(img, lut)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
