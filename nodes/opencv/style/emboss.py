"""Emboss effect using directional convolution kernel."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Emboss",
    "description": "Apply emboss effect with directional kernel.",
    "category": "style",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "strength", "display_name": "Strength", "type": "number"},
    {"var_name": "direction", "display_name": "Direction (topleft/topright/bottomleft/bottomright)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]

_KERNELS = {
    "topleft": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32),
    "topright": np.array([[0, -1, -2], [1, 1, -1], [2, 1, 0]], dtype=np.float32),
    "bottomleft": np.array([[0, 1, 2], [-1, 1, 1], [-2, -1, 0]], dtype=np.float32),
    "bottomright": np.array([[2, 1, 0], [1, 1, -1], [0, -1, -2]], dtype=np.float32),
}


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    strength = float(inputs.get("strength", 1.0))
    direction = inputs.get("direction", "topleft").lower()

    kernel = _KERNELS.get(direction, _KERNELS["topleft"]) * strength
    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    embossed = cv2.filter2D(img, -1, kernel)
    result = np.clip(embossed.astype(np.float32) + 128, 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
