"""Chromatic aberration: shift R and B channels in opposite directions."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Chromatic Aberration",
    "description": "Simulate chromatic aberration by shifting the red and blue channels in opposite directions.",
    "category": "lens",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "shift_x", "display_name": "Horizontal Shift (pixels)", "type": "number"},
    {"var_name": "shift_y", "display_name": "Vertical Shift (pixels)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


def _shift_channel(channel: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift a 2D channel by (dx, dy) pixels."""
    h, w = channel.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(channel, M, (w, h))


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    shift_x = int(inputs.get("shift_x", 3))
    shift_y = int(inputs.get("shift_y", 0))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    r_shifted = _shift_channel(r, shift_x, shift_y)
    b_shifted = _shift_channel(b, -shift_x, -shift_y)
    result = cv2.merge([b_shifted, g, r_shifted])
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
