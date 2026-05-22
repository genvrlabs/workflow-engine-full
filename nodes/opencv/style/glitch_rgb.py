"""RGB glitch: randomly shift R and B channels in horizontal bands."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Glitch RGB",
    "description": "Apply digital glitch effect by randomly shifting RGB channels in horizontal bands.",
    "category": "style",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "shift_amount", "display_name": "Max Shift (pixels)", "type": "number"},
    {"var_name": "num_glitch_bands", "display_name": "Number of Glitch Bands", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    shift_amount = int(inputs.get("shift_amount", 10))
    num_bands = int(inputs.get("num_glitch_bands", 5))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    result = img.copy()

    for _ in range(num_bands):
        y_start = np.random.randint(0, h)
        band_height = np.random.randint(5, max(6, h // 10))
        y_end = min(y_start + band_height, h)
        shift_x = np.random.randint(-shift_amount, shift_amount + 1)

        # Shift R channel for this band
        r_band = r[y_start:y_end, :]
        r_shifted = np.roll(r_band, shift_x, axis=1)
        result[y_start:y_end, :, 2] = r_shifted

        # Shift B channel opposite direction
        b_band = b[y_start:y_end, :]
        b_shifted = np.roll(b_band, -shift_x, axis=1)
        result[y_start:y_end, :, 0] = b_shifted

    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
