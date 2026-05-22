"""Add salt and pepper noise to an image."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Add Salt & Pepper Noise",
    "description": "Add random white (salt) and black (pepper) pixels to the image.",
    "category": "noise",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "amount", "display_name": "Amount (0-1, fraction of pixels)", "type": "number"},
    {"var_name": "salt_ratio", "display_name": "Salt Ratio (fraction that are white)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    amount = float(inputs.get("amount", 0.05))
    salt_ratio = float(inputs.get("salt_ratio", 0.5))

    img = load_image_from_url(url).copy()
    h, w = img.shape[:2]
    num_pixels = h * w
    num_noisy = int(amount * num_pixels)

    # Salt (white)
    num_salt = int(num_noisy * salt_ratio)
    salt_coords = (
        np.random.randint(0, h, num_salt),
        np.random.randint(0, w, num_salt),
    )
    img[salt_coords] = 255

    # Pepper (black)
    num_pepper = num_noisy - num_salt
    pepper_coords = (
        np.random.randint(0, h, num_pepper),
        np.random.randint(0, w, num_pepper),
    )
    img[pepper_coords] = 0

    output_url = save_and_upload(img, uid, token, ".png")
    return {"output_url": output_url}
