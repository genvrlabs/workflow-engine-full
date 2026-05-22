"""Extract a single channel as a grayscale image."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Extract Channel",
    "description": "Extract a single channel (red/green/blue/alpha/luminance) as a grayscale image.",
    "category": "channel",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "channel", "display_name": "Channel (red/green/blue/alpha/luminance)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL (grayscale)", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    channel = inputs.get("channel", "red").lower()

    img = load_image_from_url(url, cv2.IMREAD_UNCHANGED)

    if channel == "red":
        if img.ndim == 2:
            result = img
        else:
            result = img[:, :, 2]
    elif channel == "green":
        if img.ndim == 2:
            result = img
        else:
            result = img[:, :, 1]
    elif channel == "blue":
        if img.ndim == 2:
            result = img
        else:
            result = img[:, :, 0]
    elif channel == "alpha":
        if img.ndim == 4 or (img.ndim == 3 and img.shape[2] == 4):
            result = img[:, :, 3]
        else:
            result = np.full(img.shape[:2], 255, dtype=np.uint8)
    elif channel == "luminance":
        if img.ndim == 2:
            result = img
        else:
            bgr = img[:, :, :3]
            result = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unknown channel: {channel}. Use red/green/blue/alpha/luminance.")

    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
