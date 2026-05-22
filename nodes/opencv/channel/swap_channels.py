"""Swap two color channels in an image."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Swap Channels",
    "description": "Swap two color channel planes (e.g., swap red and blue).",
    "category": "channel",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "from_channel", "display_name": "From Channel (red/green/blue)", "type": "text"},
    {"var_name": "to_channel", "display_name": "To Channel (red/green/blue)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]

_CHANNEL_IDX = {"blue": 0, "green": 1, "red": 2}


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    from_ch = inputs.get("from_channel", "red").lower()
    to_ch = inputs.get("to_channel", "blue").lower()

    if from_ch not in _CHANNEL_IDX or to_ch not in _CHANNEL_IDX:
        raise ValueError("Channels must be one of: red, green, blue")

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    result = img.copy()
    fi = _CHANNEL_IDX[from_ch]
    ti = _CHANNEL_IDX[to_ch]
    result[:, :, fi] = img[:, :, ti]
    result[:, :, ti] = img[:, :, fi]
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
