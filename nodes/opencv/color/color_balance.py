"""Scale individual RGB channels to adjust color balance."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Color Balance",
    "description": "Scale each RGB channel independently to adjust color balance.",
    "category": "color",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "red_scale", "display_name": "Red Scale", "type": "number"},
    {"var_name": "green_scale", "display_name": "Green Scale", "type": "number"},
    {"var_name": "blue_scale", "display_name": "Blue Scale", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    red_scale = float(inputs.get("red_scale", 1.0))
    green_scale = float(inputs.get("green_scale", 1.0))
    blue_scale = float(inputs.get("blue_scale", 1.0))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    img_f = img.astype(np.float32)
    img_f[:, :, 0] = np.clip(img_f[:, :, 0] * blue_scale, 0, 255)
    img_f[:, :, 1] = np.clip(img_f[:, :, 1] * green_scale, 0, 255)
    img_f[:, :, 2] = np.clip(img_f[:, :, 2] * red_scale, 0, 255)
    result = img_f.astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
