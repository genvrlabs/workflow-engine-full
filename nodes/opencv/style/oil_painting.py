"""Oil painting effect using cv2.xphoto.oilPainting (opencv-contrib)."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Oil Painting",
    "description": "Apply oil painting artistic effect (requires opencv-contrib-python).",
    "category": "style",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "size", "display_name": "Brush Size", "type": "number"},
    {"var_name": "dynRatio", "display_name": "Dynamic Ratio", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    size = int(inputs.get("size", 7))
    dynRatio = int(inputs.get("dynRatio", 1))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    try:
        result = cv2.xphoto.oilPainting(img, size, dynRatio)
    except AttributeError:
        raise RuntimeError("cv2.xphoto.oilPainting not available. Install opencv-contrib-python.")
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
