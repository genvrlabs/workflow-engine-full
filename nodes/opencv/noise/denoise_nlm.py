"""Non-local means denoising."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Denoise NLM",
    "description": "Apply non-local means denoising (colored or grayscale).",
    "category": "noise",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "h", "display_name": "Filter Strength (h)", "type": "number"},
    {"var_name": "template_window", "display_name": "Template Window Size", "type": "number"},
    {"var_name": "search_window", "display_name": "Search Window Size", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    h_val = float(inputs.get("h", 10))
    template_window = int(inputs.get("template_window", 7))
    search_window = int(inputs.get("search_window", 21))

    img = load_image_from_url(url, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        result = cv2.fastNlMeansDenoising(img, h=h_val,
                                          templateWindowSize=template_window,
                                          searchWindowSize=search_window)
    else:
        bgr = img[:, :, :3] if img.ndim == 4 else img
        result = cv2.fastNlMeansDenoisingColored(bgr, h=h_val, hColor=h_val,
                                                  templateWindowSize=template_window,
                                                  searchWindowSize=search_window)

    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
