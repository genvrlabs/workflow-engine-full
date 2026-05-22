"""Fit image into a target canvas using fit/fill/stretch modes."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Fit Image",
    "description": "Scale and place image into a target canvas using fit, fill, or stretch mode.",
    "category": "transform",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "target_width", "display_name": "Target Width", "type": "number"},
    {"var_name": "target_height", "display_name": "Target Height", "type": "number"},
    {"var_name": "mode", "display_name": "Mode (fit/fill/stretch)", "type": "text"},
    {"var_name": "bg_color", "display_name": "Background Color (R,G,B)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    target_w = int(inputs.get("target_width", 0))
    target_h = int(inputs.get("target_height", 0))
    if target_w <= 0 or target_h <= 0:
        raise ValueError("target_width and target_height must be positive")
    mode = inputs.get("mode", "fit").lower()
    bg_color_str = inputs.get("bg_color", "0,0,0")
    bg_rgb = tuple(int(x) for x in bg_color_str.split(","))
    if len(bg_rgb) < 3:
        bg_rgb = (0, 0, 0)
    bg_bgr = (bg_rgb[2], bg_rgb[1], bg_rgb[0])

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    if mode == "stretch":
        result = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    elif mode == "fill":
        # Scale so image fills entire canvas, then center-crop
        scale = max(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        x_off = (new_w - target_w) // 2
        y_off = (new_h - target_h) // 2
        result = scaled[y_off:y_off + target_h, x_off:x_off + target_w]
    else:  # fit
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        canvas = np.full((target_h, target_w, 3), bg_bgr, dtype=np.uint8)
        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = scaled
        result = canvas

    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
