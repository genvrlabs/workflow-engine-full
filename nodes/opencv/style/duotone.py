"""Duotone: map grayscale luminance between shadow and highlight colors."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Duotone",
    "description": "Map image luminance to a two-color gradient (shadow to highlight).",
    "category": "style",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "color_shadows", "display_name": "Shadow Color (R,G,B)", "type": "text"},
    {"var_name": "color_highlights", "display_name": "Highlight Color (R,G,B)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    shadow_str = inputs.get("color_shadows", "0,0,128")
    highlight_str = inputs.get("color_highlights", "255,200,100")
    shadow_rgb = tuple(int(x) for x in shadow_str.split(","))
    highlight_rgb = tuple(int(x) for x in highlight_str.split(","))
    if len(shadow_rgb) < 3:
        shadow_rgb = (0, 0, 128)
    if len(highlight_rgb) < 3:
        highlight_rgb = (255, 200, 100)

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Map gray (0-1) between shadow and highlight colors
    # Result in BGR
    result = np.zeros((*gray.shape, 3), dtype=np.float32)
    for ch_idx, (s, h) in enumerate(zip(
        (shadow_rgb[2], shadow_rgb[1], shadow_rgb[0]),     # BGR shadow
        (highlight_rgb[2], highlight_rgb[1], highlight_rgb[0])  # BGR highlight
    )):
        result[:, :, ch_idx] = s / 255.0 * (1.0 - gray) + h / 255.0 * gray

    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
