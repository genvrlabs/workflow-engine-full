"""Composite over: Porter-Duff 'over' compositing."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Composite Over",
    "description": "Composite a foreground image over a background using Porter-Duff over operation.",
    "category": "composite",
    "color": "blue",
}

inputs = [
    {"var_name": "fg_url", "display_name": "Foreground Image URL (BGRA or BGR)", "type": "text"},
    {"var_name": "bg_url", "display_name": "Background Image URL (BGR)", "type": "text"},
    {"var_name": "fg_alpha_url", "display_name": "Foreground Alpha URL (optional)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    fg_url = inputs.get("fg_url", "")
    bg_url = inputs.get("bg_url", "")
    if not fg_url or not bg_url:
        raise ValueError("fg_url and bg_url are required")
    fg_alpha_url = inputs.get("fg_alpha_url", "")

    fg = load_image_from_url(fg_url, cv2.IMREAD_UNCHANGED)
    bg = load_image_from_url(bg_url, cv2.IMREAD_COLOR)

    # Resize bg to match fg if needed
    if bg.shape[:2] != fg.shape[:2]:
        bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))

    # Get alpha channel
    if fg_alpha_url:
        alpha_img = load_image_from_url(fg_alpha_url, cv2.IMREAD_GRAYSCALE)
        alpha = alpha_img.astype(np.float32) / 255.0
        if fg.ndim == 4:
            fg_bgr = fg[:, :, :3]
        else:
            fg_bgr = fg
    elif fg.ndim == 4:
        alpha = fg[:, :, 3].astype(np.float32) / 255.0
        fg_bgr = fg[:, :, :3]
    else:
        alpha = np.ones(fg.shape[:2], dtype=np.float32)
        fg_bgr = fg

    alpha3 = alpha[:, :, np.newaxis]
    fg_f = fg_bgr.astype(np.float32)
    bg_f = bg.astype(np.float32)
    result = fg_f * alpha3 + bg_f * (1.0 - alpha3)
    result = np.clip(result, 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
