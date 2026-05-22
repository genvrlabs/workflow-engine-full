"""Difference key: compare foreground to background to create a matte."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Difference Key",
    "description": "Generate a matte based on pixel differences between foreground and background.",
    "category": "key",
    "color": "blue",
}

inputs = [
    {"var_name": "fg_url", "display_name": "Foreground Image URL", "type": "text"},
    {"var_name": "bg_url", "display_name": "Background Image URL", "type": "text"},
    {"var_name": "threshold", "display_name": "Threshold", "type": "number"},
    {"var_name": "softness", "display_name": "Softness", "type": "number"},
]

outputs = [
    {"var_name": "mask_url", "display_name": "Mask URL", "type": "text"},
    {"var_name": "output_url", "display_name": "Output Image URL (BGRA)", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    fg_url = inputs.get("fg_url", "")
    bg_url = inputs.get("bg_url", "")
    if not fg_url or not bg_url:
        raise ValueError("fg_url and bg_url are required")
    threshold = int(inputs.get("threshold", 30))
    softness = int(inputs.get("softness", 10))

    fg = load_image_from_url(fg_url, cv2.IMREAD_COLOR)
    bg = load_image_from_url(bg_url, cv2.IMREAD_COLOR)
    # Resize bg to match fg if needed
    if bg.shape[:2] != fg.shape[:2]:
        bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))

    diff = cv2.absdiff(fg, bg)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Soft threshold
    alpha = np.clip((diff_gray.astype(np.float32) - threshold) / max(softness, 1) * 255, 0, 255).astype(np.uint8)

    bgra = cv2.cvtColor(fg, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha

    mask_url = save_and_upload(alpha, uid, token, ".png")
    output_url = save_and_upload(bgra, uid, token, ".png")
    return {"mask_url": mask_url, "output_url": output_url}
