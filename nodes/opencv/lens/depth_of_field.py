"""Depth of field: blur image based on depth map."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Depth of Field",
    "description": "Simulate depth of field by blurring based on a depth map.",
    "category": "lens",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "depth_map_url", "display_name": "Depth Map URL (grayscale, dark=near, light=far)", "type": "text"},
    {"var_name": "max_blur_radius", "display_name": "Max Blur Radius", "type": "number"},
    {"var_name": "focus_depth", "display_name": "Focus Depth (0-255, sharp point)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    depth_url = inputs.get("depth_map_url", "")
    if not url or not depth_url:
        raise ValueError("input_url and depth_map_url are required")
    max_blur_radius = int(inputs.get("max_blur_radius", 20))
    focus_depth = float(inputs.get("focus_depth", 128))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    depth = load_image_from_url(depth_url, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h))

    # Compute blur amount per pixel based on distance from focus_depth
    depth_f = depth.astype(np.float32)
    blur_amount = np.abs(depth_f - focus_depth) / 255.0  # 0-1

    result_f = img.astype(np.float32)
    # Apply multiple blur levels and blend
    num_levels = max(max_blur_radius, 1)
    for i in range(1, num_levels + 1):
        radius = i
        k = 2 * radius + 1
        blurred = cv2.GaussianBlur(img, (k, k), 0).astype(np.float32)
        # Weight: pixels where blur_amount matches this level
        level_val = i / num_levels
        weight = np.clip(1.0 - abs(blur_amount - level_val) * num_levels, 0, 1)
        if img.ndim == 3:
            weight3 = weight[:, :, np.newaxis]
        else:
            weight3 = weight
        result_f = result_f * (1.0 - weight3) + blurred * weight3

    result = np.clip(result_f, 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
