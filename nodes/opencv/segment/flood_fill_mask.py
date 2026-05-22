"""Flood fill mask: generate a mask from a seed point using flood fill."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Flood Fill Mask",
    "description": "Generate a binary mask using flood fill from a seed point with color tolerance.",
    "category": "segment",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "seed_x", "display_name": "Seed X", "type": "number"},
    {"var_name": "seed_y", "display_name": "Seed Y", "type": "number"},
    {"var_name": "tolerance", "display_name": "Color Tolerance", "type": "number"},
]

outputs = [
    {"var_name": "mask_url", "display_name": "Mask URL (grayscale)", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    seed_x = int(inputs.get("seed_x", 0))
    seed_y = int(inputs.get("seed_y", 0))
    tolerance = int(inputs.get("tolerance", 10))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # Clamp seed point
    seed_x = max(0, min(seed_x, w - 1))
    seed_y = max(0, min(seed_y, h - 1))

    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    lo = (tolerance,) * 3
    hi = (tolerance,) * 3
    cv2.floodFill(img, flood_mask, (seed_x, seed_y), 255,
                  loDiff=lo, upDiff=hi, flags=flags)

    # Trim the extra border added by floodFill
    mask = flood_mask[1:-1, 1:-1]
    mask_url = save_and_upload(mask, uid, token, ".png")
    return {"mask_url": mask_url}
