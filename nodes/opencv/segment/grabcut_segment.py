"""GrabCut segmentation using rectangular initialization."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "GrabCut Segment",
    "description": "Segment foreground from background using GrabCut algorithm with a bounding rectangle.",
    "category": "segment",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "x", "display_name": "Rectangle X", "type": "number"},
    {"var_name": "y", "display_name": "Rectangle Y", "type": "number"},
    {"var_name": "width", "display_name": "Rectangle Width", "type": "number"},
    {"var_name": "height", "display_name": "Rectangle Height", "type": "number"},
    {"var_name": "iterations", "display_name": "Iterations", "type": "number"},
]

outputs = [
    {"var_name": "mask_url", "display_name": "Mask URL (grayscale)", "type": "text"},
    {"var_name": "output_url", "display_name": "Output Image URL (BGRA)", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    x = int(inputs.get("x", 0))
    y = int(inputs.get("y", 0))
    w = int(inputs.get("width", 0))
    h = int(inputs.get("height", 0))
    iterations = int(inputs.get("iterations", 5))
    if w <= 0 or h <= 0:
        raise ValueError("width and height must be positive")

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (x, y, w, h)
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    # Create binary mask: 0/2=background, 1/3=foreground
    binary_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)

    bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = binary_mask

    mask_url = save_and_upload(binary_mask, uid, token, ".png")
    output_url = save_and_upload(bgra, uid, token, ".png")
    return {"mask_url": mask_url, "output_url": output_url}
