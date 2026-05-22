"""Combine two grayscale masks with various operations."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Combine Masks",
    "description": "Combine two grayscale masks using add, subtract, multiply, max, or min operations.",
    "category": "composite",
    "color": "blue",
}

inputs = [
    {"var_name": "mask_a_url", "display_name": "Mask A URL", "type": "text"},
    {"var_name": "mask_b_url", "display_name": "Mask B URL", "type": "text"},
    {"var_name": "operation", "display_name": "Operation (add/subtract/multiply/max/min)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Mask URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    a_url = inputs.get("mask_a_url", "")
    b_url = inputs.get("mask_b_url", "")
    if not a_url or not b_url:
        raise ValueError("mask_a_url and mask_b_url are required")
    operation = inputs.get("operation", "add").lower()

    a = load_image_from_url(a_url, cv2.IMREAD_GRAYSCALE)
    b = load_image_from_url(b_url, cv2.IMREAD_GRAYSCALE)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)

    if operation == "add":
        result = np.clip(a_f + b_f, 0, 255).astype(np.uint8)
    elif operation == "subtract":
        result = np.clip(a_f - b_f, 0, 255).astype(np.uint8)
    elif operation == "multiply":
        result = np.clip(a_f * b_f / 255.0, 0, 255).astype(np.uint8)
    elif operation == "max":
        result = np.maximum(a, b)
    elif operation == "min":
        result = np.minimum(a, b)
    else:
        raise ValueError(f"Unknown operation: {operation}. Use add/subtract/multiply/max/min.")

    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
