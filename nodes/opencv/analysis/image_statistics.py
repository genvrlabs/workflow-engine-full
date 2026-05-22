"""Compute per-channel and overall image statistics."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url

metadata = {
    "display_name": "Image Statistics",
    "description": "Compute per-channel mean, standard deviation, min, and max pixel values.",
    "category": "analysis",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
]

outputs = [
    {"var_name": "mean_r", "display_name": "Mean Red", "type": "number"},
    {"var_name": "mean_g", "display_name": "Mean Green", "type": "number"},
    {"var_name": "mean_b", "display_name": "Mean Blue", "type": "number"},
    {"var_name": "std_r", "display_name": "Std Red", "type": "number"},
    {"var_name": "std_g", "display_name": "Std Green", "type": "number"},
    {"var_name": "std_b", "display_name": "Std Blue", "type": "number"},
    {"var_name": "min_val", "display_name": "Min Value", "type": "number"},
    {"var_name": "max_val", "display_name": "Max Value", "type": "number"},
    {"var_name": "stats", "display_name": "Stats Dict", "type": "dict"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    mean_r = float(np.mean(r))
    mean_g = float(np.mean(g))
    mean_b = float(np.mean(b))
    std_r = float(np.std(r))
    std_g = float(np.std(g))
    std_b = float(np.std(b))
    min_val = float(np.min(img))
    max_val = float(np.max(img))

    stats = {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "std_r": std_r,
        "std_g": std_g,
        "std_b": std_b,
        "min_val": min_val,
        "max_val": max_val,
        "mean_overall": float(np.mean(img)),
        "std_overall": float(np.std(img)),
    }

    return {
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "std_r": std_r,
        "std_g": std_g,
        "std_b": std_b,
        "min_val": min_val,
        "max_val": max_val,
        "stats": stats,
    }
