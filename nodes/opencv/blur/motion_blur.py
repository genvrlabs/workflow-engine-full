"""Motion blur using a directional line kernel."""
import json

import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

_NODE = "opencv.motion_blur"


def _debug_console(label: str, data) -> None:
    print(f"[{_NODE}] {label}:\n{json.dumps(data, default=str, indent=2)}", flush=True)

metadata = {
    "display_name": "Motion Blur",
    "description": "Apply directional motion blur with specified length and angle.",
    "category": "blur",
    "color": "blue",
}

inputs = [
    {
        "var_name": "input_url",
        "display_name": "Input Image (URL or {name, uri, type})",
        "type": "any",
        "batch": True,
    },
    {"var_name": "length", "display_name": "Length (pixels)", "type": "number"},
    {"var_name": "angle", "display_name": "Angle (degrees)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    _debug_console("inputs", inputs)

    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    length = int(inputs.get("length", 15))
    angle = float(inputs.get("angle", 0))
    _debug_console("params", {"input_url": url, "length": length, "angle": angle})

    # Build a line kernel
    kernel_size = max(length, 1)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    # Draw a line through center at given angle
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    for i in range(-(kernel_size // 2), kernel_size // 2 + 1):
        x = int(round(center + i * cos_a))
        y = int(round(center + i * sin_a))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    s = kernel.sum()
    if s > 0:
        kernel /= s

    img = load_image_from_url(url)
    result = cv2.filter2D(img, -1, kernel)
    output_url = save_and_upload(result, uid, token, ".png")
    outputs = {"output_url": output_url}
    _debug_console("outputs", outputs)
    return outputs
