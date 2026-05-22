"""Cartoon effect: bilateral filter + Canny edges."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Cartoon",
    "description": "Apply cartoon effect using bilateral filtering for color quantization and Canny for edges.",
    "category": "style",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "num_bilateral", "display_name": "Bilateral Filter Iterations", "type": "number"},
    {"var_name": "blur_kernel", "display_name": "Blur Kernel Size (odd)", "type": "number"},
    {"var_name": "edge_threshold1", "display_name": "Canny Lower Threshold", "type": "number"},
    {"var_name": "edge_threshold2", "display_name": "Canny Upper Threshold", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    num_bilateral = int(inputs.get("num_bilateral", 7))
    blur_kernel = int(inputs.get("blur_kernel", 7))
    edge_threshold1 = float(inputs.get("edge_threshold1", 100))
    edge_threshold2 = float(inputs.get("edge_threshold2", 200))

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    img = load_image_from_url(url, cv2.IMREAD_COLOR)

    # Bilateral filter for color quantization
    color = img
    for _ in range(num_bilateral):
        color = cv2.bilateralFilter(color, 9, 300, 300)

    # Edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_kernel)
    edges = cv2.Canny(gray_blur, edge_threshold1, edge_threshold2)
    edges_inv = cv2.bitwise_not(edges)
    edges_inv_bgr = cv2.cvtColor(edges_inv, cv2.COLOR_GRAY2BGR)

    # Combine
    result = cv2.bitwise_and(color, edges_inv_bgr)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
