"""Lens distortion: barrel or pincushion distortion via radial model."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Lens Distortion",
    "description": "Apply barrel or pincushion lens distortion using a radial distortion model.",
    "category": "lens",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "k1", "display_name": "Radial Distortion k1 (positive=barrel, negative=pincushion)", "type": "number"},
    {"var_name": "k2", "display_name": "Radial Distortion k2", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    k1 = float(inputs.get("k1", 0.2))
    k2 = float(inputs.get("k2", 0.0))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    # Build camera matrix centered at image center
    fx = fy = max(w, h)
    cx, cy = w / 2.0, h / 2.0
    camera_matrix = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0,  1]], dtype=np.float64)
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    result = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
