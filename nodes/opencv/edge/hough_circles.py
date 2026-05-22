"""Hough circle detection."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Hough Circles",
    "description": "Detect circles using the Hough transform and draw them on the image.",
    "category": "edge",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "min_radius", "display_name": "Min Radius (pixels)", "type": "number"},
    {"var_name": "max_radius", "display_name": "Max Radius (pixels)", "type": "number"},
    {"var_name": "param1", "display_name": "Canny Upper Threshold (param1)", "type": "number"},
    {"var_name": "param2", "display_name": "Accumulator Threshold (param2)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL with circles drawn", "type": "text"},
    {"var_name": "count", "display_name": "Circle Count", "type": "number"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    min_radius = int(inputs.get("min_radius", 10))
    max_radius = int(inputs.get("max_radius", 100))
    param1 = float(inputs.get("param1", 50))
    param2 = float(inputs.get("param2", 30))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
                                minDist=20, param1=param1, param2=param2,
                                minRadius=min_radius, maxRadius=max_radius)
    result = img.copy()
    count = 0
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        count = len(circles)
        for (x, y, r) in circles:
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)

    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url, "count": count}
