"""Hough probabilistic line detection and drawing."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Hough Lines",
    "description": "Detect and draw lines using the probabilistic Hough transform.",
    "category": "edge",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "threshold", "display_name": "Hough Threshold (votes)", "type": "number"},
    {"var_name": "min_length", "display_name": "Minimum Line Length", "type": "number"},
    {"var_name": "max_gap", "display_name": "Maximum Line Gap", "type": "number"},
    {"var_name": "draw_color", "display_name": "Draw Color (R,G,B)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    threshold = int(inputs.get("threshold", 100))
    min_length = float(inputs.get("min_length", 50))
    max_gap = float(inputs.get("max_gap", 10))
    draw_color_str = inputs.get("draw_color", "0,0,255")
    draw_rgb = tuple(int(x) for x in draw_color_str.split(","))
    draw_bgr = (draw_rgb[2], draw_rgb[1], draw_rgb[0]) if len(draw_rgb) == 3 else (0, 0, 255)

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,
                             minLineLength=min_length, maxLineGap=max_gap)
    result = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), draw_bgr, 2)

    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
