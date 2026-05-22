"""Find and draw contours on a black canvas."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Find Contours",
    "description": "Find contours in image and draw them on a black canvas.",
    "category": "edge",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "threshold", "display_name": "Threshold (0-255)", "type": "number"},
    {"var_name": "min_area", "display_name": "Minimum Contour Area", "type": "number"},
    {"var_name": "draw_color", "display_name": "Draw Color (R,G,B)", "type": "text"},
    {"var_name": "line_width", "display_name": "Line Width (pixels)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Contour Drawing URL", "type": "text"},
    {"var_name": "count", "display_name": "Contour Count", "type": "number"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    threshold = int(inputs.get("threshold", 128))
    min_area = float(inputs.get("min_area", 0))
    draw_color_str = inputs.get("draw_color", "0,255,0")
    draw_rgb = tuple(int(x) for x in draw_color_str.split(","))
    draw_bgr = (draw_rgb[2], draw_rgb[1], draw_rgb[0]) if len(draw_rgb) == 3 else (0, 255, 0)
    line_width = int(inputs.get("line_width", 2))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    canvas = np.zeros_like(img)
    cv2.drawContours(canvas, filtered, -1, draw_bgr, line_width)

    output_url = save_and_upload(canvas, uid, token, ".png")
    return {"output_url": output_url, "count": len(filtered)}
