"""Pixel sorting: sort pixels within rows/columns by luminance in a range."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Pixel Sort",
    "description": "Sort pixels within rows or columns where luminance falls in a specified range.",
    "category": "style",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "direction", "display_name": "Direction (horizontal/vertical)", "type": "text"},
    {"var_name": "threshold_min", "display_name": "Luminance Min", "type": "number"},
    {"var_name": "threshold_max", "display_name": "Luminance Max", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    direction = inputs.get("direction", "horizontal").lower()
    threshold_min = int(inputs.get("threshold_min", 80))
    threshold_max = int(inputs.get("threshold_max", 200))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = img.copy()

    if direction == "vertical":
        for x in range(w):
            col_lum = gray[:, x]
            in_range = (col_lum >= threshold_min) & (col_lum <= threshold_max)
            # Find contiguous segments in range
            segments = _find_segments(in_range)
            for start, end in segments:
                if end > start + 1:
                    segment = result[start:end, x, :]
                    lum_segment = gray[start:end, x]
                    order = np.argsort(lum_segment)
                    result[start:end, x, :] = segment[order]
    else:
        for y in range(h):
            row_lum = gray[y, :]
            in_range = (row_lum >= threshold_min) & (row_lum <= threshold_max)
            segments = _find_segments(in_range)
            for start, end in segments:
                if end > start + 1:
                    segment = result[y, start:end, :]
                    lum_segment = gray[y, start:end]
                    order = np.argsort(lum_segment)
                    result[y, start:end, :] = segment[order]

    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}


def _find_segments(mask: np.ndarray):
    """Return list of (start, end) index pairs for contiguous True regions."""
    segments = []
    in_seg = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_seg:
            start = i
            in_seg = True
        elif not v and in_seg:
            segments.append((start, i))
            in_seg = False
    if in_seg:
        segments.append((start, len(mask)))
    return segments
