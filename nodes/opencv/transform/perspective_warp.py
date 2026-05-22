"""Perspective warp using 4-point correspondences."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Perspective Warp",
    "description": "Warp image using 4-point perspective transform.",
    "category": "transform",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "src_points", "display_name": "Source Points (x1,y1;x2,y2;x3,y3;x4,y4)", "type": "text"},
    {"var_name": "dst_points", "display_name": "Destination Points (x1,y1;x2,y2;x3,y3;x4,y4)", "type": "text"},
    {"var_name": "out_width", "display_name": "Output Width (-1=same as input)", "type": "number"},
    {"var_name": "out_height", "display_name": "Output Height (-1=same as input)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


def _parse_points(s: str) -> np.ndarray:
    pts = []
    for pair in s.split(";"):
        xy = pair.strip().split(",")
        pts.append([float(xy[0]), float(xy[1])])
    return np.array(pts, dtype=np.float32)


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    src_pts_str = inputs.get("src_points", "")
    dst_pts_str = inputs.get("dst_points", "")
    if not src_pts_str or not dst_pts_str:
        raise ValueError("src_points and dst_points are required")
    out_w = int(inputs.get("out_width", -1))
    out_h = int(inputs.get("out_height", -1))

    img = load_image_from_url(url)
    h, w = img.shape[:2]
    if out_w <= 0:
        out_w = w
    if out_h <= 0:
        out_h = h

    src = _parse_points(src_pts_str)
    dst = _parse_points(dst_pts_str)
    if len(src) != 4 or len(dst) != 4:
        raise ValueError("src_points and dst_points must each have exactly 4 points")

    M = cv2.getPerspectiveTransform(src, dst)
    result = cv2.warpPerspective(img, M, (out_w, out_h))
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
