"""Shared helpers for OpenCV nodes: download image from URL, upload result."""
import os
import tempfile
import urllib.request
import urllib.error

import numpy as np
import cv2

from nodes.ffmpeg._utils import upload_file, url_suffix
from nodes.input_utils import resolve_asset, resolve_url

resolve_image_url = resolve_url


def load_image_from_url(url: str, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    ext = url_suffix(url) or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.close()
    req = urllib.request.Request(url, headers={"User-Agent": "GenVR-OpenCV-Node/1.0"})
    try:
        with urllib.request.urlopen(req) as resp, open(tmp.name, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
    except urllib.error.URLError as e:
        os.unlink(tmp.name)
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    img = cv2.imread(tmp.name, flags)
    os.unlink(tmp.name)
    if img is None:
        raise RuntimeError(f"cv2.imread could not decode image from {url}")
    return img


def save_and_upload(img: np.ndarray, uid: str, token: str, ext: str = ".png") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.close()
    cv2.imwrite(tmp.name, img)
    try:
        url = upload_file(uid, token, tmp.name)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
    return url


def parse_color(s: str, default=(0, 0, 0)) -> tuple:
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except Exception:
        return default


def odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def ensure_bgra(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img
