"""Stitch left, center, and right image URLs into one horizontal image."""

import json
from typing import Any

import cv2
import numpy as np

from nodes.opencv._utils import ensure_bgr, load_image_from_url, save_and_upload

_NODE = "stitch_image_thirds"


def _debug_console(label: str, data) -> None:
    print(f"[{_NODE}] {label}:\n{json.dumps(data, default=str, indent=2)}", flush=True)


metadata = {
    "display_name": "Stitch Image Thirds",
    "description": "Joins left, center, and right image URLs side-by-side (full height) and uploads to GenVR.",
    "category": "core",
    "color": "teal",
}

inputs = [
    {
        "var_name": "parts",
        "display_name": "Parts [{name, uri, type}, ...]",
        "type": "list",
        "batch": False,
    },
    {"var_name": "left_url", "display_name": "Left ({name, uri, type} or URL)", "type": "any"},
    {"var_name": "center_url", "display_name": "Center ({name, uri, type} or URL)", "type": "any"},
    {"var_name": "right_url", "display_name": "Right ({name, uri, type} or URL)", "type": "any"},
    {
        "var_name": "parts_json",
        "display_name": "Parts ([{name, uri, type}, ...] or JSON string)",
        "type": "any",
        "batch": False,
    },
]

outputs = [
    {"var_name": "output_url", "display_name": "Stitched Image URL", "type": "text"},
]

_KEY_TRIPLES = (
    ("left_url", "center_url", "right_url"),
    ("left", "center", "right"),
)
_URI_KEYS = ("uri", "url", "href")
_NAME_ORDER = ("left", "center", "centre", "middle", "right")


def _strip_url(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _urls_from_dict(data: dict) -> list[str]:
    for left_k, center_k, right_k in _KEY_TRIPLES:
        if left_k in data and center_k in data and right_k in data:
            return [
                _url_from_input(data[left_k]),
                _url_from_input(data[center_k]),
                _url_from_input(data[right_k]),
            ]
    raise ValueError(
        "JSON object must include left_url, center_url, right_url "
        "(or left, center, right)"
    )


def _uri_from_item(item: Any) -> str:
    if isinstance(item, str):
        return _strip_url(item)
    if isinstance(item, dict):
        for key in _URI_KEYS:
            if key in item and item[key] not in (None, ""):
                return _strip_url(item[key])
    return ""


def _url_from_input(value: Any) -> str:
    """URL string or {name, uri, type} asset object."""
    if isinstance(value, dict):
        return _uri_from_item(value)
    return _strip_url(value)


def _name_from_item(item: dict) -> str:
    for key in ("name", "label", "id", "title"):
        if key in item and item[key] not in (None, ""):
            return str(item[key]).strip().lower()
    return ""


def _name_sort_index(name: str) -> int:
    for i, token in enumerate(_NAME_ORDER):
        if token in name:
            return i
    return 99


def _is_asset(item: Any) -> bool:
    return isinstance(item, dict) and bool(_uri_from_item(item))


def _looks_like_asset_list(items: list) -> bool:
    return sum(1 for item in items if _is_asset(item)) >= 3


def _urls_from_asset_list(items: list) -> list[str]:
    assets = [item for item in items if isinstance(item, dict) and _uri_from_item(item)]
    if len(assets) < 3:
        raise ValueError("parts must include at least 3 items with uri (or url)")

    named = [(_name_from_item(item), _uri_from_item(item)) for item in assets]
    indices = [_name_sort_index(name) for name, _ in named]

    if len(assets) == 3 and all(i < 99 for i in indices) and len(set(indices)) == 3:
        ordered = sorted(zip(indices, named), key=lambda pair: pair[0])
        return [uri for _, (_, uri) in ordered]

    urls = [_uri_from_item(item) for item in items if _uri_from_item(item)]
    if len(urls) < 3:
        raise ValueError("parts must include at least 3 items with uri (or url)")
    if len(urls) > 3:
        urls = urls[:3]
    if len(urls) != 3:
        raise ValueError("parts array must have exactly 3 URLs: left, center, right")
    return urls


def _urls_from_list(items: list) -> list[str]:
    if _looks_like_asset_list(items):
        return _urls_from_asset_list(items)
    urls = [_strip_url(u) for u in items if _strip_url(u)]
    if len(urls) != 3:
        raise ValueError("parts array must have exactly 3 URLs: left, center, right")
    return urls


def _maybe_parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _urls_from_sequence(items: list) -> list[str]:
    """Parse a list of URL strings and/or {name, uri, type} asset objects."""
    cleaned: list[Any] = []
    for item in items:
        if item is None or item == "":
            continue
        cleaned.append(_maybe_parse_json(item))

    if not cleaned:
        raise ValueError("parts array is empty")

    if len(cleaned) == 1:
        only = cleaned[0]
        if isinstance(only, (list, tuple)):
            return _urls_from_sequence(list(only))
        if isinstance(only, dict):
            try:
                return _urls_from_dict(only)
            except ValueError:
                pass

    if _looks_like_asset_list(cleaned):
        return _urls_from_asset_list(cleaned)

    urls: list[str] = []
    for item in cleaned:
        if isinstance(item, dict):
            try:
                return _urls_from_dict(item)
            except ValueError:
                u = _uri_from_item(item)
                if u:
                    urls.append(u)
                    continue
        elif isinstance(item, (list, tuple)):
            return _urls_from_sequence(list(item))
        else:
            u = _strip_url(item)
            if u:
                urls.append(u)

    return _urls_from_list(urls)


def _normalize_parts_raw(raw: Any) -> list[str] | None:
    if raw is None or raw == "" or raw == []:
        return None

    raw = _maybe_parse_json(raw)

    if isinstance(raw, dict):
        if _is_asset(raw):
            raise ValueError("expected 3 parts [{name, uri, type}, ...], got a single asset")
        try:
            return _urls_from_dict(raw)
        except ValueError:
            return None

    if isinstance(raw, (list, tuple)):
        items = list(raw)
        if len(items) == 1:
            only = _maybe_parse_json(items[0])
            if isinstance(only, (list, tuple, dict)):
                return _normalize_parts_raw(only)
        try:
            return _urls_from_sequence(items)
        except ValueError:
            return None

    if _strip_url(raw):
        return None

    return None


def _urls_from_individual(inputs: dict) -> list[str] | None:
    left = inputs.get("left_url")
    center = inputs.get("center_url")
    right = inputs.get("right_url")

    urls = [_url_from_input(left), _url_from_input(center), _url_from_input(right)]
    if all(urls):
        return urls

    # Whole array or asset list wired to one of the three inputs.
    for slot in (left, center, right):
        if isinstance(slot, str) and _strip_url(slot) and slot.strip()[0] not in "[{":
            continue
        normalized = _normalize_parts_raw(slot)
        if normalized:
            return normalized

    return None


def _parse_parts(inputs: dict) -> list[str]:
    for key in ("parts", "parts_json"):
        urls = _normalize_parts_raw(inputs.get(key))
        if urls:
            return urls

    urls = _urls_from_individual(inputs)
    if urls:
        return urls

    raise ValueError(
        "Provide parts as [{name, uri, type}, ...] (3 items), "
        "or left/center/right inputs, or parts_json"
    )


def _align_heights(images: list[np.ndarray]) -> list[np.ndarray]:
    max_h = max(img.shape[0] for img in images)
    aligned = []
    for img in images:
        h, w = img.shape[:2]
        if h == max_h:
            aligned.append(img)
            continue
        new_w = max(1, int(round(w * max_h / h)))
        aligned.append(cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_LINEAR))
    return aligned


async def execute(uid: str, token: str, inputs: dict) -> dict:
    _debug_console("inputs", inputs)

    urls = _parse_parts(inputs)
    _debug_console("resolved_uris", urls)
    if not all(urls):
        raise ValueError("all three part URLs must be non-empty")

    images = [ensure_bgr(load_image_from_url(url)) for url in urls]
    stitched = cv2.hconcat(_align_heights(images))
    output_url = save_and_upload(stitched, uid, token, ".png")
    outputs = {"output_url": output_url}
    _debug_console("outputs", outputs)
    return outputs
