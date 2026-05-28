"""Split an image into three equal vertical strips and upload each to GenVR."""

import json

from nodes.opencv._utils import load_image_from_url, save_and_upload

_NODE = "split_image_thirds"


def _debug_console(label: str, data) -> None:
    print(f"[{_NODE}] {label}:\n{json.dumps(data, default=str, indent=2)}", flush=True)

metadata = {
    "display_name": "Split Image Thirds",
    "description": "Cuts an image into three equal vertical parts (full height) and uploads each to GenVR.",
    "category": "core",
    "color": "teal",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
]

outputs = [
    {"var_name": "left_url", "display_name": "Left Third URL", "type": "text"},
    {"var_name": "center_url", "display_name": "Center Third URL", "type": "text"},
    {"var_name": "right_url", "display_name": "Right Third URL", "type": "text"},
    {
        "var_name": "parts",
        "display_name": "Parts [{name, uri, type}, ...]",
        "type": "list",
    },
]


def _asset(name: str, uri: str, mime: str = "image/png") -> dict:
    return {"name": name, "uri": uri, "type": mime}


async def execute(uid: str, token: str, inputs: dict) -> dict:
    _debug_console("inputs", inputs)

    url = str(inputs.get("input_url", "")).strip()
    if not url:
        raise ValueError("input_url is required")

    img = load_image_from_url(url)
    width = img.shape[1]
    if width < 3:
        raise ValueError("image width must be at least 3 pixels")

    third = width // 3
    left = img[:, 0:third]
    center = img[:, third : 2 * third]
    right = img[:, 2 * third : width]

    left_url = save_and_upload(left, uid, token, ".png")
    center_url = save_and_upload(center, uid, token, ".png")
    right_url = save_and_upload(right, uid, token, ".png")
    parts = [
        _asset("left", left_url),
        _asset("center", center_url),
        _asset("right", right_url),
    ]
    outputs = {
        "left_url": left_url,
        "center_url": center_url,
        "right_url": right_url,
        "parts": parts,
    }
    _debug_console("outputs", outputs)
    return outputs
