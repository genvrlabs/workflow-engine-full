"""Channel mixer: apply a 3x3 matrix to mix color channels."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "Channel Mixer",
    "description": "Apply a 3x3 channel mixing matrix to remap RGB channels.",
    "category": "channel",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "rr", "display_name": "R from R", "type": "number"},
    {"var_name": "rg", "display_name": "R from G", "type": "number"},
    {"var_name": "rb", "display_name": "R from B", "type": "number"},
    {"var_name": "gr", "display_name": "G from R", "type": "number"},
    {"var_name": "gg", "display_name": "G from G", "type": "number"},
    {"var_name": "gb", "display_name": "G from B", "type": "number"},
    {"var_name": "br", "display_name": "B from R", "type": "number"},
    {"var_name": "bg", "display_name": "B from G", "type": "number"},
    {"var_name": "bb", "display_name": "B from B", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    rr = float(inputs.get("rr", 1.0))
    rg = float(inputs.get("rg", 0.0))
    rb = float(inputs.get("rb", 0.0))
    gr = float(inputs.get("gr", 0.0))
    gg = float(inputs.get("gg", 1.0))
    gb = float(inputs.get("gb", 0.0))
    br = float(inputs.get("br", 0.0))
    bg = float(inputs.get("bg", 0.0))
    bb = float(inputs.get("bb", 1.0))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    img_f = img.astype(np.float32) / 255.0
    b_ch, g_ch, r_ch = img_f[:, :, 0], img_f[:, :, 1], img_f[:, :, 2]

    new_r = rr * r_ch + rg * g_ch + rb * b_ch
    new_g = gr * r_ch + gg * g_ch + gb * b_ch
    new_b = br * r_ch + bg * g_ch + bb * b_ch

    result_f = np.stack([new_b, new_g, new_r], axis=2)
    result = np.clip(result_f * 255, 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
