"""HDR tonemapping using Drago or Reinhard operator."""
import cv2
import numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload

metadata = {
    "display_name": "HDR Tonemap",
    "description": "Apply HDR tonemapping (Drago) for cinematic color grading.",
    "category": "style",
    "color": "blue",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Image URL", "type": "text"},
    {"var_name": "gamma", "display_name": "Gamma", "type": "number"},
    {"var_name": "saturation", "display_name": "Saturation", "type": "number"},
    {"var_name": "bias", "display_name": "Bias (Drago)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Image URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    url = inputs.get("input_url", "")
    if not url:
        raise ValueError("input_url is required")
    gamma = float(inputs.get("gamma", 1.0))
    saturation = float(inputs.get("saturation", 1.0))
    bias = float(inputs.get("bias", 0.85))

    img = load_image_from_url(url, cv2.IMREAD_COLOR)
    hdr = img.astype(np.float32) / 255.0

    tonemap = cv2.createTonemapDrago(gamma=gamma, saturation=saturation, bias=bias)
    tonemapped = tonemap.process(hdr)
    result = np.clip(tonemapped * 255, 0, 255).astype(np.uint8)
    output_url = save_and_upload(result, uid, token, ".png")
    return {"output_url": output_url}
