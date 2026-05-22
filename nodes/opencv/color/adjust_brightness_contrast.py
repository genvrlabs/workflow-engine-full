import cv2, numpy as np
from nodes.opencv._utils import load_image_from_url, save_and_upload, ensure_bgr
metadata = {"display_name":"Brightness / Contrast","description":"Adjusts brightness and contrast.","category":"color","color":"blue"}
inputs  = [{"var_name":"input_url","display_name":"Input Image URL","type":"text"},{"var_name":"brightness","display_name":"Brightness (-100 to 100)","type":"number"},{"var_name":"contrast","display_name":"Contrast (-100 to 100)","type":"number"}]
outputs = [{"var_name":"output_url","display_name":"Output Image URL","type":"text"}]
async def execute(uid, token, inputs):
    url = inputs.get("input_url","").strip()
    if not url: raise ValueError("input_url is required")
    out = cv2.convertScaleAbs(ensure_bgr(load_image_from_url(url)), alpha=1.0+float(inputs.get("contrast",0))/100, beta=float(inputs.get("brightness",0)))
    return {"output_url": save_and_upload(out, uid, token)}
