import subprocess
import os
import tempfile
from nodes.ffmpeg._utils import download_to_tempfile, upload_file, url_suffix

metadata = {
    "display_name": "Add Watermark",
    "description": "Overlays an image watermark onto a video URL and returns the output URL.",
    "category": "ffmpeg",
    "color": "orange",
}

inputs = [
    {"var_name": "video_url", "display_name": "Input Video URL", "type": "text"},
    {"var_name": "watermark_url", "display_name": "Watermark Image URL", "type": "text"},
    {"var_name": "position", "display_name": "Position (topleft/topright/bottomleft/bottomright/center)", "type": "text"},
    {"var_name": "opacity", "display_name": "Opacity 0.0–1.0 (default 1.0)", "type": "number"},
    {"var_name": "margin", "display_name": "Margin in pixels (default 10)", "type": "number"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Video URL", "type": "text"},
]

_POSITIONS = {
    "topleft":     "x={m}:y={m}",
    "topright":    "x=W-w-{m}:y={m}",
    "bottomleft":  "x={m}:y=H-h-{m}",
    "bottomright": "x=W-w-{m}:y=H-h-{m}",
    "center":      "x=(W-w)/2:y=(H-h)/2",
}

async def execute(uid: str, token: str, inputs: dict) -> dict:
    video_url = inputs.get("video_url", "").strip()
    watermark_url = inputs.get("watermark_url", "").strip()
    position = inputs.get("position", "bottomright").strip().lower()
    opacity = float(inputs.get("opacity", 1.0))
    margin = int(inputs.get("margin", 10))

    if not video_url:
        raise ValueError("video_url is required")
    if not watermark_url:
        raise ValueError("watermark_url is required")
    if position not in _POSITIONS:
        raise ValueError(f"position must be one of: {', '.join(_POSITIONS)}")

    vid_ext = url_suffix(video_url) or ".mp4"
    wm_ext = url_suffix(watermark_url) or ".png"

    tmp_vid = download_to_tempfile(video_url, suffix=vid_ext)
    tmp_wm = download_to_tempfile(watermark_url, suffix=wm_ext)
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=vid_ext)
    tmp_out.close()

    try:
        pos_expr = _POSITIONS[position].format(m=margin)
        wm_filter = f"[1:v]format=rgba,colorchannelmixer=aa={opacity}[wm];[0:v][wm]overlay={pos_expr}"

        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_vid,
            "-i", tmp_wm,
            "-filter_complex", wm_filter,
            "-c:a", "copy",
            tmp_out.name,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error:\n{result.stderr}")

        output_url = upload_file(uid, token, tmp_out.name)
    finally:
        os.unlink(tmp_vid)
        os.unlink(tmp_wm)
        if os.path.exists(tmp_out.name):
            os.unlink(tmp_out.name)

    return {"output_url": output_url}
