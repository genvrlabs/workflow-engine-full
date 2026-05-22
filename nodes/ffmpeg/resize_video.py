import subprocess
import os
import tempfile
from nodes.ffmpeg._utils import download_to_tempfile, upload_file, url_suffix

metadata = {
    "display_name": "Resize Video",
    "description": "Scales a video URL to a specified resolution and returns the output URL.",
    "category": "ffmpeg",
    "color": "orange",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Video URL", "type": "text"},
    {"var_name": "width", "display_name": "Width px (-1 = keep aspect ratio)", "type": "number"},
    {"var_name": "height", "display_name": "Height px (-1 = keep aspect ratio)", "type": "number"},
    {"var_name": "video_codec", "display_name": "Video Codec (default libx264)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Video URL", "type": "text"},
]

async def execute(uid: str, token: str, inputs: dict) -> dict:
    input_url = inputs.get("input_url", "").strip()
    width = int(inputs.get("width", -1))
    height = int(inputs.get("height", -1))
    video_codec = inputs.get("video_codec", "libx264").strip()

    if not input_url:
        raise ValueError("input_url is required")
    if width == -1 and height == -1:
        raise ValueError("At least one of width or height must be set (not both -1)")

    ext = url_suffix(input_url) or ".mp4"
    tmp_in = download_to_tempfile(input_url, suffix=ext)
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp_out.close()

    try:
        cmd = [
            "ffmpeg", "-y", "-i", tmp_in,
            "-vf", f"scale={width}:{height}",
            "-vcodec", video_codec,
            tmp_out.name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error:\n{result.stderr}")

        output_url = upload_file(uid, token, tmp_out.name)
    finally:
        os.unlink(tmp_in)
        if os.path.exists(tmp_out.name):
            os.unlink(tmp_out.name)

    return {"output_url": output_url}
