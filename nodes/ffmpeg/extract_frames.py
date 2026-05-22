import subprocess
import os
import tempfile
from nodes.ffmpeg._utils import download_to_tempfile, upload_file, url_suffix

metadata = {
    "display_name": "Extract Frames",
    "description": "Extracts frames from a video URL and returns a list of image URLs.",
    "category": "ffmpeg",
    "color": "orange",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Video URL", "type": "text"},
    {"var_name": "fps", "display_name": "Frames Per Second (e.g. 1, 0.5, 24)", "type": "number"},
    {"var_name": "image_format", "display_name": "Image Format (png or jpg)", "type": "text"},
    {"var_name": "start_time", "display_name": "Start Time (optional)", "type": "text"},
    {"var_name": "duration", "display_name": "Duration (optional)", "type": "text"},
]

outputs = [
    {"var_name": "frame_urls", "display_name": "List of Frame URLs", "type": "list"},
    {"var_name": "frame_count", "display_name": "Number of Frames", "type": "number"},
]

async def execute(uid: str, token: str, inputs: dict) -> dict:
    input_url = inputs.get("input_url", "").strip()
    fps = float(inputs.get("fps", 1))
    img_fmt = inputs.get("image_format", "png").strip().lstrip(".")
    start_time = inputs.get("start_time", "").strip()
    duration = inputs.get("duration", "").strip()

    if not input_url:
        raise ValueError("input_url is required")
    if img_fmt not in ("png", "jpg", "jpeg"):
        raise ValueError("image_format must be png or jpg")

    in_ext = url_suffix(input_url) or ".mp4"
    tmp_in = download_to_tempfile(input_url, suffix=in_ext)
    tmp_dir = tempfile.mkdtemp()

    try:
        pattern = os.path.join(tmp_dir, f"frame_%04d.{img_fmt}")
        cmd = ["ffmpeg", "-y"]
        if start_time:
            cmd += ["-ss", start_time]
        cmd += ["-i", tmp_in]
        if duration:
            cmd += ["-t", duration]
        cmd += ["-vf", f"fps={fps}", pattern]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error:\n{result.stderr}")

        frame_files = sorted(
            os.path.join(tmp_dir, f)
            for f in os.listdir(tmp_dir)
            if f.endswith(f".{img_fmt}")
        )

        frame_urls = []
        for frame_path in frame_files:
            url = upload_file(uid, token, frame_path)
            frame_urls.append(url)
    finally:
        os.unlink(tmp_in)
        for f in os.listdir(tmp_dir):
            os.unlink(os.path.join(tmp_dir, f))
        os.rmdir(tmp_dir)

    return {"frame_urls": frame_urls, "frame_count": len(frame_urls)}
