import subprocess
import os
import tempfile
from nodes.ffmpeg._utils import download_to_tempfile, upload_file, url_suffix

metadata = {
    "display_name": "Trim Video",
    "description": "Trims a video URL to a specified time range and returns the output URL.",
    "category": "ffmpeg",
    "color": "orange",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Video URL", "type": "text"},
    {"var_name": "start_time", "display_name": "Start Time (HH:MM:SS or seconds)", "type": "text"},
    {"var_name": "duration", "display_name": "Duration (HH:MM:SS or seconds)", "type": "text"},
    {"var_name": "end_time", "display_name": "End Time (overrides duration)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Video URL", "type": "text"},
]

async def execute(uid: str, token: str, inputs: dict) -> dict:
    input_url = inputs.get("input_url", "").strip()
    start_time = inputs.get("start_time", "").strip()
    duration = inputs.get("duration", "").strip()
    end_time = inputs.get("end_time", "").strip()

    if not input_url:
        raise ValueError("input_url is required")
    if not start_time and not end_time and not duration:
        raise ValueError("At least one of start_time, end_time, or duration is required")

    ext = url_suffix(input_url) or ".mp4"
    tmp_in = download_to_tempfile(input_url, suffix=ext)
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp_out.close()

    try:
        cmd = ["ffmpeg", "-y"]
        if start_time:
            cmd += ["-ss", start_time]
        cmd += ["-i", tmp_in]
        if end_time:
            cmd += ["-to", end_time]
        elif duration:
            cmd += ["-t", duration]
        cmd += ["-c", "copy", tmp_out.name]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error:\n{result.stderr}")

        output_url = upload_file(uid, token, tmp_out.name)
    finally:
        os.unlink(tmp_in)
        if os.path.exists(tmp_out.name):
            os.unlink(tmp_out.name)

    return {"output_url": output_url}
