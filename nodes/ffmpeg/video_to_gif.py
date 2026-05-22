import subprocess
import os
import tempfile
from nodes.ffmpeg._utils import download_to_tempfile, upload_file, url_suffix

metadata = {
    "display_name": "Video to GIF",
    "description": "Converts a video URL to an optimized animated GIF and returns the GIF URL.",
    "category": "ffmpeg",
    "color": "orange",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Video URL", "type": "text"},
    {"var_name": "fps", "display_name": "Frames Per Second (default 15)", "type": "number"},
    {"var_name": "width", "display_name": "Width in pixels (default 480)", "type": "number"},
    {"var_name": "start_time", "display_name": "Start Time (optional)", "type": "text"},
    {"var_name": "duration", "display_name": "Duration (optional)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output GIF URL", "type": "text"},
]

async def execute(uid: str, token: str, inputs: dict) -> dict:
    input_url = inputs.get("input_url", "").strip()
    fps = int(inputs.get("fps", 15))
    width = int(inputs.get("width", 480))
    start_time = inputs.get("start_time", "").strip()
    duration = inputs.get("duration", "").strip()

    if not input_url:
        raise ValueError("input_url is required")

    in_ext = url_suffix(input_url) or ".mp4"
    tmp_in = download_to_tempfile(input_url, suffix=in_ext)
    tmp_palette = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp_palette.close()
    tmp_gif = tempfile.NamedTemporaryFile(delete=False, suffix=".gif")
    tmp_gif.close()

    try:
        # Step 1: generate palette
        palette_cmd = ["ffmpeg", "-y"]
        if start_time:
            palette_cmd += ["-ss", start_time]
        palette_cmd += ["-i", tmp_in]
        if duration:
            palette_cmd += ["-t", duration]
        palette_cmd += ["-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen", tmp_palette.name]

        r1 = subprocess.run(palette_cmd, capture_output=True, text=True)
        if r1.returncode != 0:
            raise RuntimeError(f"FFmpeg palette error:\n{r1.stderr}")

        # Step 2: render GIF using palette
        gif_cmd = ["ffmpeg", "-y"]
        if start_time:
            gif_cmd += ["-ss", start_time]
        gif_cmd += ["-i", tmp_in, "-i", tmp_palette.name]
        if duration:
            gif_cmd += ["-t", duration]
        gif_cmd += [
            "-lavfi", f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse",
            tmp_gif.name,
        ]

        r2 = subprocess.run(gif_cmd, capture_output=True, text=True)
        if r2.returncode != 0:
            raise RuntimeError(f"FFmpeg GIF error:\n{r2.stderr}")

        output_url = upload_file(uid, token, tmp_gif.name)
    finally:
        os.unlink(tmp_in)
        if os.path.exists(tmp_palette.name):
            os.unlink(tmp_palette.name)
        if os.path.exists(tmp_gif.name):
            os.unlink(tmp_gif.name)

    return {"output_url": output_url}
