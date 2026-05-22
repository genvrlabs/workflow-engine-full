import subprocess
import json
import os
import tempfile
from nodes.ffmpeg._utils import download_to_tempfile, url_suffix

metadata = {
    "display_name": "Get Video Info",
    "description": "Returns metadata about a video URL (duration, codec, resolution, fps, etc.).",
    "category": "ffmpeg",
    "color": "orange",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Video URL", "type": "text"},
]

outputs = [
    {"var_name": "duration", "display_name": "Duration (seconds)", "type": "number"},
    {"var_name": "width", "display_name": "Width (px)", "type": "number"},
    {"var_name": "height", "display_name": "Height (px)", "type": "number"},
    {"var_name": "fps", "display_name": "Frames Per Second", "type": "number"},
    {"var_name": "video_codec", "display_name": "Video Codec", "type": "text"},
    {"var_name": "audio_codec", "display_name": "Audio Codec", "type": "text"},
    {"var_name": "size_bytes", "display_name": "File Size (bytes)", "type": "number"},
    {"var_name": "info", "display_name": "Full Probe Info (dict)", "type": "dict"},
]

async def execute(uid: str, token: str, inputs: dict) -> dict:
    input_url = inputs.get("input_url", "").strip()
    if not input_url:
        raise ValueError("input_url is required")

    ext = url_suffix(input_url) or ".mp4"
    tmp = download_to_tempfile(input_url, suffix=ext)
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            tmp,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe error:\n{result.stderr}")

        probe = json.loads(result.stdout)
    finally:
        os.unlink(tmp)

    streams = probe.get("streams", [])
    fmt = probe.get("format", {})
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

    fps = 0.0
    raw_fps = video_stream.get("avg_frame_rate", "0/1")
    if "/" in raw_fps:
        num, den = raw_fps.split("/")
        fps = round(int(num) / int(den), 3) if int(den) else 0.0

    return {
        "duration": float(fmt.get("duration", 0)),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": fps,
        "video_codec": video_stream.get("codec_name", ""),
        "audio_codec": audio_stream.get("codec_name", ""),
        "size_bytes": int(fmt.get("size", 0)),
        "info": probe,
    }
