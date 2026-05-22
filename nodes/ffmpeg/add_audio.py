import subprocess
import os
import tempfile
from nodes.ffmpeg._utils import download_to_tempfile, upload_file, url_suffix

metadata = {
    "display_name": "Add / Replace Audio",
    "description": "Replaces a video's audio with a separate audio URL and returns the output URL.",
    "category": "ffmpeg",
    "color": "orange",
}

inputs = [
    {"var_name": "video_url", "display_name": "Input Video URL", "type": "text"},
    {"var_name": "audio_url", "display_name": "Input Audio URL", "type": "text"},
    {"var_name": "shortest", "display_name": "End at shortest stream", "type": "boolean"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Video URL", "type": "text"},
]

async def execute(uid: str, token: str, inputs: dict) -> dict:
    video_url = inputs.get("video_url", "").strip()
    audio_url = inputs.get("audio_url", "").strip()
    shortest = inputs.get("shortest", True)

    if not video_url:
        raise ValueError("video_url is required")
    if not audio_url:
        raise ValueError("audio_url is required")

    vid_ext = url_suffix(video_url) or ".mp4"
    aud_ext = url_suffix(audio_url) or ".mp3"

    tmp_vid = download_to_tempfile(video_url, suffix=vid_ext)
    tmp_aud = download_to_tempfile(audio_url, suffix=aud_ext)
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=vid_ext)
    tmp_out.close()

    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", tmp_vid,
            "-i", tmp_aud,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
        ]
        if shortest:
            cmd.append("-shortest")
        cmd.append(tmp_out.name)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error:\n{result.stderr}")

        output_url = upload_file(uid, token, tmp_out.name)
    finally:
        os.unlink(tmp_vid)
        os.unlink(tmp_aud)
        if os.path.exists(tmp_out.name):
            os.unlink(tmp_out.name)

    return {"output_url": output_url}
