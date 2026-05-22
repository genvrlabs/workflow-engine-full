import subprocess
import os
import tempfile
from nodes.ffmpeg._utils import download_to_tempfile, upload_file, url_suffix

metadata = {
    "display_name": "Convert Video",
    "description": "Converts a video URL to a different format and returns the output URL.",
    "category": "ffmpeg",
    "color": "orange",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Video URL", "type": "text"},
    {"var_name": "output_extension", "display_name": "Output Extension (e.g. mp4, webm, mov)", "type": "text"},
    {"var_name": "video_codec", "display_name": "Video Codec (e.g. libx264)", "type": "text"},
    {"var_name": "audio_codec", "display_name": "Audio Codec (e.g. aac)", "type": "text"},
    {"var_name": "extra_args", "display_name": "Extra FFmpeg Args", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Video URL", "type": "text"},
]

async def execute(uid: str, token: str, inputs: dict) -> dict:
    input_url = inputs.get("input_url", "").strip()
    output_ext = inputs.get("output_extension", "mp4").strip().lstrip(".")
    video_codec = inputs.get("video_codec", "").strip()
    audio_codec = inputs.get("audio_codec", "").strip()
    extra_args = inputs.get("extra_args", "").strip()

    if not input_url:
        raise ValueError("input_url is required")

    in_ext = url_suffix(input_url) or ".mp4"
    tmp_in = download_to_tempfile(input_url, suffix=in_ext)
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_ext}")
    tmp_out.close()

    try:
        cmd = ["ffmpeg", "-y", "-i", tmp_in]
        if video_codec:
            cmd += ["-vcodec", video_codec]
        if audio_codec:
            cmd += ["-acodec", audio_codec]
        if extra_args:
            cmd += extra_args.split()
        cmd.append(tmp_out.name)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error:\n{result.stderr}")

        output_url = upload_file(uid, token, tmp_out.name)
    finally:
        os.unlink(tmp_in)
        if os.path.exists(tmp_out.name):
            os.unlink(tmp_out.name)

    return {"output_url": output_url}
