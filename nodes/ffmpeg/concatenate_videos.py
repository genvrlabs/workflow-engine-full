import subprocess
import os
import tempfile
from nodes.ffmpeg._utils import download_to_tempfile, upload_file, url_suffix

metadata = {
    "display_name": "Concatenate Videos",
    "description": "Joins a list of video URLs end-to-end and returns the output URL.",
    "category": "ffmpeg",
    "color": "orange",
}

inputs = [
    {
        "var_name": "input_urls",
        "display_name": "Input Video URLs (list of strings)",
        "type": "list",
        "batch": False,
    },
    {"var_name": "output_extension", "display_name": "Output Extension (e.g. mp4)", "type": "text"},
    {"var_name": "reencode", "display_name": "Re-encode (required for mixed codecs)", "type": "boolean"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Video URL", "type": "text"},
]

async def execute(uid: str, token: str, inputs: dict) -> dict:
    input_urls = inputs.get("input_urls", [])
    out_ext = inputs.get("output_extension", "mp4").strip().lstrip(".")
    reencode = inputs.get("reencode", False)

    if not input_urls:
        raise ValueError("input_urls list is required and must not be empty")

    tmp_inputs = []
    for url in input_urls:
        ext = url_suffix(url) or ".mp4"
        tmp_inputs.append(download_to_tempfile(url, suffix=ext))

    concat_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    for p in tmp_inputs:
        safe_path = p.replace("\\", "/").replace("'", "\\'")
        concat_file.write(f"file '{safe_path}'\n")
    concat_file.close()

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=f".{out_ext}")
    tmp_out.close()

    try:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file.name]
        if reencode:
            cmd += ["-c:v", "libx264", "-c:a", "aac"]
        else:
            cmd += ["-c", "copy"]
        cmd.append(tmp_out.name)

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error:\n{result.stderr}")

        output_url = upload_file(uid, token, tmp_out.name)
    finally:
        for p in tmp_inputs:
            os.unlink(p)
        os.unlink(concat_file.name)
        if os.path.exists(tmp_out.name):
            os.unlink(tmp_out.name)

    return {"output_url": output_url}
