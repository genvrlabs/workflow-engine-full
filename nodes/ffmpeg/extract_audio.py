import subprocess
import os
import tempfile
import logging

from nodes.ffmpeg._utils import download_to_tempfile, upload_file, url_suffix

log = logging.getLogger(__name__)

metadata = {
    "display_name": "Extract Audio",
    "description": "Extracts the audio track from a video URL and returns the audio URL.",
    "category": "ffmpeg",
    "color": "orange",
}

inputs = [
    {"var_name": "input_url", "display_name": "Input Video URL", "type": "text"},
    {"var_name": "audio_codec", "display_name": "Audio Codec (e.g. mp3, aac)", "type": "text"},
    {"var_name": "bitrate", "display_name": "Bitrate (e.g. 192k)", "type": "text"},
    {"var_name": "output_extension", "display_name": "Output Extension (e.g. mp3, wav, aac)", "type": "text"},
]

outputs = [
    {"var_name": "output_url", "display_name": "Output Audio URL", "type": "text"},
]

async def execute(uid: str, token: str, inputs: dict) -> dict:
    input_url = inputs.get("input_url", "").strip()
    audio_codec = inputs.get("audio_codec", "").strip()
    bitrate = inputs.get("bitrate", "").strip()
    out_ext = inputs.get("output_extension", "mp3").strip().lstrip(".")

    log.info("[extract_audio] inputs: url=%s codec=%r bitrate=%r out_ext=%r",
             input_url, audio_codec, bitrate, out_ext)

    if not input_url:
        raise ValueError("input_url is required")

    in_ext = url_suffix(input_url) or ".mp4"
    log.info("[extract_audio] downloading input (detected ext: %s)", in_ext)
    tmp_in = download_to_tempfile(input_url, suffix=in_ext)
    log.info("[extract_audio] downloaded to %s  size=%d bytes",
             tmp_in, os.path.getsize(tmp_in))

    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=f".{out_ext}")
    tmp_out.close()
    log.info("[extract_audio] output temp file: %s", tmp_out.name)

    try:
        cmd = ["ffmpeg", "-y", "-i", tmp_in, "-vn"]
        if audio_codec:
            cmd += ["-acodec", audio_codec]
        if bitrate:
            cmd += ["-ab", bitrate]
        cmd.append(tmp_out.name)

        log.info("[extract_audio] running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        log.info("[extract_audio] ffmpeg exit code: %d", result.returncode)
        if result.stdout:
            log.debug("[extract_audio] ffmpeg stdout:\n%s", result.stdout)
        if result.stderr:
            log.info("[extract_audio] ffmpeg stderr:\n%s", result.stderr)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error (exit {result.returncode}):\n{result.stderr}")

        out_size = os.path.getsize(tmp_out.name)
        log.info("[extract_audio] output file size: %d bytes", out_size)
        if out_size == 0:
            raise RuntimeError("FFmpeg produced an empty output file — the input may have no audio stream")

        log.info("[extract_audio] uploading to GenVR...")
        output_url = upload_file(uid, token, tmp_out.name)
        log.info("[extract_audio] upload complete: %s", output_url)
    finally:
        os.unlink(tmp_in)
        if os.path.exists(tmp_out.name):
            os.unlink(tmp_out.name)

    return {"output_url": output_url}
