"""
Uploads a local image or video file to GenVR's own storage and
returns a real, publicly-fetchable URL - directly solves the
recurring problem of GenVR API nodes needing a real hosted URL, not
a local file path (the thing we kept hitting all day, hunting for
Wikimedia/imgur links by hand).

This reuses the existing upload_file(uid, token, local_path) helper
from nodes/ffmpeg/_utils.py, documented in the README's "GenVR file
upload" section. Please double check that helper's exact parameter
order in your actual file before relying on this - it's built from
the documented interface, not a direct read of that file in this
session.
"""

from nodes.ffmpeg._utils import upload_file

metadata = {
    "display_name": "Upload to GenVR",
    "description": "Uploads a local image or video to GenVR's storage and returns a real URL other GenVR nodes can use directly.",
    "category": "genvr",
    "color": "purple",
}

inputs = [
    {"var_name": "image_path", "display_name": "File", "type": "text"},
]

outputs = [
    {"var_name": "url", "display_name": "GenVR URL", "type": "text"},
]


async def execute(uid: str, token: str, inputs: dict) -> dict:
    local_path = inputs.get("image_path", "").strip()
    if not local_path:
        raise ValueError("Please choose a file to upload first.")
    url = upload_file(uid, token, local_path)
    return {"url": url}