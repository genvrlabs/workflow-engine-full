"""Shared helpers for FFmpeg nodes: download from URL, upload to GenVR."""

import os
import mimetypes
import tempfile
import urllib.request
import urllib.parse
import urllib.error
import json
import http.client

_UPLOAD_SAS_ENDPOINT = "https://api.genvrresearch.com/api/upload/sas-url"


def download_to_tempfile(url: str, suffix: str = "") -> str:
    """Download *url* into a named temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    req = urllib.request.Request(url, headers={"User-Agent": "GenVR-FFmpeg-Node/1.0"})
    try:
        with urllib.request.urlopen(req) as resp, open(tmp.name, "wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
    except urllib.error.URLError as e:
        os.unlink(tmp.name)
        raise RuntimeError(f"Failed to download {url}: {e}") from e
    return tmp.name


def _get_sas_url(uid: str, token: str, file_name: str, category: str = "imagegen") -> tuple[str, str]:
    """Request a SAS upload URL from GenVR. Returns (upload_url, blob_url)."""
    payload = json.dumps({
        "userId": uid,
        "fileName": file_name,
        "category": category,
    }).encode()

    req = urllib.request.Request(
        _UPLOAD_SAS_ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"SAS URL request failed ({e.code}): {body}") from e

    if not data.get("success"):
        raise RuntimeError(f"SAS URL request unsuccessful: {data}")

    upload_url = data["uploadUrl"]
    # Build the blob URL with the SAS token from the uploadUrl so the
    # private container remains accessible to the caller.
    account = data["account"]
    container = data["container"]
    blob_path = data["blobPath"]
    sas_token = urllib.parse.urlparse(upload_url).query
    blob_url = f"https://{account}.blob.core.windows.net/{container}/{blob_path}?{sas_token}"
    return upload_url, blob_url


def upload_file(uid: str, token: str, local_path: str, category: str = "imagegen") -> str:
    """Upload *local_path* to GenVR blob storage and return the public blob URL."""
    file_name = os.path.basename(local_path)
    upload_url, blob_url = _get_sas_url(uid, token, file_name, category)

    mime_type, _ = mimetypes.guess_type(file_name)
    mime_type = mime_type or "application/octet-stream"

    with open(local_path, "rb") as f:
        file_data = f.read()

    parsed = urllib.parse.urlparse(upload_url)
    # Use http.client so we can PUT without redirect issues
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parsed.netloc)
    path_and_query = parsed.path + (f"?{parsed.query}" if parsed.query else "")
    conn.request(
        "PUT",
        path_and_query,
        body=file_data,
        headers={
            "x-ms-blob-type": "BlockBlob",
            "Content-Type": mime_type,
            "Content-Length": str(len(file_data)),
        },
    )
    resp = conn.getresponse()
    resp.read()
    conn.close()

    if resp.status not in (200, 201):
        raise RuntimeError(f"Blob upload failed with HTTP {resp.status}")

    return blob_url


def url_suffix(url: str) -> str:
    """Return the file extension from a URL path, e.g. '.mp4'."""
    path = urllib.parse.urlparse(url).path
    _, ext = os.path.splitext(path)
    return ext or ""
