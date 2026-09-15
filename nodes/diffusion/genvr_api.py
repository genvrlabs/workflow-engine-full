"""
GenVR API node — calls GenVR's own hosted image-generation service
(FLUX, Imagen, etc.) instead of running our local SDXL pipeline.
Downloads the result and saves it locally, so it's compatible with
our other nodes (like Image to Latent).

Configuration (uid, api_key, category, subcategory, and the model's
own parameters) is collected entirely through the node's Configure
wizard on the frontend and stored as node properties — the node
itself only declares these five fields so the backend knows what
to expect, but they are never shown as raw always-visible inputs.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid

import requests
import logging

from nodes.diffusion._project_folder import get_project_folder

API_BASE = "https://api.genvrresearch.com"

metadata = {
    "display_name": "GenVR API",
    "description": "Generates content using GenVR's hosted API instead of the local pipeline.",
    "category": "diffusion",
    "color": "orange",
}

inputs = [
    {"var_name": "uid", "display_name": "Your GenVR UID", "type": "text"},
    {"var_name": "api_key", "display_name": "GenVR API Key", "type": "text"},
    {"var_name": "category", "display_name": "Category", "type": "text"},
    {"var_name": "subcategory", "display_name": "Model", "type": "text"},
    {"var_name": "params_json", "display_name": "Parameters (JSON)", "type": "text"},
]

outputs = [
    {"var_name": "image_url", "display_name": "Output", "type": "text"},
]


def get_available_models(api_key: str):
    """Fetches the live list of all models GenVR offers."""
    if not api_key:
        raise RuntimeError("A GenVR API key is required to fetch the model list.")

    resp = requests.get(
        f"{API_BASE}/api/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    return resp.json()["data"]


def get_model_schema(category: str, subcategory: str) -> dict:
    """Fetches the parameter schema for one specific model. No auth required."""
    if not category or not subcategory:
        raise RuntimeError("category and subcategory are both required.")

    resp = requests.get(f"{API_BASE}/api/schema/{category}/{subcategory}")
    resp.raise_for_status()
    return resp.json()["data"]


def _headers(api_key: str):
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


def _call_api(uid: str, api_key: str, category: str, subcategory: str, params: dict) -> str:
    outgoing_payload = {"uid": uid, "category": category, "subcategory": subcategory, **params}
    logging.getLogger(__name__).info(f"GenVR /v2/generate outgoing payload: {outgoing_payload}")
    gen_resp = requests.post(
        f"{API_BASE}/v2/generate",
        headers=_headers(api_key),
        json={"uid": uid, "category": category, "subcategory": subcategory, **params},
    )
    gen_resp.raise_for_status()
    task_id = gen_resp.json()["data"]["id"]

    # Video/audio generation genuinely takes longer than images - give
    # those categories more time before giving up, while still failing
    # fast for quicker image models if something is actually stuck.
    LONG_RUNNING_CATEGORIES = {"videogen", "audiogen"}
    max_checks = 300 if category in LONG_RUNNING_CATEGORIES else 60

    for _ in range(max_checks):
        status_resp = requests.post(
            f"{API_BASE}/v2/status",
            headers=_headers(api_key),
            json={"id": task_id, "uid": uid, "category": category, "subcategory": subcategory},
        )
        status_resp.raise_for_status()
        status = status_resp.json()["data"]["status"]

        if status == "completed":
            break
        if status == "failed":
            raise RuntimeError(f"GenVR API task failed: {task_id}")

        time.sleep(2)
    else:
        raise RuntimeError(f"GenVR API task timed out: {task_id}")

    resp_resp = requests.post(
        f"{API_BASE}/v2/response",
        headers=_headers(api_key),
        json={"id": task_id, "uid": uid, "category": category, "subcategory": subcategory},
    )
    resp_resp.raise_for_status()
    response_data = resp_resp.json()
    logging.getLogger(__name__).info(f"GenVR /v2/response raw payload: {response_data}")
    result_url = response_data["data"]["output"][0]
    if not str(result_url).startswith("http"):
        # GenVR put an error message here instead of a real URL -
        # surface it directly instead of trying to "download" it.
        raise RuntimeError(f"GenVR could not process this request: {result_url}")
    return result_url


def _download_to_local(url: str) -> str:
    folder = get_project_folder()

    # Use the real extension from the URL itself - GenVR serves images,
    # video, and audio all through this same download step.
    from urllib.parse import urlparse
    url_path = urlparse(url).path
    ext = os.path.splitext(url_path)[1] or ".png"

    file_path = os.path.join(folder, f"genvr_api_{uuid.uuid4().hex}{ext}")

    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)

    return file_path


async def execute(uid: str, token: str, inputs: dict) -> dict:
    node_uid = inputs.get("uid", "").strip()
    api_key = inputs.get("api_key", "").strip()
    category = inputs.get("category", "").strip()
    subcategory = inputs.get("subcategory", "").strip()
    params_json = inputs.get("params_json", "")
    if isinstance(params_json, str):
        params_json = params_json.strip()

    if not node_uid:
        raise ValueError("Please enter your GenVR UID using the Configure button.")
    if not api_key:
        raise ValueError("Please enter your GenVR API key using the Configure button.")
    if not category or not subcategory:
        raise ValueError("Please choose a model using the Configure button.")

    if isinstance(params_json, dict):
        params = params_json
    else:
        try:
            params = json.loads(params_json) if params_json else {}
        except json.JSONDecodeError:
            raise ValueError("Model parameters are corrupted — please reconfigure the node.")

    remote_url = await asyncio.to_thread(
        _call_api, node_uid, api_key, category, subcategory, params
    )
    local_path = await asyncio.to_thread(_download_to_local, remote_url)

    return {"image_url": local_path}

    return {"logs": list(log_buffer)}
