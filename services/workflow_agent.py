"""
Workflow-designer agent: takes a plain-English request and a real node
catalog, calls GenVR's LLM completions endpoint, and returns the raw
text the model replied with (expected to be workflow JSON).

This endpoint is a genuinely different shape from GenVR's other APIs
(image/video/3D generation) - it's a single synchronous streaming call,
not the /v2/generate -> /v2/status -> /v2/response polling pattern used
in nodes/diffusion/genvr_api.py. Confirmed working route (found via
manual testing, 2026-09-13): POST to LLM_BASE + "/api/generate-non-polling",
with "stream": true, returns Server-Sent Events (SSE) - each line is
either "data: {json chunk}" or the literal "data: [DONE]" to end the
stream. Each chunk's real answer text lives at
chunk["choices"][0]["delta"]["content"]; a "reasoning" field is also
present on some chunks (the model's internal scratchpad) - not part of
the final answer, discard it.
"""

import json
import logging
import requests

from nodes.registry import list_nodes

LLM_BASE = "https://llm-genvr-fea4emaebhchbdbw.eastus-01.azurewebsites.net"

log = logging.getLogger(__name__)


def _build_node_catalog_summary() -> str:
    """
    Condenses the full node registry into a compact text block the LLM
    can read as part of its instructions - full inputs/outputs per node
    type, so it knows valid var_names for wiring.
    """
    lines = []
    for node in list_nodes():
        node_type = node["node_type"]
        inputs = ", ".join(f'{i["var_name"]}:{i["type"]}' for i in node["inputs"])
        outputs = ", ".join(f'{o["var_name"]}:{o["type"]}' for o in node["outputs"])
        lines.append(f"- {node_type} | inputs: [{inputs}] | outputs: [{outputs}]")
    return "\n".join(lines)


def _build_system_prompt() -> str:
    catalog = _build_node_catalog_summary()
    return f"""You are a workflow-designer agent. Given a plain-English
request, output ONLY a JSON object (no prose, no markdown fences)
matching this exact schema:

{{
  "name": "string",
  "nodes": [
    {{"id": "node-1", "type": "<a node_type from the catalog below>", "data": {{...}}}}
  ],
  "edges": [
    {{"source": "node-1", "target": "node-2", "sourceHandle": "<output var_name>", "targetHandle": "<input var_name>"}}
  ],
  "positions": {{"node-1": {{"x": 0, "y": 0}}}}
}}

Only use node types and var_names that appear in this catalog - never
invent a node type or field name that isn't listed:

{catalog}
"""


def _parse_sse_stream(resp: requests.Response) -> str:
    """Accumulates the 'content' deltas from an SSE stream into one string."""
    content = ""
    line_count = 0
    for raw_line in resp.iter_lines():
        line_count += 1
        log.info("SSE raw line #%d: %r", line_count, raw_line)
        if not raw_line:
            continue
        line = raw_line.decode("utf-8")
        if not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices", [])
        if not choices:
            continue
        piece = choices[0].get("delta", {}).get("content")
        if piece:
            content += piece
    return content


def generate_workflow_text(prompt: str, uid: str, token: str, model: str) -> str:
    """
    Calls the LLM with the node catalog + user's request, returns the
    raw text reply (expected to be workflow JSON, not yet parsed/validated
    here - that's the caller's job).
    """
    system_prompt = _build_system_prompt()
    payload = {
        "uid": uid,
        "token": token,
        "category_genvr": "textgen",
        "subcategory_genvr": "completions",
        "stream": True,
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }

    resp = requests.post(
        f"{LLM_BASE}/api/generate-non-polling",
        headers={"Content-Type": "application/json"},
        json=payload,
        stream=True,
    )
    resp.raise_for_status()
    reply_text = _parse_sse_stream(resp)
    log.info("Workflow agent raw LLM reply: %s", reply_text)
    return reply_text