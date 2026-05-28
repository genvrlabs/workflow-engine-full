"""
Normalize and batch-expand node inputs.

Supports:
  - Plain URL strings
  - GenVR assets: {name, uri, type}
  - Lists of URLs or assets (batch mode: run node once per item)
  - Nested wrappers: {items: [...]}, single-element list wrappers
"""

from __future__ import annotations

import json
from typing import Any

URI_KEYS = ("uri", "url", "href")
LIST_CONTAINER_KEYS = ("items", "value", "parts", "urls", "data", "assets", "files")

# Inputs that consume a list as one value (not N separate runs).
COLLECTION_INPUTS = frozenset({
    "input_urls",
    "parts",
    "parts_json",
})

MEDIA_PORT_SUFFIXES = ("_url",)
MEDIA_PORT_NAMES = frozenset({
    "input_url",
    "fg_url",
    "bg_url",
    "fg_alpha_url",
    "mask_url",
    "watermark_url",
    "video_url",
    "audio_url",
})


def _port_spec(port: Any) -> dict:
    if isinstance(port, dict):
        return port
    return port.to_dict()


def is_asset(value: Any) -> bool:
    return isinstance(value, dict) and bool(resolve_url(value, required=False))


def resolve_url(value: Any, *, required: bool = True) -> str:
    """Extract a URL from a string or {name, uri, type} asset."""
    if value is None or value == "":
        if required:
            raise ValueError("expected a URL or asset with uri")
        return ""

    value = unwrap_scalar(value)

    if isinstance(value, dict):
        for key in URI_KEYS:
            if key in value and value[key] not in (None, ""):
                return str(value[key]).strip()
        if required:
            raise ValueError("asset object must include uri (or url)")
        return ""

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return resolve_url(value[0], required=required)
        if required:
            raise ValueError("expected a single URL, got a list")
        return ""

    url = str(value).strip()
    if required and not url:
        raise ValueError("expected a non-empty URL")
    return url


def resolve_asset(value: Any, *, default_name: str = "0", default_type: str = "image/png") -> dict:
    """Normalize to {name, uri, type}."""
    if isinstance(value, dict) and any(k in value for k in URI_KEYS):
        uri = resolve_url(value)
        name = str(value.get("name", default_name))
        mime = str(value.get("type", default_type))
        return {"name": name, "uri": uri, "type": mime}

    uri = resolve_url(value)
    return {"name": default_name, "uri": uri, "type": default_type}


def unwrap_scalar(value: Any) -> Any:
    """Unwrap JSON strings and single-item list wrappers."""
    if isinstance(value, str):
        text = value.strip()
        if text and text[0] in "[{":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        return value

    if isinstance(value, (list, tuple)) and len(value) == 1:
        return unwrap_scalar(value[0])

    return value


def unwrap_list(value: Any) -> list[Any]:
    """Return a flat list from arrays or {items: [...]} wrappers."""
    value = unwrap_scalar(value)

    if isinstance(value, dict):
        for key in LIST_CONTAINER_KEYS:
            inner = value.get(key)
            if isinstance(inner, (list, tuple)):
                return list(inner)
        if is_asset(value):
            return [value]
        return [value]

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            inner = unwrap_scalar(value[0])
            if isinstance(inner, (list, tuple)):
                return list(inner)
        return list(value)

    if value is None or value == "":
        return []

    return [value]


def is_batchable_item(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return is_asset(value)
    return False


def is_batchable_list(value: Any) -> bool:
    items = unwrap_list(value)
    if len(items) < 2:
        return False
    return all(is_batchable_item(item) for item in items)


def _port_var_name(port: dict) -> str:
    return port.get("var_name", "")


def is_media_port(port: dict) -> bool:
    if port.get("media"):
        return True
    name = _port_var_name(port)
    if name in MEDIA_PORT_NAMES:
        return True
    return any(name.endswith(suffix) for suffix in MEDIA_PORT_SUFFIXES)


def should_batch_port(port: dict, value: Any) -> bool:
    if port.get("batch") is False:
        return False
    if port.get("batch") is True:
        items = unwrap_list(value)
        return len(items) >= 1 and all(is_batchable_item(item) for item in items)

    name = _port_var_name(port)
    if name in COLLECTION_INPUTS:
        return False
    if port.get("type") == "list" and name not in MEDIA_PORT_NAMES:
        return False

    return is_media_port(port) and is_batchable_list(value)


def detect_batches(
    inputs: dict[str, Any], input_specs: list[Any]
) -> tuple[dict[str, list[Any]], dict[str, Any], int]:
    """
    Split inputs into batched lists and scalars.
    Returns (batched, scalars, batch_count). batch_count is 0 if not batched.
    """
    specs = [_port_spec(p) for p in input_specs]
    spec_by_name = {_port_var_name(s): s for s in specs}

    batched: dict[str, list[Any]] = {}
    scalars: dict[str, Any] = dict(inputs)

    for name, value in inputs.items():
        port = spec_by_name.get(name)
        if port is None or not should_batch_port(port, value):
            continue
        items = unwrap_list(value)
        if items:
            batched[name] = items

    if not batched:
        return {}, scalars, 0

    lengths = {len(v) for v in batched.values()}
    if len(lengths) != 1:
        raise ValueError(
            f"batched inputs must have the same length, got {dict((k, len(v)) for k, v in batched.items())}"
        )

    count = lengths.pop()
    for name in batched:
        scalars.pop(name, None)

    return batched, scalars, count


def normalize_value(value: Any, port: dict) -> Any:
    """Deconstruct one input value for a single node run."""
    if value is None:
        return value

    name = _port_var_name(port)
    ptype = port.get("type", "any")

    if name in COLLECTION_INPUTS:
        return unwrap_list(value)

    if ptype == "list" and port.get("batch") is False:
        items = unwrap_list(value)
        return [resolve_url(x) if is_batchable_item(x) else x for x in items]

    if is_media_port(port):
        return resolve_url(value)

    if ptype == "list":
        return unwrap_list(value)

    return unwrap_scalar(value)


def normalize_inputs(inputs: dict[str, Any], input_specs: list[Any]) -> dict[str, Any]:
    """Deconstruct all inputs for one execution."""
    specs = [_port_spec(p) for p in input_specs]
    spec_by_name = {_port_var_name(s): s for s in specs}
    normalized: dict[str, Any] = {}

    for name, value in inputs.items():
        port = spec_by_name.get(name)
        if port is None:
            normalized[name] = value
        else:
            normalized[name] = normalize_value(value, port)

    return normalized


def aggregate_outputs(results: list[dict[str, Any]], output_specs: list[Any]) -> dict[str, Any]:
    """Merge per-run outputs into lists per output port."""
    if not results:
        return {}

    specs = [_port_spec(p) for p in output_specs]
    keys = [_port_var_name(s) for s in specs] or list(results[0].keys())

    aggregated: dict[str, Any] = {}
    for key in keys:
        values = [r.get(key) for r in results if key in r]
        if len(values) == 1:
            aggregated[key] = values[0]
        else:
            aggregated[key] = values

    if len(results) > 1:
        aggregated["batch_count"] = len(results)
    return aggregated
