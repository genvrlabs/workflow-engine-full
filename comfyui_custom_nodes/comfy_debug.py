"""Console debug logging for ComfyUI bridge execution."""

from __future__ import annotations

import json
from typing import Any


def comfy_log(node: str, label: str, data: Any) -> None:
    try:
        payload = json.dumps(data, default=str, indent=2)
    except (TypeError, ValueError):
        payload = repr(data)
    print(f"[comfy.{node}] {label}:\n{payload}", flush=True)
