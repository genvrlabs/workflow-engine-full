"""
GenVR <-> ComfyUI I/O adapter.

Inputs: URL strings, {name, uri, type} assets, or lists of those (batch handled by node_runner).
Outputs: Comfy IMAGE/MASK tensors -> GenVR {name, uri, type} assets.
"""

from __future__ import annotations

import os
import shutil
import sys
import types
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from nodes.ffmpeg._utils import download_to_tempfile, upload_file, url_suffix
from nodes.input_utils import is_asset, resolve_asset, resolve_url, unwrap_list, unwrap_scalar

# ComfyUI type names from INPUT_TYPES / RETURN_TYPES
COMFY_IMAGE_TYPES = frozenset({"IMAGE"})
COMFY_MASK_TYPES = frozenset({"MASK"})
COMFY_MEDIA_TYPES = COMFY_IMAGE_TYPES | COMFY_MASK_TYPES

NAME_HINTS_IMAGE = frozenset({
    "image",
    "images",
    "filename",
    "file",
    "path",
    "photo",
    "picture",
})
NAME_HINTS_MASK = frozenset({"mask", "alpha", "matte"})


def _guess_comfy_input_type(name: str, declared: str | None) -> str | None:
    if declared:
        return declared.upper()
    lower = name.lower()
    if lower in NAME_HINTS_MASK:
        return "MASK"
    if lower in NAME_HINTS_IMAGE:
        return "IMAGE"
    return None


def is_comfy_media_input(comfy_type: str | None, var_name: str) -> bool:
    if comfy_type and comfy_type.upper() in COMFY_MEDIA_TYPES:
        return True
    return _guess_comfy_input_type(var_name, None) in COMFY_MEDIA_TYPES


def is_comfy_media_output(comfy_type: str | None) -> bool:
    return bool(comfy_type and comfy_type.upper() in COMFY_MEDIA_TYPES)


@contextmanager
def comfy_runtime(input_dir: Path):
    """Install minimal ComfyUI stubs and a temp input directory for one node run."""
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    input_str = str(input_dir.resolve())

    folder_paths = types.ModuleType("folder_paths")

    def get_input_directory() -> str:
        return input_str

    def get_annotated_filepath(filename: str, default_dir: str | None = None) -> str:
        if not filename:
            raise ValueError("empty ComfyUI image filename")
        if os.path.isabs(filename) and os.path.isfile(filename):
            return filename
        base = os.path.basename(filename)
        path = os.path.join(input_str, base)
        if os.path.isfile(path):
            return path
        if os.path.isfile(filename):
            return filename
        raise FileNotFoundError(f"ComfyUI input file not found: {filename} (dir={input_str})")

    def get_output_directory() -> str:
        return input_str

    def add_model_folder_path(*_args, **_kwargs) -> None:
        pass

    folder_paths.get_input_directory = get_input_directory
    folder_paths.get_annotated_filepath = get_annotated_filepath
    folder_paths.get_output_directory = get_output_directory
    folder_paths.add_model_folder_path = add_model_folder_path

    node_helpers = types.ModuleType("node_helpers")
    node_helpers.hasher = types.SimpleNamespace()

    prev_fp = sys.modules.get("folder_paths")
    prev_nh = sys.modules.get("node_helpers")
    sys.modules["folder_paths"] = folder_paths
    sys.modules["node_helpers"] = node_helpers
    try:
        yield input_dir
    finally:
        if prev_fp is None:
            sys.modules.pop("folder_paths", None)
        else:
            sys.modules["folder_paths"] = prev_fp
        if prev_nh is None:
            sys.modules.pop("node_helpers", None)
        else:
            sys.modules["node_helpers"] = prev_nh


def _download_to_input_dir(value: Any, input_dir: Path) -> str:
    """Download GenVR media to input_dir; return basename for ComfyUI."""
    value = unwrap_scalar(value)
    items = unwrap_list(value)
    if len(items) > 1:
        raise ValueError(
            "expected a single image URL or asset per run; use a list on the port for batch mode"
        )
    if not items:
        raise ValueError("expected a URL, asset, or non-empty list")
    item = items[0]

    if isinstance(item, str) and os.path.isfile(item):
        dest = input_dir / os.path.basename(item)
        if dest.resolve() != Path(item).resolve():
            shutil.copy2(item, dest)
        return dest.name

    url = resolve_url(item)
    ext = url_suffix(url) or ".png"
    if not ext.startswith("."):
        ext = f".{ext}"

    tmp_path = download_to_tempfile(url, suffix=ext)
    try:
        base = f"genvr_{uuid.uuid4().hex[:12]}{ext}"
        dest = input_dir / base
        shutil.move(tmp_path, dest)
        return base
    except Exception:
        if os.path.isfile(tmp_path):
            os.unlink(tmp_path)
        raise


def adapt_comfy_input(value: Any, *, comfy_type: str | None, var_name: str, input_dir: Path) -> Any:
    """Convert GenVR wire values to what ComfyUI node methods expect."""
    if value is None:
        return None

    media_type = _guess_comfy_input_type(var_name, comfy_type)
    if media_type in COMFY_MEDIA_TYPES:
        return _download_to_input_dir(value, input_dir)

    # Local Comfy filename or filesystem path
    if isinstance(value, str) and not value.strip().startswith(("http://", "https://")):
        base = os.path.basename(value.strip())
        if (input_dir / base).is_file():
            return base
        if os.path.isfile(value):
            dest = input_dir / base
            if Path(value).resolve() != dest.resolve():
                shutil.copy2(value, dest)
            return base
        if base and not value.strip().startswith("{"):
            return base

    if is_asset(value) or (
        isinstance(value, str) and value.strip().startswith(("http://", "https://"))
    ):
        return _download_to_input_dir(value, input_dir)

    return unwrap_scalar(value)


def _tensor_to_numpy(tensor: Any) -> Any:
    import numpy as np

    t = tensor.detach().cpu() if hasattr(tensor, "detach") else tensor
    if hasattr(t, "numpy"):
        arr = t.numpy()
    else:
        arr = np.asarray(t)
    return arr


def _save_tensor_as_png(arr: Any, path: str) -> None:
    import numpy as np
    from PIL import Image

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        scaled = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        if scaled.shape[-1] == 1:
            img = Image.fromarray(scaled[:, :, 0], mode="L")
        elif scaled.shape[-1] == 4:
            img = Image.fromarray(scaled, mode="RGBA")
        else:
            img = Image.fromarray(scaled, mode="RGB")
    else:
        raise ValueError(f"unsupported tensor shape for export: {getattr(arr, 'shape', arr)}")

    img.save(path, format="PNG")


def adapt_comfy_output(
    value: Any,
    *,
    comfy_type: str | None,
    uid: str,
    token: str,
    port_name: str,
    index: int = 0,
) -> Any:
    """Convert ComfyUI return values to GenVR assets or pass-through scalars."""
    if value is None:
        return None

    ctype = (comfy_type or "").upper()
    if ctype not in COMFY_MEDIA_TYPES:
        try:
            import torch

            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                ctype = "MASK" if value.shape[-1] == 1 or value.ndim == 2 else "IMAGE"
        except ImportError:
            pass

    if ctype in COMFY_MEDIA_TYPES or (
        hasattr(value, "detach") and hasattr(value, "shape")
    ):
        arr = _tensor_to_numpy(value)
        tmp = Path(os.environ.get("TEMP", "/tmp")) / f"genvr_comfy_out_{uuid.uuid4().hex}.png"
        try:
            _save_tensor_as_png(arr, str(tmp))
            uri = upload_file(uid, token, str(tmp))
        finally:
            if tmp.is_file():
                tmp.unlink()

        mime = "image/png"
        name = f"{port_name}_{index}" if index else port_name
        asset = resolve_asset({"name": name, "uri": uri, "type": mime}, default_name=name, default_type=mime)
        asset["type"] = "img"
        return asset

    if isinstance(value, (list, tuple)):
        return [
            adapt_comfy_output(
                v,
                comfy_type=comfy_type,
                uid=uid,
                token=token,
                port_name=port_name,
                index=i,
            )
            for i, v in enumerate(value)
        ]

    return value


def adapt_comfy_inputs(
    inputs: dict[str, Any],
    input_names: list[str],
    input_types: dict[str, str],
    input_dir: Path,
) -> dict[str, Any]:
    adapted: dict[str, Any] = {}
    for name in input_names:
        if name not in inputs:
            continue
        adapted[name] = adapt_comfy_input(
            inputs[name],
            comfy_type=input_types.get(name),
            var_name=name,
            input_dir=input_dir,
        )
    for name, value in inputs.items():
        if name not in adapted:
            adapted[name] = adapt_comfy_input(
                value,
                comfy_type=input_types.get(name),
                var_name=name,
                input_dir=input_dir,
            )
    return adapted
