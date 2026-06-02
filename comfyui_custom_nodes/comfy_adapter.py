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

from comfyui_custom_nodes.comfy_debug import comfy_log

_LOG_NODE = "adapter"

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


def _comfy_pillow(fn, arg):
    """ComfyUI-compatible PIL loader (truncated image retry)."""
    from PIL import ImageFile, UnidentifiedImageError

    prev_value = None
    try:
        return fn(arg)
    except (OSError, UnidentifiedImageError, ValueError):
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            return fn(arg)
        finally:
            if prev_value is not None:
                ImageFile.LOAD_TRUNCATED_IMAGES = prev_value


def _install_comfy_shims(input_str: str) -> tuple[Any, Any]:
    """
    Patch folder_paths / node_helpers in sys.modules in-place.

    Comfy node files do ``import node_helpers`` once; replacing sys.modules
    alone leaves stale module references without ``pillow``.
    """
    fp = sys.modules.get("folder_paths")
    if fp is None:
        fp = types.ModuleType("folder_paths")
        sys.modules["folder_paths"] = fp

    nh = sys.modules.get("node_helpers")
    if nh is None:
        nh = types.ModuleType("node_helpers")
        sys.modules["node_helpers"] = nh

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

    def exists_annotated_filepath(filename: str) -> bool:
        try:
            path = get_annotated_filepath(filename)
            return os.path.isfile(path)
        except (OSError, ValueError, FileNotFoundError):
            return False

    def add_model_folder_path(*_args, **_kwargs) -> None:
        pass

    fp.get_input_directory = get_input_directory
    fp.get_annotated_filepath = get_annotated_filepath
    fp.get_output_directory = get_output_directory
    fp.exists_annotated_filepath = exists_annotated_filepath
    fp.add_model_folder_path = add_model_folder_path

    nh.pillow = _comfy_pillow
    if not hasattr(nh, "hasher"):
        nh.hasher = types.SimpleNamespace()

    comfy_log(_LOG_NODE, "shims.installed", {
        "input_dir": input_str,
        "node_helpers_has_pillow": hasattr(nh, "pillow"),
        "folder_paths_module": getattr(fp, "__file__", repr(fp)),
    })
    return fp, nh


def rebind_comfy_shims(package_module_name: str) -> None:
    """Point loaded Comfy submodules at the current shim modules."""
    nh = sys.modules.get("node_helpers")
    fp = sys.modules.get("folder_paths")
    if nh is None or fp is None:
        return

    rebound: list[str] = []
    for key, mod in list(sys.modules.items()):
        if key != package_module_name and not key.startswith(package_module_name + "."):
            continue
        if mod is None:
            continue
        changed = False
        if getattr(mod, "node_helpers", None) is not nh:
            mod.node_helpers = nh
            changed = True
        if getattr(mod, "folder_paths", None) is not fp:
            mod.folder_paths = fp
            changed = True
        if changed:
            rebound.append(key)

    if rebound:
        comfy_log(_LOG_NODE, "shims.rebound", {
            "modules": rebound,
            "node_helpers_has_pillow": hasattr(nh, "pillow"),
        })


def purge_comfy_package_modules(package_module_name: str, package_dir: Path) -> None:
    """Drop cached imports so the next load picks up shim patches."""
    stem = package_dir.name
    removed: list[str] = []
    for key in list(sys.modules):
        if (
            key == package_module_name
            or key.startswith(package_module_name + ".")
            or key == "pose_node"
            or key.endswith(".pose_node")
            or (stem in key and "pose_node" in key)
        ):
            del sys.modules[key]
            removed.append(key)
    if removed:
        comfy_log(_LOG_NODE, "purge.modules", {"removed": removed})


@contextmanager
def comfy_runtime(input_dir: Path):
    """Install minimal ComfyUI stubs and a temp input directory for one node run."""
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    input_str = str(input_dir.resolve())

    _install_comfy_shims(input_str)
    try:
        yield input_dir
    finally:
        pass


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

    comfy_log(_LOG_NODE, "download.start", {"url": url[:200], "ext": ext})
    tmp_path = download_to_tempfile(url, suffix=ext)
    try:
        base = f"genvr_{uuid.uuid4().hex[:12]}{ext}"
        dest = input_dir / base
        shutil.move(tmp_path, dest)
        comfy_log(_LOG_NODE, "download.done", {"filename": base, "bytes": dest.stat().st_size})
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
