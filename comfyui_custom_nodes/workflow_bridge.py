"""
Expose installed ComfyUI custom nodes in the GenVR workflow engine node list.

ComfyUI nodes use a different runtime (torch, folder_paths, etc.). Listing mirrors
NODE_CLASS_MAPPINGS; execute attempts import or returns a clear ComfyUI setup hint.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import traceback

from comfyui_custom_nodes.comfy_adapter import (
    adapt_comfy_input,
    adapt_comfy_output,
    coerce_media_url,
    comfy_runtime,
    is_comfy_media_input,
    is_comfy_media_output,
    purge_comfy_package_modules,
    rebind_comfy_shims,
)
from comfyui_custom_nodes.comfy_debug import comfy_log
from comfyui_custom_nodes.config import CUSTOM_NODES_DIR, INSTALLED_DIR
from comfyui_custom_nodes.scanner import read_existing_mappings

# GenVR designer: plain URL strings on ports named image / mask.
COMFY_MEDIA_OUTPUT_VAR = {
    "IMAGE": "image",
    "MASK": "mask",
}

COMFY_TYPE_TO_PORT = {
    "IMAGE": "image",
    "MASK": "mask",
    "LATENT": "latent",
    "STRING": "text",
    "INT": "number",
    "FLOAT": "number",
    "BOOLEAN": "boolean",
    "CONDITIONING": "any",
    "CLIP": "any",
    "VAE": "any",
    "MODEL": "any",
}


def is_comfy_node_type(node_type: str) -> bool:
    return node_type.startswith("comfyui.")


def _package_dirs() -> list[Path]:
    dirs: list[Path] = []
    seen: set[Path] = set()
    for root in (INSTALLED_DIR, CUSTOM_NODES_DIR):
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir() or path.name.startswith("."):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            dirs.append(path)
    return dirs


def _parse_mappings(init_path: Path) -> list[str]:
    return read_existing_mappings(init_path.parent)


def _find_class_file(package_dir: Path, class_name: str) -> Path | None:
    for py in package_dir.glob("*.py"):
        if py.name.startswith("_"):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return py
    return None


def _extract_input_type_keys(node: ast.AST) -> dict[str, str]:
    """Best-effort parse INPUT_TYPES return dict keys -> Comfy type name."""
    types: dict[str, str] = {}
    if not isinstance(node, ast.Dict):
        return types

    for key_node, val_node in zip(node.keys, node.values):
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            continue
        section = key_node.value
        if section not in ("required", "optional", "hidden"):
            continue
        if not isinstance(val_node, ast.Dict):
            continue
        for port_key, port_val in zip(val_node.keys, val_node.values):
            if not isinstance(port_key, ast.Constant) or not isinstance(port_key.value, str):
                continue
            port_name = port_key.value
            comfy_type = None
            if isinstance(port_val, ast.Tuple) and port_val.elts:
                first = port_val.elts[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    comfy_type = first.value
            types[port_name] = comfy_type or types.get(port_name, "")
    return types


def _class_comfy_meta(class_path: Path, class_name: str) -> dict[str, Any]:
    tree = ast.parse(class_path.read_text(encoding="utf-8"))
    meta: dict[str, Any] = {
        "return_types": [],
        "function": None,
        "category": "comfyui",
        "description": "",
        "input_names": [],
        "input_types": {},
    }

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue

        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if target.id == "RETURN_TYPES" and isinstance(item.value, ast.Tuple):
                        meta["return_types"] = [
                            elt.value
                            for elt in item.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
                    elif target.id == "FUNCTION" and isinstance(item.value, ast.Constant):
                        meta["function"] = item.value.value
                    elif target.id == "CATEGORY" and isinstance(item.value, ast.Constant):
                        meta["category"] = str(item.value.value).split("/")[0].strip() or "comfyui"
                    elif target.id == "DESCRIPTION" and isinstance(item.value, ast.Constant):
                        meta["description"] = str(item.value.value)

            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == "INPUT_TYPES":
                    for sub in ast.walk(item):
                        if isinstance(sub, ast.Return) and sub.value is not None:
                            meta["input_types"].update(_extract_input_type_keys(sub.value))
                if meta["function"] and item.name == meta["function"]:
                    meta["input_names"] = [
                        arg.arg
                        for arg in item.args.args
                        if arg.arg not in ("self", "cls")
                    ]

    return meta


def _build_ports(class_meta: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    input_types = class_meta.get("input_types") or {}
    input_names = class_meta.get("input_names") or ["input"]

    inputs = []
    for name in input_names:
        comfy_type = input_types.get(name, "")
        port: dict[str, Any] = {
            "var_name": name,
            "display_name": name.replace("_", " ").title(),
            "type": "text",
            "description": "URL, {name, uri, type}, or list for batch",
        }
        if is_comfy_media_input(comfy_type or None, name):
            port["media"] = True
            port["type"] = "text"
        else:
            port["type"] = COMFY_TYPE_TO_PORT.get(comfy_type.upper(), "any") if comfy_type else "any"
        inputs.append(port)

    returns = class_meta.get("return_types") or ["output"]
    outputs = []
    type_counts: dict[str, int] = {}
    for comfy_type in returns:
        key = comfy_type.lower()
        type_counts[key] = type_counts.get(key, 0) + 1

    seen: dict[str, int] = {}
    for i, comfy_type in enumerate(returns):
        base = comfy_type.lower()
        if is_comfy_media_output(comfy_type):
            var = COMFY_MEDIA_OUTPUT_VAR.get(comfy_type.upper(), f"{base}_url")
            outputs.append({
                "var_name": var,
                "display_name": f"{comfy_type} URL",
                "type": "text",
                "media": True,
                "description": "Image URL for GenVR designer",
            })
            continue

        if len(returns) == 1:
            var = "output"
        elif type_counts.get(base, 0) == 1:
            var = base
        else:
            n = seen.get(base, 0)
            seen[base] = n + 1
            var = base if n == 0 else f"{base}_{n}"
        out_type = COMFY_TYPE_TO_PORT.get(comfy_type, "any")
        outputs.append({
            "var_name": var,
            "display_name": comfy_type,
            "type": out_type,
            "description": "",
        })

    return inputs, outputs


def _make_execute(
    package_dir: Path,
    mapping_key: str,
    input_names: list[str],
    output_keys: list[str],
    media_output_keys: frozenset[str],
    input_types: dict[str, str],
    return_types: list[str],
):
    async def execute(uid: str, token: str, inputs: dict) -> dict:
        import tempfile

        init_path = package_dir / "__init__.py"
        if not init_path.is_file():
            raise ValueError(f"package missing __init__.py: {package_dir}")

        comfy_log(mapping_key, "execute.raw_inputs", inputs)

        with tempfile.TemporaryDirectory(prefix="genvr_comfy_in_") as tmp_in:
            input_dir = Path(tmp_in)
            comfy_inputs = dict(inputs)
            for name in input_names:
                if name not in comfy_inputs:
                    continue
                raw = comfy_inputs[name]
                comfy_inputs[name] = adapt_comfy_input(
                    raw,
                    comfy_type=input_types.get(name),
                    var_name=name,
                    input_dir=input_dir,
                )
                comfy_log(mapping_key, f"adapt_input.{name}", {"raw": raw, "adapted": comfy_inputs[name]})
            for name, value in inputs.items():
                if name in comfy_inputs and name in input_names:
                    continue
                comfy_inputs[name] = adapt_comfy_input(
                    value,
                    comfy_type=input_types.get(name),
                    var_name=name,
                    input_dir=input_dir,
                )

            files = sorted(p.name for p in input_dir.iterdir() if p.is_file())
            comfy_log(mapping_key, "input_dir.files", {"dir": str(input_dir), "files": files})

            module_name = f"_genvr_comfy_{package_dir.name}"
            purge_comfy_package_modules(module_name, package_dir)

            with comfy_runtime(input_dir):
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    init_path,
                    submodule_search_locations=[str(package_dir)],
                )
                if not spec or not spec.loader:
                    raise ValueError(f"Could not load ComfyUI package at {package_dir}")

                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                try:
                    spec.loader.exec_module(mod)
                    rebind_comfy_shims(module_name)
                    mappings = getattr(mod, "NODE_CLASS_MAPPINGS", {})
                    node_cls = mappings.get(mapping_key)
                    if node_cls is None:
                        raise ValueError(f"NODE_CLASS_MAPPINGS missing {mapping_key}")

                    instance = node_cls()
                    fn_name = getattr(node_cls, "FUNCTION", None)
                    if not fn_name or not hasattr(instance, fn_name):
                        raise ValueError(f"ComfyUI node {mapping_key} has no callable {fn_name}")

                    fn = getattr(instance, fn_name)
                    if input_names:
                        missing = [n for n in input_names if comfy_inputs.get(n) is None]
                        if missing:
                            raise ValueError(
                                f"missing required input(s) {missing}; "
                                f"received keys {list(inputs.keys())}"
                            )
                        args = [comfy_inputs[name] for name in input_names]
                    else:
                        args = [comfy_inputs.get("input") or comfy_inputs.get("image")]
                        args = [a for a in args if a is not None]
                        if not args and comfy_inputs:
                            args = [next(iter(comfy_inputs.values()))]

                    comfy_log(mapping_key, "execute.call", {"fn": fn_name, "args": args})
                    result = fn(*args)
                    comfy_log(mapping_key, "execute.result", {
                        "type": type(result).__name__,
                        "len": len(result) if isinstance(result, tuple) else None,
                    })
                except Exception as exc:
                    comfy_log(mapping_key, "execute.error", {
                        "error": str(exc),
                        "type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    })
                    raise ValueError(
                        f"ComfyUI node '{mapping_key}' failed "
                        f"(install torch/Pillow/opencv in this venv if missing). "
                        f"Package: {package_dir.resolve()}. Detail: {exc}"
                    ) from exc

            if isinstance(result, tuple):
                ref_image = None
                for i, value in enumerate(result):
                    rtype = (return_types[i] if i < len(return_types) else "") or ""
                    if rtype.upper() == "IMAGE":
                        ref_image = value
                        break

                out: dict[str, Any] = {}
                for i, value in enumerate(result):
                    key = output_keys[i] if i < len(output_keys) else f"output_{i}"
                    rtype = return_types[i] if i < len(return_types) else None
                    raw = adapt_comfy_output(
                        value,
                        comfy_type=rtype,
                        uid=uid,
                        token=token,
                        port_name=key,
                        index=i,
                        reference_image=ref_image,
                    )
                    out[key] = (
                        coerce_media_url(raw, port=key)
                        if key in media_output_keys
                        else raw
                    )
                comfy_log(mapping_key, "execute.outputs", {
                    k: (str(v)[:100] if not isinstance(v, dict) else f"DICT:{list(v.keys())}")
                    for k, v in out.items()
                })
                return out
            key = output_keys[0] if output_keys else "output"
            rtype = return_types[0] if return_types else None
            raw = adapt_comfy_output(
                result,
                comfy_type=rtype,
                uid=uid,
                token=token,
                port_name=key,
            )
            return {
                key: (
                    coerce_media_url(raw, port=key)
                    if key in media_output_keys
                    else raw
                )
            }

    return execute


def _build_module(package_dir: Path, mapping_key: str, class_name: str) -> types.ModuleType:
    class_file = _find_class_file(package_dir, class_name)
    class_meta = _class_comfy_meta(class_file, class_name) if class_file else {}
    inputs, outputs = _build_ports(class_meta)
    input_names = class_meta.get("input_names") or ["image"]
    input_types = class_meta.get("input_types") or {}
    return_types = class_meta.get("return_types") or []
    output_keys = [o["var_name"] for o in outputs]
    media_output_keys = frozenset(
        o["var_name"] for o in outputs if o.get("media") or o["var_name"] in ("image", "mask")
    )

    slug = package_dir.name
    node_type = f"comfyui.{slug}.{mapping_key}"

    mod = types.ModuleType(node_type)
    mod.metadata = {
        "display_name": mapping_key,
        "description": class_meta.get("description")
        or f"ComfyUI custom node from {slug}",
        "category": "comfyui",
        "color": "orange",
        "comfyui": True,
        "package_dir": str(package_dir.resolve()),
        "class_name": class_name,
        "mapping_key": mapping_key,
    }
    mod.inputs = inputs
    mod.outputs = outputs
    mod.execute = _make_execute(
        package_dir,
        mapping_key,
        input_names,
        output_keys,
        media_output_keys,
        input_types,
        return_types,
    )
    mod._node_type = node_type
    return mod


def build_comfy_node_registry() -> dict[str, types.ModuleType]:
    registry: dict[str, types.ModuleType] = {}

    for package_dir in _package_dirs():
        init_path = package_dir / "__init__.py"
        if not init_path.is_file():
            continue

        for mapping_key in _parse_mappings(init_path):
            class_file = None
            class_name = mapping_key
            # mapping key often matches class name
            cf = _find_class_file(package_dir, mapping_key)
            if cf:
                class_file = cf
                class_name = mapping_key
            else:
                # try first class in mappings value - we only have key
                for py in package_dir.glob("*.py"):
                    if py.name.startswith("_"):
                        continue
                    tree = ast.parse(py.read_text(encoding="utf-8"))
                    for node in tree.body:
                        if isinstance(node, ast.ClassDef):
                            class_name = node.name
                            class_file = py
                            break
                    if class_file:
                        break

            if not class_file:
                continue

            mod = _build_module(package_dir, mapping_key, class_name)
            registry[mod._node_type] = mod

    return registry


_comfy_cache: dict[str, types.ModuleType] | None = None


def get_comfy_registry() -> dict[str, types.ModuleType]:
    global _comfy_cache
    _comfy_cache = build_comfy_node_registry()
    return _comfy_cache


def get_comfy_node_module(node_type: str) -> types.ModuleType | None:
    return get_comfy_registry().get(node_type)


def list_comfy_nodes() -> list[dict]:
    result = []
    for node_type, module in get_comfy_registry().items():
        result.append({
            "node_type": node_type,
            "metadata": module.metadata,
            "inputs": module.inputs,
            "outputs": module.outputs,
        })
    return result
