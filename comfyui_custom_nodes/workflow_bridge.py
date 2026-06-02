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

from comfyui_custom_nodes.config import CUSTOM_NODES_DIR, INSTALLED_DIR
from comfyui_custom_nodes.scanner import read_existing_mappings

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


def _class_comfy_meta(class_path: Path, class_name: str) -> dict[str, Any]:
    tree = ast.parse(class_path.read_text(encoding="utf-8"))
    meta: dict[str, Any] = {
        "return_types": [],
        "function": None,
        "category": "comfyui",
        "description": "",
        "input_names": [],
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
                if meta["function"] and item.name == meta["function"]:
                    meta["input_names"] = [
                        arg.arg
                        for arg in item.args.args
                        if arg.arg not in ("self", "cls")
                    ]

    return meta


def _build_ports(class_meta: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    inputs = [
        {
            "var_name": name,
            "display_name": name.replace("_", " ").title(),
            "type": "any",
        }
        for name in class_meta.get("input_names") or ["input"]
    ]

    returns = class_meta.get("return_types") or ["output"]
    outputs = []
    for i, comfy_type in enumerate(returns):
        var = comfy_type.lower() if len(returns) > 1 else "output"
        if len(returns) > 1 and i > 0:
            var = f"{comfy_type.lower()}_{i}"
        outputs.append({
            "var_name": var,
            "display_name": comfy_type,
            "type": COMFY_TYPE_TO_PORT.get(comfy_type, "any"),
        })

    return inputs, outputs


def _make_execute(
    package_dir: Path,
    mapping_key: str,
    input_names: list[str],
    output_keys: list[str],
):
    async def execute(uid: str, token: str, inputs: dict) -> dict:
        init_path = package_dir / "__init__.py"
        if not init_path.is_file():
            raise ValueError(f"package missing __init__.py: {package_dir}")

        module_name = f"_genvr_comfy_{package_dir.name}"
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
                args = [inputs.get(name) for name in input_names]
            else:
                args = [inputs.get("input") or inputs.get("image")]
            args = [a for a in args if a is not None]
            if not args and inputs:
                args = [next(iter(inputs.values()))]

            result = fn(*args)
            if isinstance(result, tuple):
                out: dict[str, Any] = {}
                for i, value in enumerate(result):
                    key = output_keys[i] if i < len(output_keys) else f"output_{i}"
                    out[key] = value
                return out
            key = output_keys[0] if output_keys else "output"
            return {key: result}
        except Exception as exc:
            raise ValueError(
                f"ComfyUI node '{mapping_key}' requires a ComfyUI Python environment "
                f"(folder_paths, torch, etc.). Installed at: {package_dir.resolve()}. "
                f"Use comfyui/custom_nodes in ComfyUI, or install ComfyUI deps in this venv. "
                f"Detail: {exc}"
            ) from exc

    return execute


def _build_module(package_dir: Path, mapping_key: str, class_name: str) -> types.ModuleType:
    class_file = _find_class_file(package_dir, class_name)
    class_meta = _class_comfy_meta(class_file, class_name) if class_file else {}
    inputs, outputs = _build_ports(class_meta)
    input_names = class_meta.get("input_names") or ["image"]
    output_keys = [o["var_name"] for o in outputs]

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
    mod.execute = _make_execute(package_dir, mapping_key, input_names, output_keys)
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
