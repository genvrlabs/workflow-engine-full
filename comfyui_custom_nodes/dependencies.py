"""Discover and pip-install dependencies for ComfyUI custom node packages."""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

# Top-level imports in node code -> PyPI package names.
IMPORT_TO_PIP: dict[str, str] = {
    "PIL": "Pillow",
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "cv2": "opencv-python",
    "scipy": "scipy",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "transformers": "transformers",
    "diffusers": "diffusers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "einops": "einops",
    "omegaconf": "omegaconf",
    "yaml": "PyYAML",
    "requests": "requests",
    "httpx": "httpx",
    "aiohttp": "aiohttp",
    "websocket": "websocket-client",
    "psutil": "psutil",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "librosa": "librosa",
    "soundfile": "soundfile",
    "av": "av",
    "onnx": "onnxruntime",
    "onnxruntime": "onnxruntime",
    "google": "google-generativeai",
    "openai": "openai",
    "argostranslate": "argostranslate",
    "deep_translator": "deep-translator",
}

# ComfyUI / stdlib — not installable via pip here.
SKIP_IMPORT_ROOTS = frozenset({
    "folder_paths",
    "node_helpers",
    "comfy",
    "nodes",
    "model_management",
    "comfy_extras",
    "latent_preview",
    "server",
    "os",
    "sys",
    "json",
    "re",
    "math",
    "typing",
    "pathlib",
    "hashlib",
    "tempfile",
    "shutil",
    "subprocess",
    "asyncio",
    "functools",
    "itertools",
    "collections",
    "datetime",
    "enum",
    "abc",
    "io",
    "copy",
    "random",
    "time",
    "uuid",
    "base64",
    "struct",
    "logging",
    "traceback",
    "inspect",
    "importlib",
})


def _parse_requirements_lines(text: str) -> list[str]:
    specs: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") and not line.startswith("-e"):
            continue
        specs.append(line.split("#", 1)[0].strip())
    return specs


def _specs_from_requirements_file(path: Path) -> list[str]:
    return _parse_requirements_lines(path.read_text(encoding="utf-8"))


def _specs_from_pyproject(path: Path) -> list[str]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []

    specs: list[str] = []
    project = data.get("project", {})
    deps = project.get("dependencies", [])
    if isinstance(deps, list):
        specs.extend(str(d) for d in deps)

    optional = project.get("optional-dependencies", {})
    if isinstance(optional, dict):
        for group in optional.values():
            if isinstance(group, list):
                specs.extend(str(d) for d in group)

    return specs


def _root_import_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        return node.names[0].name.split(".")[0] if node.names else None
    if isinstance(node, ast.ImportFrom):
        if node.module:
            return node.module.split(".")[0]
    return None


def _specs_from_imports(package_dir: Path) -> list[str]:
    roots: set[str] = set()
    for py_file in package_dir.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            root = _root_import_name(node)
            if root and root not in SKIP_IMPORT_ROOTS:
                roots.add(root)

    specs: list[str] = []
    for root in sorted(roots):
        pip_name = IMPORT_TO_PIP.get(root)
        if pip_name:
            specs.append(pip_name)
    return specs


def collect_dependency_specs(package_dir: Path, repo_root: Path | None = None) -> list[str]:
    """Merge dependency specs from requirements, pyproject, and import scan."""
    seen: set[str] = set()
    ordered: list[str] = []

    def add_specs(specs: list[str], source: str) -> None:
        for spec in specs:
            key = spec.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            ordered.append(spec.strip())

    search_roots = [package_dir]
    if repo_root and repo_root.resolve() != package_dir.resolve():
        search_roots.append(repo_root)

    for root in search_roots:
        req = root / "requirements.txt"
        if req.is_file():
            add_specs(_specs_from_requirements_file(req), "requirements.txt")

        for name in ("requirements-dev.txt", "install-requirements.txt"):
            extra = root / name
            if extra.is_file():
                add_specs(_specs_from_requirements_file(extra), name)

        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            add_specs(_specs_from_pyproject(pyproject), "pyproject.toml")

    add_specs(_specs_from_imports(package_dir), "imports")

    return ordered


def install_dependency_specs(specs: list[str]) -> list[dict]:
    """pip install each dependency spec. Returns per-package results."""
    if not specs:
        return []

    results: list[dict] = []
    for spec in specs:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", spec],
            capture_output=True,
            text=True,
        )
        results.append({
            "package": spec,
            "success": proc.returncode == 0,
            "stdout": proc.stdout[-2000:] if proc.stdout else "",
            "stderr": proc.stderr[-2000:] if proc.stderr else "",
        })
    return results


def install_package_dependencies(
    package_dir: Path,
    repo_root: Path | None = None,
) -> tuple[list[str], list[dict]]:
    """
    Discover and install dependencies.
    Returns (specs installed, pip results).
    """
    specs = collect_dependency_specs(package_dir, repo_root)

    if specs:
        req_snapshot = package_dir / "requirements-genvr.txt"
        req_snapshot.write_text(
            "\n".join(specs) + "\n",
            encoding="utf-8",
        )

    results = install_dependency_specs(specs)
    return specs, results
