"""Install ComfyUI custom node packages from GitHub or local paths."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from comfyui_custom_nodes.config import CUSTOM_NODES_DIR, INSTALLED_DIR
from comfyui_custom_nodes.github import (
    GitHubSource,
    download_repo_archive,
    parse_github_url,
    resolve_package_dir,
)
from comfyui_custom_nodes.dependencies import install_package_dependencies
from comfyui_custom_nodes.scanner import ensure_init_py, read_existing_mappings


@dataclass
class InstallResult:
    id: str
    source: str
    installed_path: str
    custom_nodes_path: str
    package_dir: str
    node_names: list[str]
    dependencies: list[str] = field(default_factory=list)
    requirements: list[dict] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    github: dict | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "installed_path": self.installed_path,
            "custom_nodes_path": self.custom_nodes_path,
            "package_dir": self.package_dir,
            "node_names": self.node_names,
            "dependencies": self.dependencies,
            "requirements": self.requirements,
            "logs": self.logs,
            "github": self.github,
        }


def _is_github_source(source: str) -> bool:
    s = source.strip().lower()
    return "github.com" in s or "raw.githubusercontent.com" in s


def _link_to_custom_nodes(installed: Path, slug: str) -> Path:
    """Symlink installed package into comfyui/custom_nodes (fallback: copy)."""
    CUSTOM_NODES_DIR.mkdir(parents=True, exist_ok=True)
    link = CUSTOM_NODES_DIR / slug

    if link.exists() or link.is_symlink():
        if link.is_symlink():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
        else:
            link.unlink()

    try:
        link.symlink_to(installed, target_is_directory=True)
        return link
    except OSError:
        shutil.copytree(installed, link)
        return link


def _copy_local_package(src: Path) -> Path:
    if src.is_file():
        if src.suffix != ".py":
            raise ValueError("local file must be a .py ComfyUI node module")
        slug = src.stem
        dest = INSTALLED_DIR / slug
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        shutil.copy2(src, dest / src.name)
        return dest

    if not src.is_dir():
        raise FileNotFoundError(f"local path not found: {src}")

    slug = src.name
    dest = INSTALLED_DIR / slug
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return dest


def install_from_source(source: str, *, branch: str | None = None) -> InstallResult:
    INSTALLED_DIR.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []
    github_meta: dict | None = None

    repo_root_for_deps: Path | None = None

    if _is_github_source(source):
        gh = parse_github_url(source, branch=branch)
        github_meta = {
            "owner": gh.owner,
            "repo": gh.repo,
            "branch": gh.branch,
            "subpath": gh.subpath,
            "zipball_url": gh.zipball_url,
        }
        slug = gh.slug

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_root = download_repo_archive(gh, tmp_path)
            repo_root_for_deps = repo_root
            package_src = resolve_package_dir(repo_root, gh.subpath)
            logs.append(f"Resolved package directory: {package_src.relative_to(repo_root)}")

            dest = INSTALLED_DIR / slug
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(package_src, dest)

            for meta_name in ("requirements.txt", "pyproject.toml"):
                meta_src = repo_root / meta_name
                if meta_src.is_file() and not (dest / meta_name).is_file():
                    shutil.copy2(meta_src, dest / meta_name)
                    logs.append(f"Copied repo {meta_name}")
    else:
        local = Path(source).expanduser().resolve()
        dest = _copy_local_package(local)
        slug = dest.name
        logs.append(f"Copied local path: {local}")
        if local.is_dir():
            parent_req = local.parent / "requirements.txt"
            parent_py = local.parent / "pyproject.toml"
            if parent_req.is_file() and not (dest / "requirements.txt").is_file():
                shutil.copy2(parent_req, dest / "requirements.txt")
            if parent_py.is_file() and not (dest / "pyproject.toml").is_file():
                shutil.copy2(parent_py, dest / "pyproject.toml")
            repo_root_for_deps = local.parent if local.parent != local else None

    init_path, init_logs = ensure_init_py(dest)
    logs.extend(init_logs)

    node_names = read_existing_mappings(dest)
    if not node_names:
        from comfyui_custom_nodes.scanner import discover_node_classes

        node_names = list(discover_node_classes(dest).keys())

    dep_specs, req_results = install_package_dependencies(dest, repo_root_for_deps)
    if dep_specs:
        logs.append(f"Installing dependencies: {', '.join(dep_specs)}")
        failed = [r["package"] for r in req_results if not r["success"]]
        if failed:
            logs.append(f"pip install failed for: {', '.join(failed)}")
        else:
            logs.append(f"Installed {len(dep_specs)} package(s) via pip")
    else:
        logs.append("No pip dependencies detected")

    link = _link_to_custom_nodes(dest, slug)

    manifest_path = dest / ".genvr_install.json"
    manifest_path.write_text(
        json.dumps(
            {
                "id": slug,
                "source": source,
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "github": github_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return InstallResult(
        id=slug,
        source=source,
        installed_path=str(dest.resolve()),
        custom_nodes_path=str(link.resolve()),
        package_dir=str(dest.resolve()),
        node_names=node_names,
        dependencies=dep_specs,
        requirements=req_results,
        logs=logs,
        github=github_meta,
    )


def list_installed() -> list[dict]:
    INSTALLED_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for path in sorted(INSTALLED_DIR.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue

        manifest_file = path / ".genvr_install.json"
        manifest = {}
        if manifest_file.exists():
            try:
                manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass

        link = CUSTOM_NODES_DIR / path.name
        results.append({
            "id": path.name,
            "installed_path": str(path.resolve()),
            "custom_nodes_path": str(link.resolve()) if link.exists() else None,
            "linked": link.exists(),
            "has_init": (path / "__init__.py").exists(),
            "node_names": read_existing_mappings(path),
            "manifest": manifest,
        })

    return results


def uninstall(slug: str) -> dict:
    installed = INSTALLED_DIR / slug
    link = CUSTOM_NODES_DIR / slug

    if not installed.exists():
        raise ValueError(f"installation not found: {slug}")

    shutil.rmtree(installed)
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)

    return {"id": slug, "removed": True}
