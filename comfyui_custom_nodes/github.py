"""Parse GitHub URLs and download repository archives."""

from __future__ import annotations

import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
import httpx

GITHUB_BLOB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<branch>[^/]+)/(?P<path>.+)$"
)
GITHUB_TREE_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/tree/(?P<branch>[^/]+)/?(?P<path>.*)$"
)
GITHUB_REPO_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/?$"
)
GITHUB_RAW_RE = re.compile(
    r"^https?://raw\.githubusercontent\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/(?P<branch>[^/]+)/(?P<path>.+)$"
)


@dataclass
class GitHubSource:
    owner: str
    repo: str
    branch: str
    subpath: str  # file or directory inside repo, may be empty

    @property
    def slug(self) -> str:
        base = f"{self.owner}_{self.repo}"
        if self.subpath:
            safe = self.subpath.replace("/", "_").replace("\\", "_").removesuffix(".py")
            return f"{base}_{safe}"[:120]
        return base[:120]

    @property
    def zipball_url(self) -> str:
        return f"https://github.com/{self.owner}/{self.repo}/archive/refs/heads/{self.branch}.zip"


def parse_github_url(url: str, *, branch: str | None = None) -> GitHubSource:
    url = url.strip().rstrip("/")
    for pattern in (GITHUB_BLOB_RE, GITHUB_RAW_RE, GITHUB_TREE_RE):
        m = pattern.match(url)
        if m:
            subpath = m.group("path").strip("/")
            return GitHubSource(
                owner=m.group("owner"),
                repo=m.group("repo").removesuffix(".git"),
                branch=branch or m.group("branch"),
                subpath=subpath,
            )

    m = GITHUB_REPO_RE.match(url)
    if m:
        return GitHubSource(
            owner=m.group("owner"),
            repo=m.group("repo").removesuffix(".git"),
            branch=branch or "master",
            subpath="",
        )

    raise ValueError(
        "Unsupported GitHub URL. Use repo root, tree, blob, or raw link "
        "(e.g. .../blob/master/PoseNode/pose_node.py)"
    )


def download_repo_archive(source: GitHubSource, dest_dir: Path) -> Path:
    """Download zipball and return path to extracted repo root."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "repo.zip"

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(follow_redirects=True, timeout=120.0) as client:
                with client.stream("GET", source.zipball_url) as resp:
                    resp.raise_for_status()
                    with zip_path.open("wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=1 << 20):
                            f.write(chunk)
            last_error = None
            break
        except (httpx.HTTPError, OSError) as exc:
            last_error = exc
            zip_path.unlink(missing_ok=True)
    if last_error:
        raise RuntimeError(f"Failed to download {source.zipball_url}: {last_error}") from last_error

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    zip_path.unlink(missing_ok=True)

    roots = [p for p in dest_dir.iterdir() if p.is_dir() and p.name != "__MACOSX"]
    if not roots:
        raise RuntimeError("GitHub archive contained no root directory")
    return roots[0]


def resolve_package_dir(repo_root: Path, subpath: str) -> Path:
    """Pick the ComfyUI package directory from a repo path (file or folder)."""
    if not subpath:
        if (repo_root / "__init__.py").exists():
            return repo_root
        children = [p for p in repo_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        if len(children) == 1 and (children[0] / "__init__.py").exists():
            return children[0]
        return repo_root

    target = repo_root / subpath.replace("/", "\\") if "\\" in subpath else repo_root / subpath
    if target.is_file():
        return target.parent
    if target.is_dir():
        return target
    raise FileNotFoundError(f"Path not found in repository: {subpath}")
