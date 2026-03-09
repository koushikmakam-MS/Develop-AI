"""Repo Sync - clones / pulls the Azure DevOps repo and copies
markdown files into docs/agentKT/.

Designed to run daily (via cron, Task Scheduler, or `run_sync.py`).
Uses git commands to detect whether files have changed before copying.
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from brain_ai.config import get_config

log = logging.getLogger(__name__)


def _repo_url_with_pat(repo_url: str, pat: str) -> str:
    """Inject PAT into the HTTPS clone URL for authentication."""
    # https://msazure.visualstudio.com/... -> https://PAT@msazure.visualstudio.com/...
    if "://" in repo_url:
        scheme, rest = repo_url.split("://", 1)
        return f"{scheme}://{pat}@{rest}"
    return repo_url


def _run_git(args: List[str], cwd: Path) -> str:
    """Run a git command and return stdout."""
    import subprocess
    cmd = ["git"] + args
    log.debug("git %s (cwd=%s)", " ".join(args), cwd)
    result = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        log.error("git error: %s", result.stderr.strip())
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def clone_or_pull(cfg: dict | None = None) -> Path:
    """
    Clone the repo if not present, otherwise fetch + pull.
    Returns the path to the local clone directory.
    """
    if cfg is None:
        cfg = get_config()

    ado = cfg["azure_devops"]
    clone_dir = Path(cfg["paths"]["repo_clone_dir"]).resolve()
    branch = ado.get("branch", "master")
    authenticated_url = _repo_url_with_pat(ado["repo_url"], ado["pat"])

    if (clone_dir / ".git").is_dir():
        log.info("Repo already cloned at %s - pulling latest...", clone_dir)
        _run_git(["fetch", "origin"], cwd=clone_dir)
        _run_git(["reset", "--hard", f"origin/{branch}"], cwd=clone_dir)
    else:
        log.info("Cloning repo into %s ...", clone_dir)
        clone_dir.mkdir(parents=True, exist_ok=True)
        _run_git(
            ["clone", "--branch", branch, "--single-branch", "--depth", "1",
             authenticated_url, str(clone_dir)],
            cwd=clone_dir.parent,
        )

    return clone_dir


def get_changed_files(clone_dir: Path, since_hours: int = 25) -> List[str]:
    """
    Return list of files changed in the last `since_hours` hours.
    Uses git log to detect changes.
    """
    since = f"--since={since_hours} hours ago"
    try:
        output = _run_git(
            ["log", since, "--name-only", "--pretty=format:"],
            cwd=clone_dir,
        )
        files = [f for f in output.splitlines() if f.strip()]
        return list(set(files))
    except RuntimeError:
        log.warning("Could not get changed files; treating all as changed.")
        return []


def sync_docs(cfg: dict | None = None, force: bool = False) -> dict:
    """
    Main sync entry point.
    1. Clone / pull the repo.
    2. Detect changed markdown files.
    3. Copy relevant files into docs/agentKT/.

    Returns a summary dict with counts.
    """
    if cfg is None:
        cfg = get_config()

    clone_dir = clone_or_pull(cfg)
    ado = cfg["azure_devops"]
    docs_dir = Path(cfg["paths"]["docs_dir"]).resolve()
    docs_dir.mkdir(parents=True, exist_ok=True)

    sync_paths = ado.get("sync_paths", ["."])
    extensions = set(ado.get("file_extensions", [".md"]))

    # Gather all matching files from the sync_paths
    all_md_files: List[Path] = []
    for sp in sync_paths:
        src = clone_dir / sp
        if not src.exists():
            log.warning("Sync path %s does not exist in repo, skipping.", sp)
            continue
        for ext in extensions:
            all_md_files.extend(src.rglob(f"*{ext}"))

    # If not forcing, check which files changed recently
    changed_set = None
    if not force:
        changed = get_changed_files(clone_dir)
        if changed:
            changed_set = set(changed)
            log.info("%d files changed in last 25 hours.", len(changed_set))

    copied = 0
    skipped = 0
    for md_file in all_md_files:
        rel = md_file.relative_to(clone_dir)
        # If we have a change set, only copy changed files
        if changed_set is not None and str(rel).replace("\\", "/") not in changed_set:
            skipped += 1
            continue

        dest = docs_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_file, dest)
        copied += 1

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_md_files": len(all_md_files),
        "copied": copied,
        "skipped": skipped,
        "force": force,
        "docs_dir": str(docs_dir),
    }
    log.info("Sync complete: %s", summary)
    return summary
