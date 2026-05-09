#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Copy key training/config sources into the current working directory.

Usage:
  cd /path/to/desired/output/dir
  python /path/to/copy_training_sources.py

The repo root is found by walking up from this script's directory and from cwd
until `legged_gym/envs/base/legged_robot_config.py` exists. Override with env
`LEGGED_GYM_ROOT_DIR` if needed.

Files are always written to the process cwd (where you run the command).

Also writes ``changed_files_vs_github.txt``: repo-relative paths that differ from
the **current branch's upstream** (``git diff @{upstream}``), i.e. the remote
tracking branch for the branch checked out in the repo. Untracked-only files are
not listed. If no upstream exists, tries ``origin/<same-branch-name>``, then
``origin/HEAD`` / ``origin/main`` / ``origin/master``; if none apply, falls back to
``git diff HEAD`` (local commits + working tree vs last commit).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_MARKER = Path("legged_gym") / "envs" / "base" / "legged_robot_config.py"


def find_repo_root() -> Optional[Path]:
    env = os.environ.get("LEGGED_GYM_ROOT_DIR", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if (p / _MARKER).is_file():
            return p
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    for start in (script_dir, cwd):
        for parent in [start, *start.parents]:
            if (parent / _MARKER).is_file():
                return parent
    return None


def _git_verify_ref(repo_root: Path, ref: str) -> bool:
    r = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--verify", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return r.returncode == 0


def _resolve_remote_comparison_ref(repo_root: Path) -> tuple[Optional[str], str]:
    """Ref to diff working tree + index against: prefer current branch's upstream."""
    up = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "@{u}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if up.returncode == 0:
        ref = up.stdout.strip()
        if ref:
            return ref, f"upstream of current branch ({ref})"

    br = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if br.returncode == 0:
        head_name = br.stdout.strip()
        if head_name and head_name != "HEAD":
            origin_same = f"origin/{head_name}"
            if _git_verify_ref(repo_root, origin_same):
                return origin_same, f"no configured @{{u}}; using {origin_same}"

    for ref in ("origin/HEAD", "origin/main", "origin/master"):
        if _git_verify_ref(repo_root, ref):
            return ref, f"no upstream; using default remote ref ({ref})"
    return None, "no usable remote ref"


def git_paths_differing_from_remote(repo_root: Path) -> tuple[list[str], str]:
    """Paths under repo_root that differ from the remote tracking branch (git diff only).

    Uses ``@{upstream}`` when set; see ``_resolve_remote_comparison_ref``. Does not
    list untracked-only paths.

    Returns (sorted_unique_relative_paths, note) where note describes the comparison or an error.
    """
    try:
        inside = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return [], "not a git repository (skipped)"
    except FileNotFoundError:
        return [], "git executable not found (skipped)"

    compare_ref, resolve_note = _resolve_remote_comparison_ref(repo_root)
    paths: set[str] = set()

    if compare_ref:
        diff = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--name-only", compare_ref],
            capture_output=True,
            text=True,
            check=False,
        )
        if diff.returncode != 0:
            return [], f"git diff failed vs {compare_ref}: {diff.stderr.strip() or diff.stdout.strip()}"
        for line in diff.stdout.splitlines():
            line = line.strip()
            if line:
                paths.add(line.replace("\\", "/"))
        note = f"compared working tree + index to remote: {resolve_note}"
    else:
        head_diff = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--name-only", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if head_diff.returncode != 0:
            return [], f"git diff HEAD failed: {head_diff.stderr.strip() or head_diff.stdout.strip()}"
        for line in head_diff.stdout.splitlines():
            line = line.strip()
            if line:
                paths.add(line.replace("\\", "/"))
        note = f"{resolve_note}; compared to local HEAD only"

    return sorted(paths), note


def write_changed_files_report(dest_dir: Path, repo_root: Path, paths: list[str], note: str) -> Path:
    report_path = dest_dir / "changed_files_vs_github.txt"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"# Generated by copy_training_sources.py at {stamp}",
        f"# Repo: {repo_root}",
        f"# {note}",
        f"# Count: {len(paths)}",
        "",
    ]
    lines.extend(paths)
    report_path.write_text("\n".join(lines) + ("\n" if paths else ""), encoding="utf-8")
    return report_path


def main() -> int:
    repo_root = find_repo_root()
    if repo_root is None:
        print(
            "ERROR: could not find legged_gym repo root (file "
            f"{_MARKER.as_posix()}). Set LEGGED_GYM_ROOT_DIR=/home/hanlin/legged_gym",
            file=sys.stderr,
        )
        return 1

    lg = repo_root / "legged_gym"
    sources = [
        lg / "envs" / "base" / "legged_robot_config.py",
        lg / "envs" / "base" / "legged_robot.py",
        lg / "utils" / "terrain.py",
        lg / "utils" / "helpers.py",
        lg / "envs" / "go2" / "go2_config.py",
        repo_root / "rsl_rl" / "rsl_rl" / "runners" / "on_policy_runner.py",
        repo_root / "rsl_rl" / "rsl_rl" / "algorithms" / "ppo.py",
        repo_root / "rsl_rl" / "rsl_rl" / "modules" / "actor_critic.py",
    ]

    dest_dir = Path.cwd()
    missing = []
    for src in sources:
        if not src.is_file():
            missing.append(str(src))
            continue
        dst = dest_dir / src.name
        shutil.copy2(src, dst)
        print(f"Copied {src.name} -> {dst}")

    if missing:
        print("ERROR: missing source(s):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    changed_paths, git_note = git_paths_differing_from_remote(repo_root)
    report = write_changed_files_report(dest_dir, repo_root, changed_paths, git_note)
    print(f"Wrote {report.name} ({len(changed_paths)} path(s), {git_note})")

    print(f"Done ({len(sources)} files) into {dest_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
