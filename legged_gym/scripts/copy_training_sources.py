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
"""
from __future__ import annotations

import os
import shutil
import sys
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

    print(f"Done ({len(sources)} files) into {dest_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
