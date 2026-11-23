"""Locate the pinned upstream checkout and make it importable. No upstream code is copied here."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import PINNED_COMMIT


def use_upstream(path: str | Path, allow_unpinned: bool = False) -> Path:
    """Put the checkout on ``sys.path`` after confirming it is at the pinned commit."""
    root = Path(path).expanduser().resolve()
    if not (root / "demo" / "export" / "stream.py").is_file():
        raise SystemExit(f"Not a Depth Pro 2 checkout: {root}")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if commit != PINNED_COMMIT and not allow_unpinned:
        raise SystemExit(f"Upstream is at {commit[:7]}, expected {PINNED_COMMIT[:7]}.")
    for entry in (str(root), str(root / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return root
