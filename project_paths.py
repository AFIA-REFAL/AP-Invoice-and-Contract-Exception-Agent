from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_project_root(start_path: Optional[str | Path] = None) -> Path:
    """Return the repository root based on the location of a file or folder."""
    candidate = Path(start_path).resolve() if start_path is not None else Path(__file__).resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for current in (candidate, *candidate.parents):
        if (
            (current / "data").exists()
            and (current / "agents").exists()
            and (current / "schemas").exists()
            and (current / "tools").exists()
        ):
            return current

    return candidate
