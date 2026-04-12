#!/usr/bin/env python3
"""
Shared project-relative path helpers.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
EXPORTS_ROOT = Path(os.environ.get("PERSONAL_AI_EXPORTS_ROOT", str(PROJECT_ROOT / "exports"))).expanduser()
MODELS_DIR = Path(os.environ.get("PERSONAL_AI_MODELS_DIR", str(PROJECT_ROOT / "models"))).expanduser()
STATE_DB_PATH = DATA_DIR / "state.db"
LIFE_ARCHIVE_DB_PATH = DATA_DIR / "life_archive.db"
PORTFOLIO_PATH = Path(
    os.environ.get("PERSONAL_AI_PORTFOLIO_PATH", str(PROJECT_ROOT.parent / "portfolio"))
).expanduser()


def resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
