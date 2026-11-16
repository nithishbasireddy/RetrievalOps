"""Resolve a Hugging Face model id to a local, pre-downloaded copy when one exists.

Configs always name models by their Hub id (for reproducibility); if
``$RETRIEVALOPS_MODEL_DIR/<org>/<name>`` exists (filled by scripts/fetch_models.py),
that directory is loaded instead of downloading through the Hub cache.
"""

from __future__ import annotations

import getpass
import os
from pathlib import Path


def model_root() -> Path:
    env = os.environ.get("RETRIEVALOPS_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return Path("/scratch") / getpass.getuser() / "retrievalops-models"


def resolve_model_path(model_id: str) -> str:
    local = model_root() / model_id
    if (local / "config.json").is_file():
        return str(local)
    return model_id
