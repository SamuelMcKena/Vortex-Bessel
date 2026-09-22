from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .config import AppConfig

PRESET_SCHEMA_VERSION = 3


def save_preset(path: Path, config: AppConfig) -> None:
    """Save the complete experiment state.

    Locked hardware values are still recorded for reproducibility, but the loader
    re-applies the installed hardware profile so an old preset cannot silently
    redefine a panel's geometry, serial or carrier.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.as_dict()
    payload["preset_schema_version"] = PRESET_SCHEMA_VERSION
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_preset(path: Path) -> AppConfig:
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    cfg = AppConfig.from_dict(data)
    cfg.apply_locked_hardware()
    return cfg
