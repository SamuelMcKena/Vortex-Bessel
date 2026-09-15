from __future__ import annotations

import json

import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping

from .config import AppConfig, SlmPhaseConfig
from .phase import PhaseResult, save_phase_png


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def create_cast_folder(output_root: Path, label: str = "cast") -> Path:
    folder = output_root / f"{timestamp()}_{label}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_cast_bundle(folder: Path, app_config: AppConfig, results: Mapping[str, PhaseResult]) -> Dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    metadata = app_config.as_dict()
    metadata["results"] = {}

    for name, result in results.items():
        png_path = folder / f"{name.lower()}_phase_mask.png"
        phase_path = folder / f"{name.lower()}_phase_rad.npy"
        command_path = folder / f"{name.lower()}_phase_wrapped_command_rad.npy"
        save_phase_png(png_path, result.gray_uint8)
        np.save(phase_path, np.asarray(result.phase_rad, dtype=np.float32))
        wrapped = np.ascontiguousarray(np.mod(result.phase_rad, 2.0 * np.pi), dtype=np.float32)
        wrapped[wrapped >= np.float32(2.0 * np.pi)] = 0.0
        np.save(command_path, wrapped)
        written[name] = str(png_path)
        metadata["results"][name] = {
            "png": str(png_path),
            "phase_rad_npy": str(phase_path),
            "phase_wrapped_command_rad_npy": str(command_path),
            "stats": result.stats,
            "warnings": result.warnings,
            "components": list(result.components.keys()),
        }

    meta_path = folder / "cast_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    written["metadata"] = str(meta_path)
    return written
