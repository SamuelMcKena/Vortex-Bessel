"""Estimate transverse beam walk versus z from Beamage-4M BMG captures.

This is a diagnostic, not a wavefront retrieval.  It estimates the centre of a
roughly rotationally symmetric spot/ring pattern from image gradients, averages
repeat captures at each z, and fits x(z), y(z).  The fitted slope is the beam
axis *relative to the camera translation direction*.  It does not by itself
separate true optical pointing from camera-stage runout.

Supported filename examples (first number is taken as z in mm unless --z-map is
provided):
    z031_r01.bmg
    z38_r02.bmg
    31_1.bmg

Typical use:
    python tools/analyze_beam_walk.py measurements/q0 --pixel-um 5.5
    python tools/analyze_beam_walk.py measurements/q20 --pixel-um 5.5

Write CSV/JSON outputs next to the captures or choose --output.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab_gui"))

from labcontrol.beam_walk import PlaneObservation, fit_beam_walk
from labcontrol.metrics import gradient_symmetry_center
from vbb_study.lab.beamage import read_bmg


_gradient_symmetry_center = gradient_symmetry_center


def _first_number(name: str) -> float:
    m = re.search(r"(?:^|[^0-9])(?:z)?(\d+(?:\.\d+)?)(?:[^0-9]|$)", name, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot infer z from filename: {name}")
    return float(m.group(1))


def _load_z_map(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in obj.items()}


def analyse(folder: Path, pixel_um: float, z_map: dict[str, float]) -> dict:
    files = sorted(folder.glob("*.bmg"))
    if not files:
        raise ValueError(f"No .bmg files found in {folder}")

    rows = []
    by_z: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for f in files:
        z = z_map.get(f.name, z_map.get(f.stem, None))
        if z is None:
            z = _first_number(f.stem)
        image, _ = read_bmg(f)
        cy, cx = _gradient_symmetry_center(image)
        by_z[float(z)].append((cy, cx))
        rows.append({"file": f.name, "z_mm": float(z), "center_y_px": cy, "center_x_px": cx})

    if len(by_z) < 2:
        raise ValueError("Need at least two distinct z positions to fit beam walk.")

    means = []
    for z in sorted(by_z):
        a = np.asarray(by_z[z], dtype=float)
        means.append({
            "z_mm": z,
            "repeats": int(len(a)),
            "mean_y_px": float(np.mean(a[:, 0])),
            "mean_x_px": float(np.mean(a[:, 1])),
            "std_y_px": float(np.std(a[:, 0], ddof=1)) if len(a) > 1 else 0.0,
            "std_x_px": float(np.std(a[:, 1], ddof=1)) if len(a) > 1 else 0.0,
        })

    fitted = fit_beam_walk(
        (
            PlaneObservation(
                frame_id=row["file"],
                z_mm=row["z_mm"],
                centre_y_px=row["center_y_px"],
                centre_x_px=row["center_x_px"],
            )
            for row in rows
        ),
        pixel_size_um=pixel_um,
        camera_axis_calibrated=False,
    )
    result = {
        "folder": str(folder),
        "camera_pixel_um": float(pixel_um),
        "per_frame": rows,
        "per_z_mean": means,
        "fit": {
            **fitted.fit,
            # Compatibility aliases retained for existing notebooks/reports.
            "x_um_per_mm": fitted.fit["x_mrad"],
            "y_um_per_mm": fitted.fit["y_mrad"],
            "relative_axis_x_mrad": fitted.fit["x_mrad"],
            "relative_axis_y_mrad": fitted.fit["y_mrad"],
            "relative_axis_total_mrad": fitted.fit["total_mrad"],
            "relative_axis_total_deg_small_angle": float(
                np.degrees(float(fitted.fit["total_mrad"]) / 1000.0)
            ),
        },
        "interpretation": fitted.interpretation,
    }
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Estimate q=0/q=20 beam walk versus camera z from BMG files.")
    p.add_argument("folder", type=Path, help="Folder containing BMG captures for one beam state.")
    p.add_argument("--pixel-um", type=float, default=5.5, help="Camera pixel pitch in micrometres (default: 5.5).")
    p.add_argument("--z-map", type=Path, help="Optional JSON mapping filename/stem to physical z in mm.")
    p.add_argument("--output", type=Path, help="Output directory (default: <folder>/beam_walk_analysis).")
    args = p.parse_args(argv)

    if not np.isfinite(args.pixel_um) or args.pixel_um <= 0:
        p.error("--pixel-um must be finite and positive")
    out = args.output or (args.folder / "beam_walk_analysis")
    out.mkdir(parents=True, exist_ok=True)

    result = analyse(args.folder, args.pixel_um, _load_z_map(args.z_map))
    (out / "beam_walk.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with (out / "beam_walk_means.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["z_mm", "repeats", "mean_y_px", "mean_x_px", "std_y_px", "std_x_px"])
        writer.writeheader()
        writer.writerows(result["per_z_mean"])

    fit = result["fit"]
    print(json.dumps(fit, indent=2))
    print(f"Wrote {out / 'beam_walk.json'}")
    print(f"Wrote {out / 'beam_walk_means.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
