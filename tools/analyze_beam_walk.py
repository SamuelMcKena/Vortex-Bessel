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
from collections import defaultdict
from pathlib import Path

import numpy as np

from vbb_study.lab.beamage import read_bmg


def _preprocess(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image, dtype=np.float64)
    # BMG exports may contain signed background-corrected values. Estimate the
    # offset before clipping; never reinterpret negative values as unsigned.
    bg = float(np.percentile(a, 35.0))
    a = a - bg
    a[a < 0.0] = 0.0
    return a


def _gradient_symmetry_center(image: np.ndarray, half_width: int = 70) -> tuple[float, float]:
    """Return (y,x) centre using weighted radial-gradient line intersections."""
    a = _preprocess(image)
    if not np.any(a > 0):
        raise ValueError("No positive signal after background subtraction.")

    # The brightest q=20 principal-ring pixel is only a few pixels from the ring
    # centre, so it is a safe seed for a local crop.  q=0 works as well.
    seed_y, seed_x = np.unravel_index(int(np.argmax(a)), a.shape)
    y0 = max(1, int(seed_y) - half_width)
    y1 = min(a.shape[0] - 1, int(seed_y) + half_width + 1)
    x0 = max(1, int(seed_x) - half_width)
    x1 = min(a.shape[1] - 1, int(seed_x) + half_width + 1)
    crop = a[y0:y1, x0:x1]
    if min(crop.shape) < 25:
        raise ValueError("Signal is too close to the sensor edge for centre estimation.")

    # Mild dependency-free smoothing suppresses isolated pixel noise.
    p = np.pad(crop, 1, mode="edge")
    smooth = (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    ) / 9.0

    gy, gx = np.gradient(smooth)
    mag = np.hypot(gx, gy)
    yy, xx = np.indices(smooth.shape, dtype=np.float64)

    # Use meaningful radial structure only.  The percentile gate adapts between
    # a q=0 central spot and a q=20 annulus without a hard-coded ring radius.
    signal_gate = smooth >= np.percentile(smooth, 65.0)
    grad_gate = mag >= np.percentile(mag, 70.0)
    use = signal_gate & grad_gate
    if np.count_nonzero(use) < 50:
        use = mag >= np.percentile(mag, 60.0)
    if np.count_nonzero(use) < 20:
        raise ValueError("Insufficient structured signal for a stable centre fit.")

    gxv = gx[use]
    gyv = gy[use]
    xv = xx[use]
    yv = yy[use]
    w = np.maximum(mag[use], 1e-12)

    # For a rotationally symmetric pattern the local gradient is radial, so
    # (x-cx, y-cy) is parallel to (gx, gy):
    #       gy*cx - gx*cy = gy*x - gx*y
    A = np.column_stack((gyv, -gxv))
    b = gyv * xv - gxv * yv
    sw = np.sqrt(w / np.max(w))
    sol, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
    cx_local, cy_local = map(float, sol)

    if not (-half_width <= cx_local <= crop.shape[1] + half_width and
            -half_width <= cy_local <= crop.shape[0] + half_width):
        raise ValueError("Centre fit escaped the local signal region.")

    return y0 + cy_local, x0 + cx_local


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

    z = np.asarray([r["z_mm"] for r in means], dtype=float)
    x = np.asarray([r["mean_x_px"] for r in means], dtype=float)
    y = np.asarray([r["mean_y_px"] for r in means], dtype=float)
    fx = np.polyfit(z, x, 1)
    fy = np.polyfit(z, y, 1)

    def r2(v, fit):
        pred = np.polyval(fit, z)
        den = float(np.sum((v - np.mean(v)) ** 2))
        return 1.0 if den == 0 else float(1.0 - np.sum((v - pred) ** 2) / den)

    # px/mm * um/px = um/mm, numerically identical to mrad for small angles.
    sx = float(fx[0] * pixel_um)
    sy = float(fy[0] * pixel_um)
    mag = float(np.hypot(sx, sy))
    result = {
        "folder": str(folder),
        "camera_pixel_um": float(pixel_um),
        "per_frame": rows,
        "per_z_mean": means,
        "fit": {
            "x_px_per_mm": float(fx[0]),
            "y_px_per_mm": float(fy[0]),
            "x_um_per_mm": sx,
            "y_um_per_mm": sy,
            "relative_axis_x_mrad": sx,
            "relative_axis_y_mrad": sy,
            "relative_axis_total_mrad": mag,
            "relative_axis_total_deg_small_angle": float(np.degrees(mag / 1000.0)),
            "r2_x": r2(x, fx),
            "r2_y": r2(y, fy),
        },
        "interpretation": (
            "Relative beam axis versus camera translation direction. Do not call this an absolute optical "
            "beam angle until camera-stage runout has been measured independently."
        ),
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
