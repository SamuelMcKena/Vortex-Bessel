"""Regenerate the complete live-presentation figure set under one Phase 2J standard.

This is presentation rendering only. The forward-model physics, physical error
injection planes, fixed-laboratory coordinate convention, common nominal
normalisation, tip sampling gate and claim boundaries are preserved.

Version 3 keeps the fixed-laboratory physics but improves the decentre XY
presentation framing: each z=60 mm XY panel is cropped about its measured beam
centroid while retaining true laboratory x/y coordinates.  The x-z maps remain
strictly fixed-laboratory and are not recentered or altered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_phase2i_presentation_figures as core
import build_phase2j_ideal_beam_profile_figure as ideal
import build_presentation_extended_evidence as ext
import build_presentation_extended_evidence_v2 as extv2
import presentation_phase2j_style as style


Z_VALUES_M = np.linspace(5.0e-3, 120.0e-3, 72)
IDEAL_COORD_M = np.linspace(-0.18e-3, 0.18e-3, 481)
DECENTRE_COORD_M = np.linspace(-0.62e-3, 0.62e-3, 621)
TIP_COORD_M = np.linspace(-0.30e-3, 0.30e-3, 601)
ERROR_COORD_M = np.linspace(-0.52e-3, 0.52e-3, 521)

# Presentation-only XY window for the axicon-decentre row.  This is deliberately
# tighter than the old ±0.52 mm window and follows the measured beam centroid.
# The extent remains in laboratory coordinates; the data themselves are never
# shifted, interpolated or numerically recentered.
DECENTRE_XY_HALF_M = 0.24e-3

STALE_GLOBS = (
    "*_inferno*.png",
    "*_inferno*.txt",
)


def _remove_stale_outputs(output_dir: Path) -> None:
    for pattern in STALE_GLOBS:
        for path in output_dir.glob(pattern):
            path.unlink()


def _patch_core_renderer() -> None:
    core.FIG_BG = style.FIG_BG
    core.AX_BG = style.AX_BG
    core.TEXT = style.TEXT
    core.MUTED = style.MUTED
    core.RED = style.RED
    core.GREEN = style.GREEN
    core.BORDER = style.BORDER
    core.CMAP = style.CMAP
    core.Z_VALUES_M = Z_VALUES_M
    core.IDEAL_COORD_M = IDEAL_COORD_M
    core.DECENTRE_COORD_M = DECENTRE_COORD_M
    core.TIP_COORD_M = TIP_COORD_M
    core._normalise = style.normalise
    core._style_figure = style.style_fig
    core._style_axis = style.style_ax
    core._save = style.save
    core._draw_xy = style.draw_xy

    def draw_xz(ax, values, coordinate_m, *, peak=None, show_y=False):
        return style.draw_xz(
            ax,
            values,
            np.asarray(coordinate_m, dtype=float),
            np.asarray(core.Z_VALUES_M, dtype=float),
            peak=peak,
            show_y=show_y,
            z_ref_m=float(core.Z_REF_M),
        )

    def crop(intensity: np.ndarray, grid: Mapping[str, Any], halfwidth_m: float):
        requested = float(halfwidth_m)
        x = np.asarray(grid["x"], dtype=float)

        # build_v1_decentre historically requests a 1 mm fixed-lab crop.  For
        # Phase 2J presentation rendering only, use that request as the signal
        # that this is the decentre XY row and frame each panel around the actual
        # z=60 mm beam centroid.  The axis values remain absolute laboratory
        # coordinates, so the displacement is still visible quantitatively.
        if requested >= 0.8e-3:
            values = np.maximum(np.asarray(intensity, dtype=float), 0.0)
            total = float(np.sum(values))
            X = np.asarray(grid["X"], dtype=float)
            Y = np.asarray(grid["Y"], dtype=float)
            centre_x = float(np.sum(values * X) / max(total, np.finfo(float).tiny))
            centre_y = float(np.sum(values * Y) / max(total, np.finfo(float).tiny))
            halfwidth = float(DECENTRE_XY_HALF_M)
            x_ids = np.flatnonzero(np.abs(x - centre_x) <= halfwidth)
            y_ids = np.flatnonzero(np.abs(x - centre_y) <= halfwidth)
        else:
            halfwidth = style.presentation_crop_halfwidth(requested)
            x_ids = np.flatnonzero(np.abs(x) <= halfwidth)
            y_ids = x_ids

        if x_ids.size < 70 or y_ids.size < 70:
            raise RuntimeError(
                f"Phase 2J crop has only {x_ids.size}x{y_ids.size} native samples; "
                "increase grid_n instead of inventing display resolution."
            )

        crop_values = np.asarray(intensity)[np.ix_(y_ids, x_ids)]
        extent = [
            x[x_ids[0]] * 1e3,
            x[x_ids[-1]] * 1e3,
            x[y_ids[0]] * 1e3,
            x[y_ids[-1]] * 1e3,
        ]
        return crop_values, extent

    core._draw_xz = draw_xz
    core._fixed_lab_crop = crop


def _patch_extended_renderer() -> None:
    ext.FIG_BG = style.FIG_BG
    ext.AX_BG = style.AX_BG
    ext.TEXT = style.TEXT
    ext.MUTED = style.MUTED
    ext.RED = style.RED
    ext.GREEN = style.GREEN
    ext.BORDER = style.BORDER
    ext.CMAP = style.CMAP
    ext.Z_VALUES_M = Z_VALUES_M
    ext.IDEAL_COORD_M = IDEAL_COORD_M
    ext.ERROR_COORD_M = ERROR_COORD_M
    ext._style_fig = style.style_fig
    ext._style_ax = style.style_ax
    ext._save = style.save
    ext._norm = style.normalise
    ext._draw_xy = style.draw_xy
    ext._draw_xz = style.draw_xz

    def crop(values: np.ndarray, grid: Mapping[str, Any], halfwidth_m: float):
        halfwidth = style.presentation_crop_halfwidth(float(halfwidth_m))
        x = np.asarray(grid["x"], dtype=float)
        ids = np.flatnonzero(np.abs(x) <= halfwidth)
        if ids.size < 70:
            raise RuntimeError(
                f"Phase 2J crop has only {ids.size} native samples; "
                "increase grid_n instead of inventing display resolution."
            )
        return np.asarray(values)[np.ix_(ids, ids)], [
            x[ids[0]] * 1e3,
            x[ids[-1]] * 1e3,
            x[ids[0]] * 1e3,
            x[ids[-1]] * 1e3,
        ]

    ext._crop = crop


def _rename(path: Path, new_name: str) -> Path:
    new_path = path.with_name(new_name)
    if new_path.exists():
        new_path.unlink()
    path.replace(new_path)
    return new_path


def _contact_sheet(paths: list[Path], output_path: Path) -> Path:
    """Create one inspectable overview of the exact GitHub-generated outputs."""
    thumbs: list[Image.Image] = []
    labels: list[str] = []
    thumb_w, thumb_h = 900, 520
    for path in paths:
        im = Image.open(path).convert("RGB")
        im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h + 52), (5, 7, 10))
        x = (thumb_w - im.width) // 2
        y = (thumb_h - im.height) // 2
        canvas.paste(im, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.text((16, thumb_h + 15), path.name, fill=(235, 235, 230))
        thumbs.append(canvas)
        labels.append(path.name)

    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 52)), (5, 7, 10))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * thumb_w, (i // cols) * (thumb_h + 52)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)
    return output_path


def build_suite(output_dir: Path, grid_n: int, inverse_grid_n: int) -> dict[str, Any]:
    if grid_n < 2048:
        raise ValueError("Phase 2J full presentation suite requires grid_n >= 2048")
    if inverse_grid_n < 512:
        raise ValueError("Phase 2J inverse proof requires inverse_grid_n >= 512")

    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_outputs(output_dir)
    style.validate_palette_has_no_cool_segment()
    _patch_core_renderer()
    _patch_extended_renderer()

    p1 = core.build_computational_route(output_dir, grid_n)
    p1 = _rename(p1, "01_computational_route_phase2j.png")

    p2 = ideal.build_figure(output_dir, grid_n)

    p4, m4 = core.build_v1_decentre(output_dir, grid_n)
    p4 = _rename(p4, "04_V1_axicon_decentre_fixed_lab_thermal_tight.png")

    p5, m5 = core.build_v1_tip(output_dir, grid_n)
    p5 = _rename(p5, "05_V1_nonideal_tip_fixed_lab_thermal_tight.png")

    p6 = core.build_closure(output_dir)
    p6 = _rename(p6, "06_simulation_experiment_closure_phase2j.png")

    p8, m8 = ext.build_real_error_fingerprints(output_dir, grid_n)
    p8 = _rename(p8, "08_V1_real_error_fingerprints_thermal_tight.png")

    p9, m9 = ext.build_tip_avoidance_proxy(output_dir, grid_n)
    p9 = _rename(p9, "09_tip_avoidance_planning_proxy_thermal.png")

    p10, m10 = extv2.build_synthetic_inverse_recovery(output_dir, inverse_grid_n)
    p10 = _rename(p10, "10_synthetic_zstack_inverse_recovery_thermal.png")

    figures = [p1, p2, p4, p5, p6, p8, p9, p10]
    contact = _contact_sheet(figures, output_dir / "00_phase2j_visual_audit_contact_sheet.jpg")

    manifest: dict[str, Any] = {
        "outcome": "PHASE2J-PRESENTATION-VISUAL-STANDARDISATION-V3",
        "physics_source": "current repository dual-SLM -> explicit 4F -> physical axicon model",
        "grid_n": int(grid_n),
        "inverse_grid_n": int(inverse_grid_n),
        "visual_policy": {
            "intensity_colormap": style.CMAP_NAME,
            "palette": "black -> deep red -> red -> orange -> amber -> yellow; no blue/cyan/green/purple segment",
            "palette_hex": list(style.THERMAL_HEX),
            "save_dpi": style.SAVE_DPI,
            "display_interpolation": style.DISPLAY_INTERPOLATION,
            "display_interpolation_changes_numerical_field": False,
            "z_samples_for_main_longitudinal_maps": int(len(Z_VALUES_M)),
            "fixed_lab_longitudinal": True,
            "per_z_recentering": False,
            "ideal_transverse_halfwidth_m": 0.18e-3,
            "tip_output_transverse_halfwidth_m": 0.24e-3,
            "decentre_xy_halfwidth_m": float(DECENTRE_XY_HALF_M),
            "decentre_xy_display_center": "measured z=60 mm intensity centroid, with absolute laboratory x/y axes preserved",
            "decentre_xy_data_recentering": False,
            "decentre_longitudinal_halfwidth_m": abs(float(DECENTRE_COORD_M[0])),
            "tip_longitudinal_halfwidth_m": abs(float(TIP_COORD_M[0])),
            "error_longitudinal_halfwidth_m": abs(float(ERROR_COORD_M[0])),
        },
        "figures": [str(path) for path in figures],
        "visual_audit_contact_sheet": str(contact),
        "axicon_decentre_metrics": m4,
        "axicon_tip_metrics": m5,
        "real_error_fingerprints": m8,
        "tip_avoidance_planning_proxy": m9,
        "synthetic_inverse_recovery": m10,
        "claim_boundaries": [
            "presentation rendering changes do not alter optical physics",
            "decentre XY windows follow the measured beam centroid for visibility but retain absolute laboratory coordinates",
            "decentre x-z maps remain fixed laboratory coordinates with no recentering",
            "phase panels remain cyclic because phase is not an intensity quantity",
            "signed residual panels remain diverging because residual sign must be visible",
            "axicon perturbation magnitudes remain sensitivity examples unless independently calibrated",
            "tip-avoidance image remains a planning proxy rather than calibrated SLM efficiency evidence",
            "inverse-recovery image remains synthetic model-to-model validation until calibrated camera data are fitted",
        ],
    }
    manifest_path = output_dir / "presentation_phase2j_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-n", type=int, default=2048)
    parser.add_argument("--inverse-grid-n", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/presentation_phase2j"),
    )
    args = parser.parse_args()
    build_suite(args.output_dir, args.grid_n, args.inverse_grid_n)


if __name__ == "__main__":
    main()
