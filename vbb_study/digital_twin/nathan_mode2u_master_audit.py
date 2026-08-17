"""MODE 2U master high-resolution Nathan hexagon audit.

This layer is deliberately a reporting/audit layer over the validated MODE 2P,
2N, 2Q, and 2S machinery.  It regenerates readable figures, energy ledgers,
profile-response plots, a bounded physically interpretable operating-point
sweep, high-N confirmation rows, and a practical source-scale build plan.  It
does not change the solved source-scale physics.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import json
import math

import numpy as np

from vbb_study.digital_twin.nathan_vector_hexagon import (
    EPS,
    MODE2N_DEFAULT_CARRIER_LPMM,
    MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
    MODE2S_PASS_CORRELATION,
    Mode2NRouteResult,
    Mode2SCorrection,
    Mode2SPerturbation,
    NathanSourceParityConfig,
    _json_ready,
    _mode1b_even_axis_crop,
    _mode2n_mm_axis,
    _normalise_image,
    _write_rows,
    alpha_angle_rms_mod_pi,
    angular_profile_on_ring,
    mode1_symmetry_class,
    mode2q_strict_hexagon_gate,
    mode2s_case_row,
    mode2s_combined_cases,
    mode2s_scope_manifest,
    mode2s_slm_aperture_fit_report,
    plot_mode2n_pre_axicon,
    plot_mode2n_xz_comparison,
    plot_mode2p_route_vs_target,
    plot_mode2p_target_alpha_and_sector_map,
    plot_mode2q_backward_vs_raw,
    plot_mode2q_candidate_z60,
    plot_mode2q_masks,
    plot_mode2q_required_hv,
    plot_mode2q_zstack_summary,
    plot_mode2s_case_z60,
    plot_mode2s_slm_fit,
    route_dual_slm_linear_then_qwp_ideal,
    route_patterned_hwp_ideal,
    run_mode2n_dual_slm_4f_route,
    run_mode2n_dual_slm_qwp_route,
    run_mode2n_patterned_hwp_route,
    run_mode2n_v0_reference,
    run_mode2p_jones_synthesis,
    run_mode2q_backward_initialisation,
    run_mode2q_backward_mask_synthesis,
    run_mode2s_degraded_forward,
    run_mode2s_lab_realism,
    run_mode2s_precompensation,
    stokes_from_linear_components,
    mode2n_source_target,
    mode2n_route_metric_row,
)


MODE2U_STAGE = "nathan_mode2u_master_highres_audit"
MODE2U_DEFAULT_OUTPUT_ROOT = Path("outputs/figures/digital_twin/nathan_mode2u_master_highres_audit")
MODE2U_DOC_PATH = Path("docs/68_nathan_master_highres_audit_and_build_plan.md")
MODE2U_RENDER_INTERPOLATION = "lanczos"
MODE2U_FOCUS_CROP_FRACTION = 0.16
MODE2U_PUBLICATION_MIN_FRINGE_SAMPLES = 8.0
MODE2U_PUBLICATION_MIN_RING_DIAMETER_PIXELS = 30.0
MODE2U_MAX_CARRIER_BAND_FRACTION_OF_NYQUIST = 0.50
MODE2U_SUBDIRS = {
    "v0": "00_v0_reference",
    "m2p": "01_m2p_preaxicon",
    "m2n": "02_m2n_source_replication",
    "m2q": "03_m2q_inverse_masks",
    "m2s": "04_m2s_lab_realism",
    "mode1": "05_mode1_microfabrication_contrast",
    "energy": "06_energy_ledgers",
    "profiles": "07_profile_differences",
    "optimal": "08_optimal_hexagon_sweep",
    "build": "09_realistic_build_plan",
    "highn_focus": "10_highN_focus",
}


def mode2u_visual_acceptability(strict_class: str) -> bool:
    """Strict MODE 2U visual acceptance: triangular/six-lobed failures never pass."""

    return str(strict_class) == "visual_hexagonal_field"


def _save_highres(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


def _close(fig: Any) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _plane_metrics(plane: np.ndarray, grid: Mapping[str, Any], *, reference: Any | None = None) -> dict[str, Any]:
    gate = mode2q_strict_hexagon_gate(np.asarray(plane, dtype=float), grid)
    sym = dict(gate["symmetry"])
    row = {
        "strict_class": str(gate["strict_class"]),
        "classifier_label": str(gate["strict_class"]),
        "c60": float(sym.get("rot_corr_60", np.nan)),
        "c120": float(sym.get("rot_corr_120", np.nan)),
        "c120_minus_c60": float(gate["c120_minus_c60"]),
        "dark_core_ratio": float(gate["dark_core_ratio"]),
        "ring_island_count": int(gate["ring_island_count"]),
        "passes": bool(gate["passes_true_hexagon_gate"]),
        "correlation_to_reference": np.nan,
        "angular_profile_correlation_to_reference": np.nan,
        "ring_radius_m": float(gate["ring_radius_m"]),
    }
    if reference is not None:
        ref_plane = np.asarray(reference.reference_plane, dtype=float)
        row["correlation_to_reference"] = float(np.corrcoef(np.asarray(plane, dtype=float).ravel(), ref_plane.ravel())[0, 1])
        _, prof = angular_profile_on_ring(plane, grid, float(row["ring_radius_m"]))
        _, ref_prof = angular_profile_on_ring(ref_plane, grid, float(reference.ring_radius_m))
        row["angular_profile_correlation_to_reference"] = float(np.corrcoef(prof, ref_prof)[0, 1])
    return row


def _result_metrics(result: Mode2NRouteResult) -> dict[str, Any]:
    sym = dict(result.symmetry)
    return {
        "strict_class": str(result.symmetry_class),
        "classifier_label": str(result.symmetry_class),
        "c60": float(sym.get("rot_corr_60", np.nan)),
        "c120": float(sym.get("rot_corr_120", np.nan)),
        "c120_minus_c60": float(sym.get("c120_minus_c60", np.nan)),
        "dark_core_ratio": float(result.dark_core_ratio),
        "ring_island_count": int(sym.get("ring_island_count", -1)),
        "passes": bool(result.passes_v0_match),
        "correlation_to_reference": float(result.v0_comparison.get("z60_full_field_correlation", np.nan)),
        "angular_profile_correlation_to_reference": float(result.v0_comparison.get("angular_profile_correlation_to_v0", np.nan)),
        "ring_radius_m": float(result.ring_radius_m),
    }


def _case_metrics(case: Mapping[str, Any]) -> dict[str, Any]:
    strict = dict(case["strict_gate"])
    sym = dict(strict["symmetry"])
    cmp_ = dict(case["comparison"])
    return {
        "strict_class": str(strict["strict_class"]),
        "classifier_label": str(strict["strict_class"]),
        "c60": float(sym.get("rot_corr_60", np.nan)),
        "c120": float(sym.get("rot_corr_120", np.nan)),
        "c120_minus_c60": float(strict["c120_minus_c60"]),
        "dark_core_ratio": float(strict["dark_core_ratio"]),
        "ring_island_count": int(strict["ring_island_count"]),
        "passes": bool(case["passes"]),
        "correlation_to_reference": float(cmp_.get("z60_full_field_correlation", np.nan)),
        "angular_profile_correlation_to_reference": float(cmp_.get("angular_profile_correlation_to_v0", np.nan)),
        "ring_radius_m": float(strict["ring_radius_m"]),
    }


def _metric_title(stage: str, case_id: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"{stage} / {case_id}\n"
        f"{metrics['strict_class']} | corr {float(metrics['correlation_to_reference']):.4f} | "
        f"c60 {float(metrics['c60']):.3f} c120 {float(metrics['c120']):.3f} "
        f"dC {float(metrics['c120_minus_c60']):.3f} | dark {float(metrics['dark_core_ratio']):.3f} | "
        f"pass={bool(metrics['passes'])}"
    )


def _visual_row(
    *,
    stage: str,
    case_id: str,
    figure: Path,
    root: Path,
    metrics: Mapping[str, Any],
    notes: str = "",
) -> dict[str, Any]:
    strict = str(metrics["strict_class"])
    return {
        "stage": stage,
        "case_id": case_id,
        "figure": _rel(figure, root),
        "correlation_to_reference": float(metrics.get("correlation_to_reference", np.nan)),
        "angular_profile_correlation_to_reference": float(metrics.get("angular_profile_correlation_to_reference", np.nan)),
        "c60": float(metrics.get("c60", np.nan)),
        "c120": float(metrics.get("c120", np.nan)),
        "c120_minus_c60": float(metrics.get("c120_minus_c60", np.nan)),
        "dark_core_ratio": float(metrics.get("dark_core_ratio", np.nan)),
        "ring_island_count": int(metrics.get("ring_island_count", -1)),
        "strict_class": strict,
        "acceptable_hexagon": mode2u_visual_acceptability(strict),
        "pass_fail": "pass" if bool(metrics.get("passes", False)) else "fail",
        "notes": notes,
    }


def _plot_beam_image(
    plane: np.ndarray,
    grid: Mapping[str, Any],
    *,
    title: str,
    path: Path,
    crop_fraction: float = MODE2U_FOCUS_CROP_FRACTION,
    log_version_path: Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    plane = np.asarray(plane, dtype=float)
    x_mm = _mode2n_mm_axis(grid)
    ext = [float(x_mm[0]), float(x_mm[-1]), float(x_mm[0]), float(x_mm[-1])]
    crop, crop_grid = _mode1b_even_axis_crop(plane, grid, float(crop_fraction))
    xc = np.asarray(crop_grid["x"], dtype=float) / 1e-3
    ext_c = [float(xc[0]), float(xc[-1]), float(xc[0]), float(xc[-1])]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    axes[0].imshow(
        _normalise_image(plane, local=True),
        origin="lower", extent=ext, cmap="inferno", vmin=0.0, vmax=1.0,
        interpolation=MODE2U_RENDER_INTERPOLATION, resample=True,
    )
    axes[0].set_title("full field, linear")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[1].imshow(
        _normalise_image(crop, local=True),
        origin="lower", extent=ext_c, cmap="inferno", vmin=0.0, vmax=1.0,
        interpolation=MODE2U_RENDER_INTERPOLATION, resample=True,
    )
    axes[1].set_title("focus crop, linear")
    axes[1].set_xlabel("x (mm)")
    fig.suptitle(title, fontsize=11)
    _save_highres(fig, path)
    _close(fig)
    if log_version_path is not None:
        log_plane = np.log10(np.maximum(plane, 0.0) / max(float(np.max(plane)), EPS) + 1.0e-5)
        log_crop, _ = _mode1b_even_axis_crop(log_plane, grid, float(crop_fraction))
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
        axes[0].imshow(log_plane, origin="lower", extent=ext, cmap="magma", interpolation=MODE2U_RENDER_INTERPOLATION, resample=True)
        axes[0].set_title("full field, log enhanced")
        axes[0].set_xlabel("x (mm)")
        axes[0].set_ylabel("y (mm)")
        axes[1].imshow(log_crop, origin="lower", extent=ext_c, cmap="magma", interpolation=MODE2U_RENDER_INTERPOLATION, resample=True)
        axes[1].set_title("focus crop, log enhanced")
        axes[1].set_xlabel("x (mm)")
        fig.suptitle(title + " (log contrast)", fontsize=11)
        _save_highres(fig, log_version_path)
        _close(fig)
    return path


def _plot_z_lateral_map(
    z_lateral_map: np.ndarray,
    z_values_m: np.ndarray,
    grid: Mapping[str, Any],
    *,
    title: str,
    path: Path,
    lateral_label: str,
    crop_fraction: float = MODE2U_FOCUS_CROP_FRACTION,
) -> Path:
    import matplotlib.pyplot as plt

    x_mm = _mode2n_mm_axis(grid)
    n = len(x_mm)
    half = max(2, int(round(0.5 * float(crop_fraction) * n)))
    mid = n // 2
    lo = max(0, mid - half)
    hi = min(n, mid + half + 1)
    x_crop = x_mm[lo:hi]
    arr = np.asarray(z_lateral_map, dtype=float)[:, lo:hi]
    z_mm = np.asarray(z_values_m, dtype=float) / 1e-3
    ext = [float(x_crop[0]), float(x_crop[-1]), float(z_mm[0]), float(z_mm[-1])]
    fig, ax = plt.subplots(figsize=(11.0, 5.4), constrained_layout=True)
    ax.imshow(
        _normalise_image(arr, local=True),
        origin="lower", aspect="auto", extent=ext, cmap="inferno", vmin=0, vmax=1,
        interpolation=MODE2U_RENDER_INTERPOLATION, resample=True,
    )
    ax.set_xlabel(f"{lateral_label} (mm), focus-cropped")
    ax.set_ylabel("z (mm)")
    ax.set_title(title)
    _save_highres(fig, path)
    _close(fig)
    return path


def _plot_xz_map(xz_map: np.ndarray, z_values_m: np.ndarray, grid: Mapping[str, Any], *, title: str, path: Path) -> Path:
    return _plot_z_lateral_map(xz_map, z_values_m, grid, title=title, path=path, lateral_label="x")


def _plot_yz_map(yz_map: np.ndarray, z_values_m: np.ndarray, grid: Mapping[str, Any], *, title: str, path: Path) -> Path:
    return _plot_z_lateral_map(yz_map, z_values_m, grid, title=title, path=path, lateral_label="y")


def _plot_route_zslice_comparison(
    results: Sequence[Mode2NRouteResult],
    *,
    title: str,
    path: Path,
    orientation: str,
) -> Path:
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(10.8, 2.7 * n), constrained_layout=True, squeeze=False)
    for idx, result in enumerate(results):
        zmap = np.asarray(result.xz_map if orientation == "xz" else result.yz_map, dtype=float)
        grid = result.grid
        x_mm = _mode2n_mm_axis(grid)
        n_axis = len(x_mm)
        half = max(2, int(round(0.5 * MODE2U_FOCUS_CROP_FRACTION * n_axis)))
        mid = n_axis // 2
        lo = max(0, mid - half)
        hi = min(n_axis, mid + half + 1)
        lateral = x_mm[lo:hi]
        z_mm = np.asarray(result.z_values_m, dtype=float) / 1e-3
        ext = [float(lateral[0]), float(lateral[-1]), float(z_mm[0]), float(z_mm[-1])]
        ax = axes[idx, 0]
        ax.imshow(
            _normalise_image(zmap[:, lo:hi], local=True),
            origin="lower", aspect="auto", extent=ext, cmap="inferno", vmin=0, vmax=1,
            interpolation=MODE2U_RENDER_INTERPOLATION, resample=True,
        )
        ax.axhline(result.reference_z_m / 1e-3, color="white", lw=0.8, alpha=0.8)
        corr = float(result.v0_comparison.get("xz_map_correlation_to_v0", np.nan))
        ax.set_title(f"{result.route_id} ({orientation}, focus-cropped, x-z corr {corr:.4f})", fontsize=9)
        ax.set_ylabel("z (mm)")
    axes[-1, 0].set_xlabel(("x" if orientation == "xz" else "y") + " (mm), focus-cropped")
    fig.suptitle(title)
    _save_highres(fig, path)
    _close(fig)
    return path


def _radial_profile(plane: np.ndarray, grid: Mapping[str, Any], bins: int = 220) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(grid["R"], dtype=float).ravel()
    y = np.asarray(plane, dtype=float).ravel()
    edges = np.linspace(0.0, float(np.max(r)), int(bins) + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, int(bins) - 1)
    sums = np.bincount(idx, weights=y, minlength=int(bins))
    counts = np.bincount(idx, minlength=int(bins))
    prof = sums / np.maximum(counts, 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, prof


def _plot_profiles(
    cases: Sequence[Mapping[str, Any]],
    grid: Mapping[str, Any],
    *,
    title: str,
    path: Path,
    reference_ring_radius_m: float | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    x_mm = _mode2n_mm_axis(grid)
    mid = len(x_mm) // 2
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    for case in cases:
        plane = np.asarray(case["plane"], dtype=float)
        label = str(case["label"])
        norm = max(float(np.max(plane)), EPS)
        axes[0, 0].plot(x_mm, plane[mid, :] / norm, label=label)
        axes[0, 1].plot(x_mm, plane[:, mid] / norm, label=label)
        rr, rp = _radial_profile(plane, grid)
        axes[1, 0].plot(rr / 1e-3, rp / max(float(np.max(rp)), EPS), label=label)
        ring = float(reference_ring_radius_m) if reference_ring_radius_m is not None else float(case.get("ring_radius_m", 0.0))
        ang, ap = angular_profile_on_ring(plane, grid, ring)
        axes[1, 1].plot(np.rad2deg(ang), ap / max(float(np.max(ap)), EPS), label=label)
    axes[0, 0].set_title("horizontal line cut")
    axes[0, 1].set_title("vertical line cut")
    axes[1, 0].set_title("radial mean")
    axes[1, 1].set_title("angular profile on main ring")
    axes[0, 0].set_xlabel("x (mm)")
    axes[0, 1].set_xlabel("y (mm)")
    axes[1, 0].set_xlabel("r (mm)")
    axes[1, 1].set_xlabel("angle (deg)")
    axes[0, 0].set_ylabel("normalised intensity")
    axes[0, 1].set_ylabel("normalised intensity")
    axes[1, 0].set_ylabel("normalised intensity")
    axes[1, 1].set_ylabel("normalised intensity")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    _save_highres(fig, path)
    _close(fig)
    return path


def _plot_angular_profiles(
    cases: Sequence[Mapping[str, Any]],
    grid: Mapping[str, Any],
    *,
    title: str,
    path: Path,
    reference_ring_radius_m: float,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    for case in cases:
        plane = np.asarray(case["plane"], dtype=float)
        label = str(case["label"])
        ring = float(case.get("ring_radius_m", reference_ring_radius_m))
        ang, prof = angular_profile_on_ring(plane, grid, ring)
        ax.plot(np.rad2deg(ang), prof / max(float(np.max(prof)), EPS), label=label)
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("normalised intensity on main ring")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _save_highres(fig, path)
    _close(fig)
    return path


def _plot_difference_grid(
    cases: Sequence[Mapping[str, Any]],
    reference_plane: np.ndarray,
    grid: Mapping[str, Any],
    *,
    title: str,
    path: Path,
    max_cols: int = 3,
    crop_fraction: float = MODE2U_FOCUS_CROP_FRACTION,
) -> Path:
    import matplotlib.pyplot as plt

    chosen = list(cases)
    n = len(chosen)
    cols = min(int(max_cols), max(1, n))
    rows = int(math.ceil(n / cols))
    x_mm = _mode2n_mm_axis(grid)
    ref = np.asarray(reference_plane, dtype=float)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.0 * rows), constrained_layout=True, squeeze=False)
    lim = 1.0
    for ax in axes.ravel():
        ax.axis("off")
    for ax, case in zip(axes.ravel(), chosen, strict=False):
        diff = _normalise_image(np.asarray(case["plane"], dtype=float), local=True) - _normalise_image(ref, local=True)
        diff, crop_grid = _mode1b_even_axis_crop(diff, grid, float(crop_fraction))
        xc = np.asarray(crop_grid["x"], dtype=float) / 1e-3
        ext = [float(xc[0]), float(xc[-1]), float(xc[0]), float(xc[-1])]
        lim = max(lim, float(np.max(np.abs(diff))))
        im = ax.imshow(diff, origin="lower", extent=ext, cmap="coolwarm", vmin=-lim, vmax=lim, interpolation=MODE2U_RENDER_INTERPOLATION, resample=True)
        ax.set_title(str(case["label"]), fontsize=8)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.axis("on")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.suptitle(title)
    _save_highres(fig, path)
    _close(fig)
    return path


def _plot_contact_sheet(
    cases: Sequence[Mapping[str, Any]],
    grid: Mapping[str, Any],
    *,
    title: str,
    path: Path,
    crop_fraction: float = MODE2U_FOCUS_CROP_FRACTION,
    max_cols: int = 4,
) -> Path:
    import matplotlib.pyplot as plt

    chosen = list(cases)
    cols = min(int(max_cols), max(1, len(chosen)))
    rows = int(math.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), constrained_layout=True, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, case in zip(axes.ravel(), chosen, strict=False):
        crop, crop_grid = _mode1b_even_axis_crop(np.asarray(case["plane"], dtype=float), grid, float(crop_fraction))
        x = np.asarray(crop_grid["x"], dtype=float) / 1e-3
        ext = [float(x[0]), float(x[-1]), float(x[0]), float(x[-1])]
        ax.imshow(
            _normalise_image(crop, local=True),
            origin="lower", extent=ext, cmap="inferno", vmin=0.0, vmax=1.0,
            interpolation=MODE2U_RENDER_INTERPOLATION, resample=True,
        )
        ax.set_title(str(case["label"]), fontsize=8)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.axis("on")
    fig.suptitle(title)
    _save_highres(fig, path)
    _close(fig)
    return path


def _plot_sweep_metrics(cases: Sequence[Mapping[str, Any]], *, title: str, xlabel: str, path: Path) -> Path:
    import matplotlib.pyplot as plt

    rows = sorted(cases, key=lambda c: float(c["sweep_value"]))
    x = np.asarray([float(c["sweep_value"]) for c in rows], dtype=float)
    corr = np.asarray([float(c["comparison"]["z60_full_field_correlation"]) for c in rows], dtype=float)
    dark = np.asarray([float(c["strict_gate"]["dark_core_ratio"]) for c in rows], dtype=float)
    dc = np.asarray([float(c["strict_gate"]["c120_minus_c60"]) for c in rows], dtype=float)
    passes = np.asarray([bool(c["passes"]) for c in rows], dtype=bool)
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), constrained_layout=True, sharex=True)
    axes[0].plot(x, corr, color="0.35")
    axes[0].scatter(x[passes], corr[passes], color="tab:green", label="strict pass")
    axes[0].scatter(x[~passes], corr[~passes], color="tab:red", label="fail")
    axes[0].axhline(MODE2S_PASS_CORRELATION, color="0.25", ls="--", lw=0.9)
    axes[0].set_ylabel("corr to V0")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, dc, marker="o", color="tab:purple")
    axes[1].set_ylabel("c120 - c60")
    axes[2].plot(x, dark, marker="o", color="tab:blue")
    axes[2].set_ylabel("dark-core ratio")
    axes[2].set_xlabel(xlabel)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    _save_highres(fig, path)
    _close(fig)
    return path


def _as_profile_case(label: str, plane: np.ndarray, ring_radius_m: float) -> dict[str, Any]:
    return {"label": label, "plane": np.asarray(plane, dtype=float), "ring_radius_m": float(ring_radius_m)}


def _normalised_central_power(plane: np.ndarray, grid: Mapping[str, Any], radius_m: float) -> float:
    mask = np.asarray(grid["R"], dtype=float) <= float(radius_m)
    return float(np.sum(np.asarray(plane, dtype=float)[mask]))


def _build_m2p_source_scale(data: Mapping[str, Any]) -> dict[str, Any]:
    report = run_mode2p_jones_synthesis(grid=data["grid"])
    return report


def _sampling_audit_rows(
    config: NathanSourceParityConfig | None,
    *,
    grid_values: Sequence[int],
    carrier_lpmm: float,
    iris_radius_frac: float,
    ring_radius_m: float,
) -> list[dict[str, Any]]:
    cfg = config or NathanSourceParityConfig()
    rows: list[dict[str, Any]] = []
    radial_freq_lpmm = float(cfg.k_r_m_inv / (2.0 * np.pi) / 1.0e3)
    radial_period_m = 1.0 / max(radial_freq_lpmm * 1.0e3, EPS)
    for n in grid_values:
        dx = float(cfg.window_m) / float(n)
        nyquist_lpmm = 0.5 / dx / 1.0e3
        carrier_band_lpmm = float(carrier_lpmm) * (1.0 + float(iris_radius_frac))
        samples_per_fringe = radial_period_m / dx
        ring_diameter_px = 2.0 * float(ring_radius_m) / dx
        spectral_fraction = carrier_band_lpmm / max(nyquist_lpmm, EPS)
        nyquist_pass = bool(samples_per_fringe >= 2.0 and carrier_band_lpmm < nyquist_lpmm)
        publication_pass = bool(
            samples_per_fringe >= MODE2U_PUBLICATION_MIN_FRINGE_SAMPLES
            and ring_diameter_px >= MODE2U_PUBLICATION_MIN_RING_DIAMETER_PIXELS
            and spectral_fraction <= MODE2U_MAX_CARRIER_BAND_FRACTION_OF_NYQUIST
        )
        rows.append({
            "grid_n": int(n),
            "window_mm": float(cfg.window_m / 1e-3),
            "dx_um": float(dx / 1e-6),
            "nyquist_lpmm": nyquist_lpmm,
            "axicon_radial_frequency_lpmm": radial_freq_lpmm,
            "radial_fringe_period_um": float(radial_period_m / 1e-6),
            "samples_per_radial_fringe": float(samples_per_fringe),
            "ring_radius_mm": float(ring_radius_m / 1e-3),
            "ring_diameter_pixels": float(ring_diameter_px),
            "carrier_lpmm": float(carrier_lpmm),
            "iris_radius_frac": float(iris_radius_frac),
            "carrier_plus_iris_band_lpmm": float(carrier_band_lpmm),
            "carrier_band_fraction_of_nyquist": float(spectral_fraction),
            "nyquist_pass": nyquist_pass,
            "publication_recommended": publication_pass,
            "recommendation": (
                "publication_selected"
                if publication_pass else
                "nyquist_ok_visual_only" if nyquist_pass else "undersampled_do_not_use"
            ),
        })
    return rows


def _plot_sampling_audit(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    import matplotlib.pyplot as plt

    x = np.asarray([int(r["grid_n"]) for r in rows], dtype=int)
    fringe = np.asarray([float(r["samples_per_radial_fringe"]) for r in rows], dtype=float)
    ring = np.asarray([float(r["ring_diameter_pixels"]) for r in rows], dtype=float)
    band = np.asarray([float(r["carrier_band_fraction_of_nyquist"]) for r in rows], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 8.2), constrained_layout=True, sharex=True)
    axes[0].plot(x, fringe, marker="o")
    axes[0].axhline(2.0, color="tab:red", ls="--", lw=0.9, label="Nyquist minimum")
    axes[0].axhline(MODE2U_PUBLICATION_MIN_FRINGE_SAMPLES, color="tab:green", ls="--", lw=0.9, label="publication target")
    axes[0].set_ylabel("samples / radial fringe")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, ring, marker="o", color="tab:purple")
    axes[1].axhline(MODE2U_PUBLICATION_MIN_RING_DIAMETER_PIXELS, color="tab:green", ls="--", lw=0.9)
    axes[1].set_ylabel("ring diameter (px)")
    axes[2].plot(x, band, marker="o", color="tab:blue")
    axes[2].axhline(MODE2U_MAX_CARRIER_BAND_FRACTION_OF_NYQUIST, color="tab:green", ls="--", lw=0.9)
    axes[2].set_ylabel("carrier band / Nyquist")
    axes[2].set_xlabel("grid N on 10 mm source window")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("MODE 2U publication sampling and Nyquist audit")
    _save_highres(fig, path)
    _close(fig)
    return path


def _write_v0_reference(out: Path, data: Mapping[str, Any], v0: Mode2NRouteResult, visual_rows: list[dict[str, Any]], root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    metrics = _result_metrics(v0)
    title = _metric_title("V0 source reference", "v0_reference", metrics)
    p = out / "v0_reference_z60_highres.png"
    _plot_beam_image(v0.reference_plane, data["grid"], title=title, path=p, log_version_path=out / "v0_reference_z60_log_highres.png")
    paths["v0_reference_z60"] = p
    visual_rows.append(_visual_row(stage="V0", case_id="v0_reference_z60", figure=p, root=root, metrics=metrics))

    crop_p = out / "v0_reference_crop_highres.png"
    _plot_contact_sheet([_as_profile_case("central crop", v0.reference_plane, v0.ring_radius_m)], data["grid"], title=title, path=crop_p, max_cols=1)
    paths["v0_reference_crop"] = crop_p
    xz_p = out / "v0_reference_xz_highres.png"
    _plot_xz_map(v0.xz_map, v0.z_values_m, data["grid"], title="V0 source reference x-z map", path=xz_p)
    paths["v0_reference_xz"] = xz_p
    yz_p = out / "v0_reference_yz_highres.png"
    _plot_yz_map(v0.yz_map, v0.z_values_m, data["grid"], title="V0 source reference y-z map", path=yz_p)
    paths["v0_reference_yz"] = yz_p
    prof_p = out / "v0_reference_profiles_highres.png"
    _plot_profiles([_as_profile_case("V0", v0.reference_plane, v0.ring_radius_m)], data["grid"], title="V0 source reference line/radial/angular profiles", path=prof_p, reference_ring_radius_m=v0.ring_radius_m)
    paths["v0_reference_profiles"] = prof_p
    return paths


def _write_m2p(out: Path, report: Mapping[str, Any], visual_rows: list[dict[str, Any]], root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    import matplotlib.pyplot as plt

    fig, _ = plot_mode2p_target_alpha_and_sector_map(report["target_data"])
    p = out / "m2p_target_alpha_sector_highres.png"
    _save_highres(fig, p)
    _close(fig)
    paths["m2p_target_alpha_sector"] = p
    fig, _ = plot_mode2p_route_vs_target(report["patterned_hwp"], report["target_data"])
    p = out / "m2p_patterned_hwp_vs_target_highres.png"
    _save_highres(fig, p)
    _close(fig)
    paths["m2p_patterned_hwp_vs_target"] = p
    fig, _ = plot_mode2p_route_vs_target(report["dual_slm_qwp"], report["target_data"])
    p = out / "m2p_dual_slm_qwp_vs_target_highres.png"
    _save_highres(fig, p)
    _close(fig)
    paths["m2p_dual_slm_qwp_vs_target"] = p

    target = report["target_data"]["target"]
    routes = [("target", target[0], target[1]), ("patterned HWP", report["patterned_hwp"].Ex, report["patterned_hwp"].Ey), ("dual SLM + QWP", report["dual_slm_qwp"].Ex, report["dual_slm_qwp"].Ey)]
    grid = report["target_data"]["grid"]
    x_mm = _mode2n_mm_axis(grid)
    ext = [float(x_mm[0]), float(x_mm[-1]), float(x_mm[0]), float(x_mm[-1])]
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 11.0), constrained_layout=True)
    for row, (label, ex, ey) in enumerate(routes):
        st = stokes_from_linear_components(ex, ey)
        panels = [(st["S1"], "S1"), (st["S2"], "S2"), (st["S3"], "S3")]
        for ax, (arr, name) in zip(axes[row], panels, strict=True):
            lim = max(float(np.max(np.abs(arr))), EPS)
            im = ax.imshow(arr, origin="lower", extent=ext, cmap="coolwarm", vmin=-lim, vmax=lim)
            ax.set_title(f"{label}: {name}", fontsize=9)
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            fig.colorbar(im, ax=ax, shrink=0.72)
    fig.suptitle("M2P source-scale pre-axicon Stokes comparison (Gaussian intensity plane, not propagated hexagon)")
    p = out / "m2p_stokes_comparison_highres.png"
    _save_highres(fig, p)
    _close(fig)
    paths["m2p_stokes_comparison"] = p

    visual_rows.append({
        "stage": "M2P",
        "case_id": "preaxicon_jones_synthesis",
        "figure": _rel(paths["m2p_dual_slm_qwp_vs_target"], root),
        "correlation_to_reference": float(report["dual_slm_qwp"].overlap_to_target),
        "angular_profile_correlation_to_reference": np.nan,
        "c60": np.nan,
        "c120": np.nan,
        "c120_minus_c60": np.nan,
        "dark_core_ratio": np.nan,
        "ring_island_count": -1,
        "strict_class": "pre_axicon_gaussian_not_propagated",
        "acceptable_hexagon": False,
        "pass_fail": "pass",
        "notes": "M2P verifies pre-axicon Jones synthesis only; intensity is Gaussian before the axicon.",
    })
    return paths


def _write_m2n(
    out: Path,
    data: Mapping[str, Any],
    routes: Sequence[Mode2NRouteResult],
    v0: Mode2NRouteResult,
    visual_rows: list[dict[str, Any]],
    root: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    name_map = {
        "route_2na_patterned_hwp": "m2n_route_patterned_hwp_z60_highres.png",
        "route_2nb_dual_slm_qwp": "m2n_route_dual_slm_qwp_z60_highres.png",
        "route_2nc_dual_slm_4f": "m2n_route_dual_slm_4f_z60_highres.png",
    }
    for result in routes:
        if result.route_id == "v0_reference":
            continue
        metrics = _result_metrics(result)
        p = out / name_map.get(result.route_id, f"m2n_{result.route_id}_z60_highres.png")
        _plot_beam_image(result.reference_plane, result.grid, title=_metric_title("M2N source replication", result.route_id, metrics), path=p)
        paths[result.route_id] = p
        visual_rows.append(_visual_row(stage="M2N", case_id=result.route_id, figure=p, root=root, metrics=metrics))

    p = out / "m2n_route_xz_comparison_highres.png"
    _plot_route_zslice_comparison(routes, title="M2N x-z propagation comparison (focus-cropped)", path=p, orientation="xz")
    paths["m2n_route_xz_comparison"] = p
    p = out / "m2n_route_yz_comparison_highres.png"
    _plot_route_zslice_comparison(routes, title="M2N y-z propagation comparison (focus-cropped)", path=p, orientation="yz")
    paths["m2n_route_yz_comparison"] = p

    profile_cases = [_as_profile_case(r.route_id, r.reference_plane, r.ring_radius_m) for r in routes]
    p = out / "m2n_route_profile_comparison_highres.png"
    _plot_profiles(profile_cases, data["grid"], title="M2N route profiles versus V0", path=p, reference_ring_radius_m=v0.ring_radius_m)
    paths["m2n_route_profile_comparison"] = p
    p = out / "m2n_route_difference_to_v0_highres.png"
    _plot_difference_grid(profile_cases[1:], v0.reference_plane, data["grid"], title="M2N route difference to V0", path=p)
    paths["m2n_route_difference_to_v0"] = p
    return paths


def _write_m2q(out: Path, report: Mapping[str, Any], visual_rows: list[dict[str, Any]], root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    data = report["data"]
    v0 = report["v0"]
    backward = report["backward"]
    candidates = list(report["candidates"])
    by_id = {str(c["candidate_id"]): c for c in candidates}
    candidate = by_id.get("phase_only_4f", candidates[-1])

    fig, _ = plot_mode2n_pre_axicon(backward.Ex_required_pre_axicon, backward.Ey_required_pre_axicon, data, title="M2Q backward recovered pre-axicon field")
    p = out / "m2q_backward_recovered_preaxicon_highres.png"
    _save_highres(fig, p)
    _close(fig)
    paths["m2q_backward_recovered_preaxicon"] = p
    fig, _ = plot_mode2q_backward_vs_raw(backward, data)
    p = out / "m2q_backward_vs_raw_nathan_highres.png"
    _save_highres(fig, p)
    _close(fig)
    paths["m2q_backward_vs_raw_nathan"] = p
    fig, _ = plot_mode2q_required_hv(backward, data)
    p = out / "m2q_required_hv_amp_phase_highres.png"
    _save_highres(fig, p)
    _close(fig)
    paths["m2q_required_hv_amp_phase"] = p
    fig, _ = plot_mode2q_masks(backward.phi_H_initial, backward.phi_V_initial, data)
    p = out / "m2q_phase_only_masks_highres.png"
    _save_highres(fig, p)
    _close(fig)
    paths["m2q_phase_only_masks"] = p
    fig, _ = plot_mode2q_candidate_z60(candidate, data)
    p = out / "m2q_forward_verification_z60_highres.png"
    _save_highres(fig, p)
    _close(fig)
    paths["m2q_forward_verification_z60"] = p
    p = out / "m2q_forward_verification_xz_highres.png"
    _plot_xz_map(candidate["xz_map"], candidate["z_values_m"], data["grid"], title="M2Q forward verification x-z map (focus-cropped)", path=p)
    paths["m2q_forward_verification_xz"] = p
    p = out / "m2q_forward_verification_yz_highres.png"
    _plot_yz_map(candidate["yz_map"], candidate["z_values_m"], data["grid"], title="M2Q forward verification y-z map (focus-cropped)", path=p)
    paths["m2q_forward_verification_yz"] = p

    metrics = _case_metrics({
        "strict_gate": candidate["strict_gate"],
        "comparison": candidate["comparison"],
        "passes": candidate["passes"],
    })
    visual_rows.append(_visual_row(stage="M2Q", case_id=str(candidate["candidate_id"]), figure=paths["m2q_forward_verification_z60"], root=root, metrics=metrics))
    return paths


def _write_m2s(out: Path, profile_out: Path, report: Mapping[str, Any], visual_rows: list[dict[str, Any]], root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    data = report["data"]
    fig, _ = plot_mode2s_case_z60(report["clean_case"], data)
    p = out / "m2s_clean_baseline_highres.png"
    _save_highres(fig, p)
    _close(fig)
    paths["m2s_clean_baseline"] = p
    clean_xy_path = p
    p = out / "m2s_clean_baseline_xz_highres.png"
    _plot_xz_map(report["clean_case"]["xz_map"], report["clean_case"]["z_values_m"], data["grid"], title="M2S clean baseline x-z map (focus-cropped)", path=p)
    paths["m2s_clean_baseline_xz"] = p
    p = out / "m2s_clean_baseline_yz_highres.png"
    _plot_yz_map(report["clean_case"]["yz_map"], report["clean_case"]["z_values_m"], data["grid"], title="M2S clean baseline y-z map (focus-cropped)", path=p)
    paths["m2s_clean_baseline_yz"] = p
    visual_rows.append(_visual_row(stage="M2S", case_id="clean_baseline", figure=clean_xy_path, root=root, metrics=_case_metrics(report["clean_case"])))

    fig, _ = plot_mode2s_slm_fit(report["fit_report"], data)
    p = out / "m2s_slm_fit_aperture_highres.png"
    _save_highres(fig, p)
    _close(fig)
    paths["m2s_slm_fit_aperture"] = p

    sweep_names = {
        "phase_quantisation": "phase quantisation",
        "hv_piston": "H/V piston",
        "hv_amplitude_ratio": "H/V power imbalance",
        "qwp_angle": "QWP angle",
        "qwp_retardance": "QWP retardance",
        "iris_radius": "iris radius",
        "iris_decentre": "iris decentre",
        "hv_shift": "H/V registration",
        "axicon_decentre": "axicon decentre",
        "z_offset": "z offset",
    }
    for key, label in sweep_names.items():
        cases = list(report["sweep_cases"].get(key, ()))
        if not cases:
            continue
        metric_path = out / f"m2s_{key}_metrics_highres.png"
        _plot_sweep_metrics(cases, title=f"M2S sweep: {label}", xlabel=str(cases[0]["sweep_parameter"]), path=metric_path)
        paths[f"m2s_{key}_metrics"] = metric_path
        sheet_cases = [
            _as_profile_case(f"{c['sweep_value']:.4g}: {c['failure_mode']}", c["reference_plane"], c["strict_gate"]["ring_radius_m"])
            for c in cases
        ]
        contact_path = out / f"m2s_{key}_contact_sheet_highres.png"
        _plot_contact_sheet(sheet_cases, data["grid"], title=f"M2S {label}: fields at z reference", path=contact_path)
        paths[f"m2s_{key}_contact"] = contact_path
        boundary = [cases[0], cases[len(cases) // 2], cases[-1]]
        boundary_cases = [_as_profile_case(f"{c['sweep_value']:.4g}: {c['failure_mode']}", c["reference_plane"], c["strict_gate"]["ring_radius_m"]) for c in boundary]
        boundary_path = out / f"m2s_{key}_boundary_cases_highres.png"
        _plot_contact_sheet(boundary_cases, data["grid"], title=f"M2S {label}: boundary cases", path=boundary_path, max_cols=3)
        paths[f"m2s_{key}_boundary"] = boundary_path
        for c in cases:
            visual_rows.append(_visual_row(stage="M2S", case_id=str(c["label"]), figure=contact_path, root=root, metrics=_case_metrics(c)))

        profile_subset = [cases[0], cases[len(cases) // 2], cases[-1]]
        profile_cases = [_as_profile_case(str(c["label"]), c["reference_plane"], c["strict_gate"]["ring_radius_m"]) for c in profile_subset]
        _plot_profiles(profile_cases, data["grid"], title=f"Profiles: {label}", path=profile_out / f"profiles_{key}_lines.png", reference_ring_radius_m=report["v0"].ring_radius_m)
        _plot_angular_profiles(profile_cases, data["grid"], title=f"Angular profiles: {label}", path=profile_out / f"profiles_{key}_angular.png", reference_ring_radius_m=report["v0"].ring_radius_m)
        _plot_difference_grid(profile_cases, report["clean_case"]["reference_plane"], data["grid"], title=f"Difference to clean: {label}", path=profile_out / f"profiles_{key}_difference_heatmaps.png")

    combined = list(report["combined_cases"])
    comp = list(report["compensated_cases"])
    combined_cases = [_as_profile_case(str(c["label"]), c["reference_plane"], c["strict_gate"]["ring_radius_m"]) for c in combined + comp]
    if combined_cases:
        p = out / "m2s_combined_and_compensated_contact_sheet_highres.png"
        _plot_contact_sheet(combined_cases, data["grid"], title="M2S combined and compensated cases", path=p)
        paths["m2s_combined_and_compensated"] = p
        for c in combined + comp:
            safe = str(c["label"]).replace("/", "_").replace("\\", "_")
            p_xz = out / f"m2s_{safe}_xz_highres.png"
            _plot_xz_map(c["xz_map"], c["z_values_m"], data["grid"], title=f"M2S {c['label']} x-z map (focus-cropped)", path=p_xz)
            paths[f"m2s_{safe}_xz"] = p_xz
            p_yz = out / f"m2s_{safe}_yz_highres.png"
            _plot_yz_map(c["yz_map"], c["z_values_m"], data["grid"], title=f"M2S {c['label']} y-z map (focus-cropped)", path=p_yz)
            paths[f"m2s_{safe}_yz"] = p_yz
        p2 = profile_out / "profiles_combined_mild_moderate_bad_lines.png"
        _plot_profiles(combined_cases, data["grid"], title="M2S combined and compensated profiles", path=p2, reference_ring_radius_m=report["v0"].ring_radius_m)
        _plot_angular_profiles(combined_cases, data["grid"], title="M2S combined and compensated angular profiles", path=profile_out / "profiles_combined_mild_moderate_bad_angular.png", reference_ring_radius_m=report["v0"].ring_radius_m)
        p3 = profile_out / "profiles_combined_mild_moderate_bad_difference_heatmaps.png"
        _plot_difference_grid(combined_cases, report["clean_case"]["reference_plane"], data["grid"], title="Combined cases difference to clean", path=p3)
        for c in combined + comp:
            visual_rows.append(_visual_row(stage="M2S", case_id=str(c["label"]), figure=p, root=root, metrics=_case_metrics(c)))

    return paths


def _write_mode1_contrast(out: Path, source_result: Mode2NRouteResult, visual_rows: list[dict[str, Any]], root: Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {}
    notes = [
        ("MODE 1 inherited downstream", "Failed as triangular/lobed under inherited objective/sample geometry.", "triangular_dark_core"),
        ("MODE 1E redesigned downstream", "Redesign search did not pass the V0-template hexagon gate.", "six_lobed_structured"),
        ("Source-scale MODE 2N/2S", "Passes as visual_hexagonal_field in source-scale simulation.", source_result.symmetry_class),
    ]
    for filename, title, detail, cls in [
        ("mode1_inherited_failure_highres.png", notes[0][0], notes[0][1], notes[0][2]),
        ("mode1e_redesigned_failure_highres.png", notes[1][0], notes[1][1], notes[1][2]),
        ("source_vs_micro_branch_contrast_highres.png", "Source-scale success versus microfabrication contrast", "MODE 2 source-scale success does not unblock the separate microfabrication branch.", notes[2][2]),
    ]:
        fig, ax = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
        ax.axis("off")
        ax.text(0.05, 0.72, title, fontsize=18, weight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.52, detail, fontsize=12, transform=ax.transAxes, wrap=True)
        ax.text(0.05, 0.34, f"classifier / audit label: {cls}", fontsize=12, transform=ax.transAxes)
        ax.text(0.05, 0.18, "Claim boundary: historical microfabrication/sample-plane branch remains separate and blocked.", fontsize=11, transform=ax.transAxes)
        p = out / filename
        _save_highres(fig, p)
        _close(fig)
        paths[filename] = p
        metrics = {
            "strict_class": cls,
            "correlation_to_reference": np.nan,
            "angular_profile_correlation_to_reference": np.nan,
            "c60": np.nan,
            "c120": np.nan,
            "c120_minus_c60": np.nan,
            "dark_core_ratio": np.nan,
            "ring_island_count": -1,
            "passes": mode2u_visual_acceptability(cls),
        }
        visual_rows.append(_visual_row(stage="MODE1_CONTRAST", case_id=filename.removesuffix(".png"), figure=p, root=root, metrics=metrics))
    return paths


def _energy_route_rows(
    *,
    route_id: str,
    branch: str,
    first_order_efficiency: float,
    zero_order_fraction: float,
    rejected_fraction: float,
    pre_power_ratio: float,
    z60_integrated_ratio: float,
    notes: str,
) -> list[dict[str, Any]]:
    stages = [
        ("input_gaussian", 1.0, 1.0, "input reference power"),
        ("after_hv_split", 1.0, 1.0, "lossless ideal H/V split or equivalent"),
        ("after_slm_phase_application", 1.0, 1.0, "phase-only modulation"),
        ("after_carrier_application", 1.0, 1.0, "carrier phase is lossless"),
        ("fourier_plane_total_power", 1.0, 1.0, "Parseval-normalised spectral power"),
        ("selected_first_order_power", 1.0, first_order_efficiency, "4F first-order selection"),
        ("zero_order_power_diagnostic", 1.0, zero_order_fraction, "diagnostic leakage/content, not a forward stage"),
        ("rejected_or_clipped_power", 1.0, rejected_fraction, "4F/aperture rejected power"),
        ("after_inverse_4f_reconstruction", first_order_efficiency, first_order_efficiency, "selected order reconstructed"),
        ("after_hv_recombination", first_order_efficiency, first_order_efficiency, "ideal H/V recombination"),
        ("after_qwp", first_order_efficiency, first_order_efficiency, "uniform QWP is lossless"),
        ("before_axicon", first_order_efficiency, first_order_efficiency * pre_power_ratio, "pre-axicon usable power ratio"),
        ("after_axicon", first_order_efficiency * pre_power_ratio, first_order_efficiency * pre_power_ratio, "thin axicon model, no measured coating loss"),
        ("z60_integrated_plane", first_order_efficiency * pre_power_ratio, z60_integrated_ratio, "numerical z=60 integrated-intensity ratio"),
        ("zstack_power_consistency", z60_integrated_ratio, z60_integrated_ratio, "diagnostic stack consistency placeholder"),
    ]
    rows = []
    for idx, (stage, in_p, out_p, reason) in enumerate(stages, start=1):
        rows.append({
            "route_id": route_id,
            "branch": branch,
            "stage_index": idx,
            "stage": stage,
            "input_power_norm": float(in_p),
            "output_power_norm": float(out_p),
            "transmission_fraction": float(out_p / max(in_p, EPS)),
            "loss_fraction": float(max(0.0, 1.0 - out_p / max(in_p, EPS))),
            "loss_reason": reason,
            "notes": notes,
        })
    return rows


def _build_energy_ledgers(m2n_routes: Sequence[Mode2NRouteResult], m2q: Mapping[str, Any], m2s: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v0_power = max(float(np.sum(np.asarray(m2n_routes[0].reference_plane, dtype=float))), EPS)
    for route in m2n_routes:
        if route.route_id == "v0_reference":
            continue
        f4 = dict(route.slm_4f_report or {})
        eff = float(f4.get("first_order_efficiency", 1.0))
        zero = float(f4.get("zero_order_leakage_after_iris", 0.0))
        rejected = float(f4.get("blocked_power_fraction", 0.0))
        pre = float(route.pre_axicon_metrics.get("power_ratio", 1.0))
        z60 = float(np.sum(np.asarray(route.reference_plane, dtype=float)) / v0_power)
        rows.extend(_energy_route_rows(
            route_id=route.route_id,
            branch="M2N",
            first_order_efficiency=eff,
            zero_order_fraction=zero,
            rejected_fraction=rejected,
            pre_power_ratio=pre,
            z60_integrated_ratio=z60,
            notes="M2N source-scale route",
        ))
    candidate = next((c for c in m2q["candidates"] if str(c["candidate_id"]) == "phase_only_4f"), m2q["candidates"][-1])
    f4 = dict(candidate.get("slm_4f_report") or {})
    rows.extend(_energy_route_rows(
        route_id="m2q_phase_only_4f_recovered_mask",
        branch="M2Q",
        first_order_efficiency=float(f4.get("first_order_efficiency", 1.0)),
        zero_order_fraction=0.0,
        rejected_fraction=float(1.0 - f4.get("first_order_efficiency", 1.0)),
        pre_power_ratio=float(candidate["pre_axicon_vs_required"]["power_ratio"]),
        z60_integrated_ratio=float(np.sum(np.asarray(candidate["reference_plane"], dtype=float)) / v0_power),
        notes="M2Q backward recovered phase-only mask route",
    ))
    selected_cases = [m2s["clean_case"]]
    selected_cases += [c for c in m2s["combined_cases"] if any(k in str(c["label"]) for k in ("moderate", "bad"))]
    selected_cases += list(m2s["compensated_cases"])
    for case in selected_cases:
        iris = dict(case["iris"])
        rows.extend(_energy_route_rows(
            route_id=str(case["label"]),
            branch="M2S",
            first_order_efficiency=float(iris["first_order_efficiency"]),
            zero_order_fraction=float(iris["zero_order_leakage_after_iris"]),
            rejected_fraction=float(iris["rejected_power_fraction"]),
            pre_power_ratio=float(case["pre_axicon"]["power_ratio"]),
            z60_integrated_ratio=float(np.sum(np.asarray(case["reference_plane"], dtype=float)) / v0_power),
            notes=f"M2S degraded route; compensated={bool(case['compensated'])}; failure={case['failure_mode']}",
        ))
    return rows


def _plot_energy_ledgers(rows: Sequence[Mapping[str, Any]], out: Path) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    paths: dict[str, Path] = {}
    route_ids = list(dict.fromkeys(str(r["route_id"]) for r in rows))
    final = []
    first = []
    for rid in route_ids:
        route = [r for r in rows if str(r["route_id"]) == rid]
        first.append(next((float(r["output_power_norm"]) for r in route if r["stage"] == "selected_first_order_power"), np.nan))
        final.append(next((float(r["output_power_norm"]) for r in route if r["stage"] == "z60_integrated_plane"), np.nan))
        fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
        ax.bar([str(r["stage"]) for r in route], [float(r["output_power_norm"]) for r in route], color="tab:blue")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.set_ylabel("normalised power / diagnostic ratio")
        ax.set_title(f"Energy ledger: {rid}")
        p = out / f"energy_ledger_route_{rid}.png"
        _save_highres(fig, p)
        _close(fig)
        paths[f"route_{rid}"] = p
    fig, ax = plt.subplots(figsize=(11.0, 5.0), constrained_layout=True)
    x = np.arange(len(route_ids))
    ax.bar(x - 0.18, first, width=0.36, label="selected first order")
    ax.bar(x + 0.18, final, width=0.36, label="z60 integrated ratio")
    ax.set_xticks(x)
    ax.set_xticklabels(route_ids, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("normalised ratio")
    ax.set_title("MODE 2U source-scale energy ledger summary")
    ax.legend()
    p = out / "energy_ledger_summary_plot.png"
    _save_highres(fig, p)
    _close(fig)
    paths["summary"] = p
    return paths


def _shape_power_scores(case: Mapping[str, Any], v0_peak: float, v0_central: float, grid: Mapping[str, Any], v0_ring_m: float) -> dict[str, float]:
    cmp_ = dict(case["comparison"])
    strict = dict(case["strict_gate"])
    iris = dict(case["iris"])
    corr = float(cmp_["z60_full_field_correlation"])
    ang = float(cmp_["angular_profile_correlation_to_v0"])
    c_pen = min(1.0, abs(float(strict["c120_minus_c60"])) / 0.20)
    dark = float(strict["dark_core_ratio"])
    dark_score = max(0.0, 1.0 - min(1.0, dark / 0.25))
    gate_bonus = 1.0 if bool(strict["passes_true_hexagon_gate"]) else 0.0
    shape = float(np.clip(0.50 * corr + 0.20 * ang + 0.15 * dark_score + 0.15 * gate_bonus - 0.10 * c_pen, 0.0, 1.0))
    plane = np.asarray(case["reference_plane"], dtype=float)
    peak = float(np.max(plane) / max(v0_peak, EPS))
    central = float(_normalised_central_power(plane, grid, 2.0 * v0_ring_m) / max(v0_central, EPS))
    throughput = float(iris["first_order_efficiency"])
    power = float(np.clip((throughput * peak * central) ** (1.0 / 3.0), 0.0, 2.0))
    combined = float(0.60 * shape + 0.40 * min(power, 1.0))
    return {
        "shape_score": shape,
        "power_score": power,
        "combined_score": combined,
        "throughput": throughput,
        "peak_ratio_to_v0": peak,
        "central_power_ratio_to_v0": central,
    }


def _run_optimal_sweep(data: Mapping[str, Any], v0: Mode2NRouteResult, backward: Any, *, max_cases: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    carrier_values = [5.75, 6.25, 6.75]
    iris_values = [0.32, 0.40, 0.52]
    qwp_values_deg = [-0.25, 0.0, 0.25]
    rotation_values_deg = [-1.0, 0.0, 1.0]
    piston_values = [0.0, 0.10]
    combos = list(product(carrier_values, iris_values, qwp_values_deg, rotation_values_deg, piston_values))
    if max_cases is not None:
        combos = combos[: int(max_cases)]
    v0_peak = float(np.max(np.asarray(v0.reference_plane, dtype=float)))
    v0_central = _normalised_central_power(v0.reference_plane, data["grid"], 2.0 * float(v0.ring_radius_m))
    rows: list[dict[str, Any]] = []
    cases: dict[str, Mapping[str, Any]] = {}
    for idx, (carrier, iris, qwp_deg, rot_deg, piston) in enumerate(combos):
        case_id = f"opt_{idx:03d}_c{carrier:.2f}_i{iris:.2f}_q{qwp_deg:+.2f}_r{rot_deg:+.1f}_p{piston:.2f}"
        pert = Mode2SPerturbation(
            label=case_id,
            slm_aperture_clip=True,
            phase_levels=256,
            fill_factor=0.95,
            carrier_lpmm=float(carrier),
            iris_radius_frac=float(iris),
        )
        corr = Mode2SCorrection(
            qwp_angle_correction_rad=float(np.deg2rad(qwp_deg)),
            sector_rotation_rad=float(np.deg2rad(rot_deg)),
            global_v_piston_rad=float(piston),
        )
        case = run_mode2s_degraded_forward(data, v0, backward, pert, correction=corr, fast_single_plane=True)
        scores = _shape_power_scores(case, v0_peak, v0_central, data["grid"], float(v0.ring_radius_m))
        strict = dict(case["strict_gate"])
        row = {
            "case_id": case_id,
            "carrier_lpmm": float(carrier),
            "iris_radius_frac": float(iris),
            "qwp_angle_correction_deg": float(qwp_deg),
            "sector_rotation_deg": float(rot_deg),
            "global_v_piston_rad": float(piston),
            "z60_full_field_correlation": float(case["comparison"]["z60_full_field_correlation"]),
            "angular_profile_correlation_to_v0": float(case["comparison"]["angular_profile_correlation_to_v0"]),
            "first_order_efficiency": float(case["iris"]["first_order_efficiency"]),
            "dark_core_ratio": float(strict["dark_core_ratio"]),
            "c120_minus_c60": float(strict["c120_minus_c60"]),
            "strict_class": str(strict["strict_class"]),
            "passes": bool(case["passes"]),
            **scores,
        }
        rows.append(row)
        cases[case_id] = case
    return rows, cases


def _pareto_front(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    front: set[str] = set()
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                float(other["shape_score"]) >= float(row["shape_score"])
                and float(other["power_score"]) >= float(row["power_score"])
                and (
                    float(other["shape_score"]) > float(row["shape_score"])
                    or float(other["power_score"]) > float(row["power_score"])
                )
            ):
                dominated = True
                break
        if not dominated:
            front.add(str(row["case_id"]))
    return front


def _write_optimal_outputs(out: Path, data: Mapping[str, Any], v0: Mode2NRouteResult, rows: list[dict[str, Any]], cases: Mapping[str, Mapping[str, Any]], visual_rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    front = _pareto_front(rows)
    for row in rows:
        row["pareto_front"] = str(row["case_id"]) in front
    best_shape = max(rows, key=lambda r: float(r["shape_score"]))
    best_power = max(rows, key=lambda r: float(r["power_score"]))
    best_compromise = max(rows, key=lambda r: float(r["combined_score"]))
    _write_rows(out / "optimal_hexagon_pareto.csv", rows)
    (out / "optimal_hexagon_pareto.json").write_text(json.dumps(_json_ready(rows), indent=2), encoding="utf-8")
    _write_rows(out / "optimal_hexagon_parameter_table.csv", [best_shape, best_power, best_compromise])

    fig, ax = plt.subplots(figsize=(7.5, 5.8), constrained_layout=True)
    colors = ["tab:green" if r["pareto_front"] else "0.55" for r in rows]
    ax.scatter([r["shape_score"] for r in rows], [r["power_score"] for r in rows], c=colors, s=34, alpha=0.9)
    for label, row in [("shape", best_shape), ("power", best_power), ("compromise", best_compromise)]:
        ax.scatter([row["shape_score"]], [row["power_score"]], s=110, label=f"best {label}")
    ax.set_xlabel("shape score")
    ax.set_ylabel("power score")
    ax.set_title("MODE 2U physically interpretable Pareto sweep")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    p = out / "optimal_hexagon_pareto_plot.png"
    _save_highres(fig, p)
    _close(fig)

    bests = {
        "best_shape": best_shape,
        "best_power": best_power,
        "best_compromise": best_compromise,
    }
    image_names = {
        "best_shape": "optimal_hexagon_best_shape.png",
        "best_power": "optimal_hexagon_best_power.png",
        "best_compromise": "optimal_hexagon_best_compromise.png",
    }
    profile_cases = []
    for key, row in bests.items():
        case = cases[str(row["case_id"])]
        metrics = _case_metrics(case)
        p = out / image_names[key]
        _plot_beam_image(case["reference_plane"], data["grid"], title=_metric_title("MODE 2U optimal sweep", str(row["case_id"]), metrics), path=p)
        visual_rows.append(_visual_row(stage="MODE2U_OPTIMAL", case_id=str(row["case_id"]), figure=p, root=root, metrics=metrics))
        profile_cases.append(_as_profile_case(key, case["reference_plane"], case["strict_gate"]["ring_radius_m"]))
    _plot_profiles(profile_cases, data["grid"], title="Optimal hexagon profile comparison", path=out / "optimal_hexagon_profile_comparison.png", reference_ring_radius_m=v0.ring_radius_m)
    return {"rows": rows, "cases": cases, "best_shape": best_shape, "best_power": best_power, "best_compromise": best_compromise}


def _write_high_n_confirmation(
    root: Path,
    *,
    config: NathanSourceParityConfig | None,
    base_rows: Mapping[str, Mapping[str, Any]],
    run_high_n: bool,
    high_n_values: Sequence[int],
    z_planes: int,
    focus_output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if focus_output_dir is not None:
        focus_output_dir.mkdir(parents=True, exist_ok=True)

    def _write_focus_triplet(case_id: str, plane: np.ndarray, xz_map: np.ndarray, yz_map: np.ndarray, z_values_m: np.ndarray, grid: Mapping[str, Any], metrics: Mapping[str, Any] | None = None) -> None:
        if focus_output_dir is None:
            return
        safe = str(case_id).replace("/", "_").replace("\\", "_")
        met = dict(metrics or _plane_metrics(plane, grid))
        _plot_beam_image(
            plane,
            grid,
            title=_metric_title("MODE 2U high-N focus", safe, met),
            path=focus_output_dir / f"highN_{safe}_xy_focus.png",
            log_version_path=focus_output_dir / f"highN_{safe}_xy_focus_log.png",
        )
        _plot_xz_map(xz_map, z_values_m, grid, title=f"High-N {safe} x-z focus crop", path=focus_output_dir / f"highN_{safe}_xz_focus.png")
        _plot_yz_map(yz_map, z_values_m, grid, title=f"High-N {safe} y-z focus crop", path=focus_output_dir / f"highN_{safe}_yz_focus.png")

    if not run_high_n:
        rows.append({
            "case_id": "highN_not_run",
            "grid_n": 0,
            "z_planes": 0,
            "z60_full_field_correlation": np.nan,
            "strict_class": "not_run",
            "notes": "High-N confirmation disabled for this run.",
        })
    else:
        for n in high_n_values:
            cfg = replace(config or NathanSourceParityConfig(), grid_n=int(n), z_planes=int(z_planes))
            data = mode2n_source_target(cfg, grid_n=int(n), z_planes=int(z_planes))
            v0 = run_mode2n_v0_reference(data)
            backward = run_mode2q_backward_initialisation(data)
            _write_focus_triplet("clean_source_v0_reference", v0.reference_plane, v0.xz_map, v0.yz_map, v0.z_values_m, data["grid"], _result_metrics(v0))
            rows.append({
                "case_id": "clean_source_v0_reference",
                "grid_n": int(n),
                "z_planes": int(z_planes),
                "z60_full_field_correlation": 1.0,
                "strict_class": str(v0.symmetry_class),
                "passes": bool(v0.passes_v0_match),
                "notes": "selected high-N clean V0/source-scale reference",
            })
            route = run_mode2n_dual_slm_4f_route(data, v0)
            _write_focus_triplet("realistic_dual_slm_4f_baseline", route.reference_plane, route.xz_map, route.yz_map, route.z_values_m, data["grid"], _result_metrics(route))
            rows.append({
                "case_id": "realistic_dual_slm_4f_baseline",
                "grid_n": int(n),
                "z_planes": int(z_planes),
                "z60_full_field_correlation": float(route.v0_comparison["z60_full_field_correlation"]),
                "strict_class": str(route.symmetry_class),
                "passes": bool(route.passes_v0_match),
                "notes": "N=1024 source-scale 4F baseline rerun" if int(n) == 1024 else "higher-N source-scale 4F baseline rerun",
            })
            moderate = Mode2SPerturbation(
                label="highN_moderate_combined",
                slm_aperture_clip=True, phase_levels=256, fill_factor=0.90,
                hv_shift_x_m=24.0e-6, hv_rotation_rad=float(np.deg2rad(0.2)), hv_piston_rad=0.3, hv_amplitude_ratio=1.1,
                qwp_angle_error_rad=float(np.deg2rad(1.0)), qwp_retardance_error_rad=float(np.deg2rad(2.0)),
                iris_radius_frac=0.36, iris_decentre_fx_lpmm=0.5, axicon_decentre_x_m=0.2e-3, z_offset_m=5.0e-3,
                zernike_common={"defocus": 0.3, "astig0": 0.2},
            )
            mcase = run_mode2s_degraded_forward(data, v0, backward, moderate)
            _write_focus_triplet("moderate_combined", mcase["reference_plane"], mcase["xz_map"], mcase["yz_map"], mcase["z_values_m"], data["grid"], _case_metrics(mcase))
            rows.append({
                "case_id": "moderate_combined",
                "grid_n": int(n),
                "z_planes": int(z_planes),
                "z60_full_field_correlation": float(mcase["comparison"]["z60_full_field_correlation"]),
                "strict_class": str(mcase["strict_gate"]["strict_class"]),
                "passes": bool(mcase["passes"]),
                "notes": "selected high-N M2S moderate combined case",
            })
            comp_pert = Mode2SPerturbation(label="highN_compensated_axicon_0p5", slm_aperture_clip=True, axicon_decentre_x_m=0.5e-3)
            comp = Mode2SCorrection(mask_recentre_x_m=0.5e-3)
            ccase = run_mode2s_degraded_forward(data, v0, backward, comp_pert, correction=comp)
            _write_focus_triplet("compensated_0p5mm_axicon_decentre_seeded_recentre", ccase["reference_plane"], ccase["xz_map"], ccase["yz_map"], ccase["z_values_m"], data["grid"], _case_metrics(ccase))
            rows.append({
                "case_id": "compensated_0p5mm_axicon_decentre_seeded_recentre",
                "grid_n": int(n),
                "z_planes": int(z_planes),
                "z60_full_field_correlation": float(ccase["comparison"]["z60_full_field_correlation"]),
                "strict_class": str(ccase["strict_gate"]["strict_class"]),
                "passes": bool(ccase["passes"]),
                "notes": "seeded physical mask recentre at high N; no full optimiser rerun",
            })
            for label, opt in base_rows.items():
                pert = Mode2SPerturbation(
                    label=f"highN_{label}_{opt['case_id']}",
                    slm_aperture_clip=True,
                    phase_levels=256,
                    fill_factor=0.95,
                    carrier_lpmm=float(opt["carrier_lpmm"]),
                    iris_radius_frac=float(opt["iris_radius_frac"]),
                )
                corr = Mode2SCorrection(
                    qwp_angle_correction_rad=float(np.deg2rad(float(opt["qwp_angle_correction_deg"]))),
                    sector_rotation_rad=float(np.deg2rad(float(opt["sector_rotation_deg"]))),
                    global_v_piston_rad=float(opt["global_v_piston_rad"]),
                )
                ocase = run_mode2s_degraded_forward(data, v0, backward, pert, correction=corr)
                _write_focus_triplet(f"{label}_{opt['case_id']}", ocase["reference_plane"], ocase["xz_map"], ocase["yz_map"], ocase["z_values_m"], data["grid"], _case_metrics(ocase))
                rows.append({
                    "case_id": f"{label}_{opt['case_id']}",
                    "grid_n": int(n),
                    "z_planes": int(z_planes),
                    "z60_full_field_correlation": float(ocase["comparison"]["z60_full_field_correlation"]),
                    "strict_class": str(ocase["strict_gate"]["strict_class"]),
                    "passes": bool(ocase["passes"]),
                    "notes": "selected high-N optimal operating-point confirmation",
                })
        if run_high_n and not any(int(v) >= 1536 for v in high_n_values):
            rows.append({
                "case_id": "n1536_or_n2048_not_run",
                "grid_n": 1536,
                "z_planes": int(z_planes),
                "z60_full_field_correlation": np.nan,
                "strict_class": "not_run",
                "passes": False,
                "notes": "N=1536/2048 was not run in this pass; N=1024 was the selected high-N confirmation because it already resolves the source-scale branch and keeps runtime bounded.",
            })
    (root / "highN_confirmation.csv").parent.mkdir(parents=True, exist_ok=True)
    _write_rows(root / "highN_confirmation.csv", rows)
    (root / "highN_confirmation.json").write_text(json.dumps(_json_ready(rows), indent=2), encoding="utf-8")
    _plot_high_n_summary(rows, root / "highN_confirmation_summary.png")
    return rows


def _plot_high_n_summary(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    import matplotlib.pyplot as plt

    plot_rows = [r for r in rows if np.isfinite(float(r.get("z60_full_field_correlation", np.nan)))]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    if plot_rows:
        labels = [f"{r['case_id']}\nN={r['grid_n']}" for r in plot_rows]
        vals = [float(r["z60_full_field_correlation"]) for r in plot_rows]
        ax.bar(labels, vals, color=["tab:green" if bool(r.get("passes", False)) else "tab:orange" for r in plot_rows])
        ax.axhline(MODE2S_PASS_CORRELATION, color="0.3", ls="--", lw=0.9)
        ax.tick_params(axis="x", rotation=25, labelsize=7)
        ax.set_ylim(0.0, 1.02)
    else:
        ax.text(0.05, 0.5, "High-N confirmation was not run in this lightweight audit call.", transform=ax.transAxes)
    ax.set_ylabel("z=60 mm correlation to V0")
    ax.set_title("MODE 2U selected high-N confirmation")
    _save_highres(fig, path)
    _close(fig)
    return path


def _write_publication_focus_panels(
    output_dir: Path,
    *,
    config: NathanSourceParityConfig | None,
    best_compromise: Mapping[str, Any],
    grid_n: int,
    z_planes: int,
    z_start_m: float,
    z_end_m: float,
) -> list[dict[str, Any]]:
    """Write selected publication-grade focus panels at a grid above the visual audit grid."""

    cfg = replace(
        config or NathanSourceParityConfig(),
        grid_n=int(grid_n),
        z_planes=int(z_planes),
        z_start_m=float(z_start_m),
        z_end_m=float(z_end_m),
        z_span_m=None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data = mode2n_source_target(cfg, grid_n=int(grid_n), z_planes=int(z_planes))
    v0 = run_mode2n_v0_reference(data)
    backward = run_mode2q_backward_initialisation(data)
    route = run_mode2n_dual_slm_4f_route(data, v0)
    pert = Mode2SPerturbation(
        label=f"publication_best_compromise_{best_compromise['case_id']}",
        slm_aperture_clip=True,
        phase_levels=256,
        fill_factor=0.95,
        carrier_lpmm=float(best_compromise["carrier_lpmm"]),
        iris_radius_frac=float(best_compromise["iris_radius_frac"]),
    )
    corr = Mode2SCorrection(
        qwp_angle_correction_rad=float(np.deg2rad(float(best_compromise["qwp_angle_correction_deg"]))),
        sector_rotation_rad=float(np.deg2rad(float(best_compromise["sector_rotation_deg"]))),
        global_v_piston_rad=float(best_compromise["global_v_piston_rad"]),
    )
    comp = run_mode2s_degraded_forward(data, v0, backward, pert, correction=corr)
    rows: list[dict[str, Any]] = []

    def _triplet(case_id: str, plane: np.ndarray, xz: np.ndarray, yz: np.ndarray, zvals: np.ndarray, metrics: Mapping[str, Any]) -> None:
        safe = str(case_id).replace("/", "_").replace("\\", "_")
        _plot_beam_image(
            plane,
            data["grid"],
            title=_metric_title("publication numerical focus", safe, metrics),
            path=output_dir / f"publicationN{grid_n}_{safe}_xy_focus.png",
            log_version_path=output_dir / f"publicationN{grid_n}_{safe}_xy_focus_log.png",
        )
        _plot_xz_map(xz, zvals, data["grid"], title=f"Publication N={grid_n} {safe} x-z focus", path=output_dir / f"publicationN{grid_n}_{safe}_xz_focus.png")
        _plot_yz_map(yz, zvals, data["grid"], title=f"Publication N={grid_n} {safe} y-z focus", path=output_dir / f"publicationN{grid_n}_{safe}_yz_focus.png")
        rows.append({
            "case_id": case_id,
            "grid_n": int(grid_n),
            "z_planes": int(z_planes),
            "z_start_mm": float(z_start_m / 1e-3),
            "z_end_mm": float(z_end_m / 1e-3),
            "z60_full_field_correlation": float(metrics.get("correlation_to_reference", 1.0)),
            "strict_class": str(metrics["strict_class"]),
            "passes": bool(metrics["passes"]),
            "notes": "publication-grade selected focus panel set with tight xy/x-z/y-z crops",
        })

    _triplet("publication_clean_source_v0_reference", v0.reference_plane, v0.xz_map, v0.yz_map, v0.z_values_m, _result_metrics(v0))
    _triplet("publication_realistic_dual_slm_4f_baseline", route.reference_plane, route.xz_map, route.yz_map, route.z_values_m, _result_metrics(route))
    _triplet(f"publication_best_compromise_{best_compromise['case_id']}", comp["reference_plane"], comp["xz_map"], comp["yz_map"], comp["z_values_m"], _case_metrics(comp))
    return rows


def _write_build_plan(out: Path, bests: Mapping[str, Mapping[str, Any]], m2s_outcome: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    compromise = bests["best_compromise"]
    carrier_lpmm = float(compromise["carrier_lpmm"])
    iris_radius_frac = float(compromise["iris_radius_frac"])
    qwp_trim_deg = float(compromise["qwp_angle_correction_deg"])
    components = [
        {"order": 1, "component": "1030 nm Gaussian source", "setting": "beam radius 2 mm at source/SLM scale", "purpose": "illuminate both H/V channels"},
        {"order": 2, "component": "PBS / H-V split", "setting": "balanced H/V power", "purpose": "feed two phase-only SLM channels"},
        {"order": 3, "component": "SLM-H", "setting": "phase mask phi_H + carrier", "purpose": "encode H channel sector phase"},
        {"order": 4, "component": "SLM-V", "setting": "phase mask phi_V + carrier", "purpose": "encode V channel sector phase"},
        {"order": 5, "component": "4F relay and iris", "setting": f"compromise carrier {carrier_lpmm:.2f} lp/mm, iris {iris_radius_frac:.2f} of carrier", "purpose": "select first diffraction order"},
        {"order": 6, "component": "QWP", "setting": f"nominal -45 deg plus {qwp_trim_deg:+.2f} deg trim in the model convention", "purpose": "convert linear channel basis into required vector field"},
        {"order": 7, "component": "source-scale physical axicon", "setting": "n=1.458, apex 176 deg", "purpose": "generate the vector-Bessel hexagon"},
        {"order": 8, "component": "camera", "setting": "near z=60 mm, align to beam/axicon axis", "purpose": "measure z=60 mm and z-stack response"},
    ]
    _write_rows(out / "realistic_build_plan_component_table.csv", components)
    (out / "realistic_build_plan_component_table.json").write_text(json.dumps(_json_ready(components), indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(13.0, 4.0), constrained_layout=True)
    ax.axis("off")
    x = np.linspace(0.04, 0.96, len(components))
    for idx, (xx, comp) in enumerate(zip(x, components, strict=True)):
        ax.text(xx, 0.62, comp["component"], ha="center", va="center", fontsize=8, bbox={"boxstyle": "round,pad=0.3", "fc": "#edf2f7", "ec": "#2d3748"})
        if idx < len(components) - 1:
            ax.annotate("", xy=(x[idx + 1] - 0.035, 0.62), xytext=(xx + 0.035, 0.62), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.text(0.5, 0.20, "Dual phase-only SLM + 4F first-order filter + QWP + physical axicon replaces segmented polarizer assembly.", ha="center", fontsize=11)
    p = out / "realistic_build_plan_block_diagram.png"
    _save_highres(fig, p)
    _close(fig)

    recommendation = {
        "source_scale_branch_validated": True,
        "microfabrication_branch_validated": False,
        "recommended_build_route": "dual-SLM + carrier + shared 4F first-order iris + QWP + source-scale physical axicon",
        "recommended_qwp_angle": f"-45 deg nominal model convention plus {qwp_trim_deg:+.2f} deg trim for the compromise point",
        "recommended_hv_phase_convention": "H mask +alpha, V mask -alpha + pi/2 before the -45 deg QWP; M2Q backward masks are equivalent in the clean bench",
        "recommended_carrier_frequency": carrier_lpmm,
        "recommended_iris_radius": iris_radius_frac,
        "validated_baseline_carrier_frequency": float(MODE2N_DEFAULT_CARRIER_LPMM),
        "validated_baseline_iris_radius": float(MODE2N_DEFAULT_IRIS_RADIUS_FRAC),
        "best_shape_case_id": str(bests["best_shape"]["case_id"]),
        "best_power_case_id": str(bests["best_power"]["case_id"]),
        "best_compromise_case_id": str(bests["best_compromise"]["case_id"]),
        "six_polarizer_route_needed": False,
        "six_polarizer_route_note": "Not needed for the source-scale build; keep only as a conceptual fallback if dual-SLM access is unavailable.",
        "biggest_practical_sensitivities": [
            "hologram centre to axicon axis registration",
            "real SLM active-area fit and orientation",
            "4F first-order iris centring",
            "QWP angle convention and H/V channel power balance",
        ],
        "m2s_outcome": str(m2s_outcome.get("suggested_outcome", "")),
        "next_recommended_stage": "MODE 2T/2V lab implementation package: export SLM masks, waveplate settings, 4F stop sizing, and alignment workflow",
    }
    (out / "realistic_build_plan_recommendation.md").write_text(_build_recommendation_md(recommendation, components), encoding="utf-8")
    return recommendation, components


def _build_recommendation_md(recommendation: Mapping[str, Any], components: Sequence[Mapping[str, Any]]) -> str:
    rows = "\n".join(
        f"| {c['order']} | {c['component']} | {c['setting']} | {c['purpose']} |" for c in components
    )
    return f"""# MODE 2U Realistic Source-Scale Build Recommendation

**Recommended route:** {recommendation['recommended_build_route']}

The weird six-polarizer / segmented-polarizer assembly is **not needed** for the
source-scale build. It is replaced by two phase-only SLM channels, a carrier/4F
first-order filter, a uniform QWP, and the source-scale physical axicon.

| order | component | setting | purpose |
|---:|---|---|---|
{rows}

Key conventions:

- QWP: {recommendation['recommended_qwp_angle']}
- H/V phase convention: {recommendation['recommended_hv_phase_convention']}
- Carrier: {recommendation['recommended_carrier_frequency']} lp/mm
- Iris radius fraction: {recommendation['recommended_iris_radius']}

Biggest sensitivities: {', '.join(recommendation['biggest_practical_sensitivities'])}.

This is still a source-scale optical prediction. It does not validate the
separate microfabrication/sample-plane branch.
"""


def _write_master_doc(
    path: Path,
    *,
    output_root: Path,
    recommendation: Mapping[str, Any],
    outcome: Mapping[str, Any],
    optimal: Mapping[str, Mapping[str, Any]],
    fit_report: Mapping[str, Any],
    high_n_rows: Sequence[Mapping[str, Any]],
) -> Path:
    high_n_note = "; ".join(
        f"{r['case_id']} N={r['grid_n']} corr={float(r.get('z60_full_field_correlation', np.nan)):.4f} class={r.get('strict_class')}"
        for r in high_n_rows
        if np.isfinite(float(r.get("z60_full_field_correlation", np.nan)))
    ) or "High-N confirmation was not run for this lightweight call."
    text = f"""# Nathan MODE 2U - Master High-Resolution Audit And Build Plan

**Status:** source-scale high-resolution audit and build-planning layer. It does
not change the validated MODE 2P/2N/2Q/2S physics and it makes no
microfabrication/sample-plane success claim.

## Regeneration

The audit regenerated high-resolution PNG figures under
`{output_root.as_posix()}` for V0, M2P, M2N, M2Q, M2S, historical MODE 1
contrast, energy ledgers, profile response, optimal-hexagon sweep, high-N focus
panels, and the realistic build plan. Rendered beam panels use focus crops and
`{MODE2U_RENDER_INTERPOLATION}` interpolation so the sub-mm structure is not
lost in the 10 mm source window.

The file `publication_sampling_audit.csv/json` separates numerical Nyquist
adequacy from publication-grade visual adequacy. In this model the axicon radial
fringe period is about 64 um, so grids near N=1536 on the 10 mm source window are
the first practical publication-recommended setting by the stricter samples per
fringe and ring-diameter-pixels criteria.

## Source-Scale Visual Outcome

The source-scale branch remains visually and metrically confirmed. MODE 2S
outcome is `{outcome.get('suggested_outcome')}`. The SLM active window does not
fit the 10 mm source field vertically (`window_fits_vertically =
{fit_report.get('window_fits_vertically')}`), but the 2 mm Gaussian beam clips
only `{float(fit_report.get('beam_power_clipped_by_active_area_fraction', np.nan)):.3e}` of its power.

The microfabrication/sample-objective branch remains a separate negative
contrast branch and is not unblocked by MODE 2U.

## Energy And Sensitivity

The dominant explicit power loss in the practical source-scale route is the 4F
first-order selection/rejected order budget. The most dangerous shape
misalignment remains hologram/axicon axis registration; bounded recentering is
the practical correction.

## Optimal Operating Points

- Best shape: `{optimal['best_shape']['case_id']}` with shape score `{float(optimal['best_shape']['shape_score']):.4f}`.
- Best power: `{optimal['best_power']['case_id']}` with power score `{float(optimal['best_power']['power_score']):.4f}`.
- Recommended compromise: `{optimal['best_compromise']['case_id']}` with combined score `{float(optimal['best_compromise']['combined_score']):.4f}`.

## High-N Confirmation

{high_n_note}

## Build Recommendation

Recommended route: **{recommendation['recommended_build_route']}**.

The six-polarizer/segmented-polarizer route is not needed for the source-scale
implementation. It is replaced by the dual-SLM carrier/4F route, a uniform QWP,
and a source-scale physical axicon. The practical next stage is
`{recommendation['next_recommended_stage']}`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_mode2u_master_highres_audit(
    config: NathanSourceParityConfig | None = None,
    *,
    output_dir: str | Path = MODE2U_DEFAULT_OUTPUT_ROOT,
    grid_n: int = 384,
    z_planes: int = 9,
    run_compensation: bool = True,
    compensation_maxiter: int = 24,
    optimisation_max_cases: int | None = None,
    run_high_n: bool = True,
    high_n_values: Sequence[int] = (1024,),
    high_n_z_planes: int = 9,
    run_publication_focus: bool | None = None,
    publication_focus_grid_n: int = 1536,
    publication_focus_z_planes: int = 41,
    publication_focus_z_start_m: float = 30.0e-3,
    publication_focus_z_end_m: float = 90.0e-3,
    doc_path: str | Path = MODE2U_DOC_PATH,
) -> dict[str, Any]:
    """Generate the MODE 2U master audit products in a separate output root."""

    import matplotlib.pyplot as plt

    plt.rcParams["image.interpolation"] = MODE2U_RENDER_INTERPOLATION
    plt.rcParams["image.resample"] = True
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["figure.dpi"] = 120

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    dirs = {key: root / value for key, value in MODE2U_SUBDIRS.items()}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    data = mode2n_source_target(config, grid_n=int(grid_n), z_planes=int(z_planes))
    v0 = run_mode2n_v0_reference(data)
    patterned = run_mode2n_patterned_hwp_route(data, v0)
    dual_qwp = run_mode2n_dual_slm_qwp_route(data, v0)
    dual_4f = run_mode2n_dual_slm_4f_route(data, v0)
    m2n_routes = [v0, patterned, dual_qwp, dual_4f]
    m2p = _build_m2p_source_scale(data)
    m2q = run_mode2q_backward_mask_synthesis(config, grid_n=int(grid_n), z_planes=int(z_planes), run_optimisation=False)
    m2s = run_mode2s_lab_realism(
        config,
        grid_n=int(grid_n),
        z_planes=int(z_planes),
        run_compensation=bool(run_compensation),
        compensation_maxiter=int(compensation_maxiter),
        max_compensated_cases=2,
    )

    visual_rows: list[dict[str, Any]] = []
    generated: dict[str, str] = {}
    for mapping in (
        _write_v0_reference(dirs["v0"], data, v0, visual_rows, root),
        _write_m2p(dirs["m2p"], m2p, visual_rows, root),
        _write_m2n(dirs["m2n"], data, m2n_routes, v0, visual_rows, root),
        _write_m2q(dirs["m2q"], m2q, visual_rows, root),
        _write_m2s(dirs["m2s"], dirs["profiles"], m2s, visual_rows, root),
        _write_mode1_contrast(dirs["mode1"], dual_4f, visual_rows, root),
    ):
        generated.update({key: _rel(path, root) for key, path in mapping.items()})

    energy_rows = _build_energy_ledgers(m2n_routes, m2q, m2s)
    _write_rows(dirs["energy"] / "energy_ledger_routes.csv", energy_rows)
    (dirs["energy"] / "energy_ledger_routes.json").write_text(json.dumps(_json_ready(energy_rows), indent=2), encoding="utf-8")
    _plot_energy_ledgers(energy_rows, dirs["energy"])

    optimal_rows, optimal_cases = _run_optimal_sweep(data, v0, m2s["backward"], max_cases=optimisation_max_cases)
    optimal = _write_optimal_outputs(dirs["optimal"], data, v0, optimal_rows, optimal_cases, visual_rows, root)

    sampling_rows = _sampling_audit_rows(
        config,
        grid_values=(384, 512, 768, 1024, 1536, 2048),
        carrier_lpmm=float(optimal["best_compromise"]["carrier_lpmm"]),
        iris_radius_frac=float(optimal["best_compromise"]["iris_radius_frac"]),
        ring_radius_m=float(v0.ring_radius_m),
    )
    _write_rows(root / "publication_sampling_audit.csv", sampling_rows)
    (root / "publication_sampling_audit.json").write_text(json.dumps(_json_ready(sampling_rows), indent=2), encoding="utf-8")
    _plot_sampling_audit(sampling_rows, root / "publication_sampling_audit_plot.png")

    high_n_rows = _write_high_n_confirmation(
        root,
        config=config,
        base_rows={
            "best_shape": optimal["best_shape"],
            "best_power": optimal["best_power"],
            "best_compromise": optimal["best_compromise"],
        },
        run_high_n=bool(run_high_n),
        high_n_values=tuple(int(v) for v in high_n_values),
        z_planes=int(high_n_z_planes),
        focus_output_dir=dirs["highn_focus"],
    )
    should_publication_focus = bool(run_high_n) if run_publication_focus is None else bool(run_publication_focus)
    if should_publication_focus:
        publication_rows = _write_publication_focus_panels(
            dirs["highn_focus"],
            config=config,
            best_compromise=optimal["best_compromise"],
            grid_n=int(publication_focus_grid_n),
            z_planes=int(publication_focus_z_planes),
            z_start_m=float(publication_focus_z_start_m),
            z_end_m=float(publication_focus_z_end_m),
        )
        high_n_rows.extend(publication_rows)
        _write_rows(root / "highN_confirmation.csv", high_n_rows)
        (root / "highN_confirmation.json").write_text(json.dumps(_json_ready(high_n_rows), indent=2), encoding="utf-8")
        _plot_high_n_summary(high_n_rows, root / "highN_confirmation_summary.png")

    recommendation, components = _write_build_plan(dirs["build"], optimal, m2s["outcome"])
    (root / "nathan_master_recommendation.json").write_text(json.dumps(_json_ready(recommendation), indent=2), encoding="utf-8")
    _write_rows(root / "nathan_master_visual_audit.csv", visual_rows)
    (root / "nathan_master_visual_audit.json").write_text(json.dumps(_json_ready(visual_rows), indent=2), encoding="utf-8")
    _write_master_doc(
        Path(doc_path),
        output_root=root,
        recommendation=recommendation,
        outcome=m2s["outcome"],
        optimal=optimal,
        fit_report=m2s["fit_report"],
        high_n_rows=high_n_rows,
    )

    manifest = {
        "stage": MODE2U_STAGE,
        "output_root": str(root),
        "subdirectories": {k: _rel(v, root) for k, v in dirs.items()},
        "included_stages": ["V0", "M2P", "M2N", "M2Q", "M2S", "MODE1_CONTRAST", "ENERGY", "PROFILES", "OPTIMAL", "BUILD_PLAN"],
        "grid_n": int(grid_n),
        "z_planes": int(z_planes),
        "highres_png_dpi_minimum": 300,
        "render_interpolation": MODE2U_RENDER_INTERPOLATION,
        "focus_crop_fraction": MODE2U_FOCUS_CROP_FRACTION,
        "publication_sampling_min_fringe_samples": MODE2U_PUBLICATION_MIN_FRINGE_SAMPLES,
        "publication_sampling_min_ring_diameter_pixels": MODE2U_PUBLICATION_MIN_RING_DIAMETER_PIXELS,
        "publication_sampling_max_carrier_band_fraction_of_nyquist": MODE2U_MAX_CARRIER_BAND_FRACTION_OF_NYQUIST,
        "publication_focus_grid_n": int(publication_focus_grid_n) if should_publication_focus else None,
        "publication_focus_z_planes": int(publication_focus_z_planes) if should_publication_focus else None,
        "publication_focus_z_range_mm": [
            float(publication_focus_z_start_m / 1e-3),
            float(publication_focus_z_end_m / 1e-3),
        ] if should_publication_focus else None,
        "physics_changed": False,
        "source_scale_branch_validated": True,
        "microfabrication_branch_validated": False,
        "microfabrication_sample_plane_claim": False,
        "m2s_outcome": str(m2s["outcome"].get("suggested_outcome", "")),
        "recommended_build_route": recommendation["recommended_build_route"],
        "six_polarizer_route_needed": bool(recommendation["six_polarizer_route_needed"]),
        "master_doc": str(Path(doc_path)),
        "machine_files": {
            "manifest": "nathan_master_highres_manifest.json",
            "visual_audit_csv": "nathan_master_visual_audit.csv",
            "visual_audit_json": "nathan_master_visual_audit.json",
            "recommendation_json": "nathan_master_recommendation.json",
            "publication_sampling_audit_csv": "publication_sampling_audit.csv",
            "publication_sampling_audit_json": "publication_sampling_audit.json",
        },
        "generated_key_figures": generated,
        "scope_manifest_m2s": mode2s_scope_manifest(m2s["outcome"]),
    }
    (root / "nathan_master_highres_manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")
    return {
        "manifest": manifest,
        "visual_rows": visual_rows,
        "energy_rows": energy_rows,
        "sampling_rows": sampling_rows,
        "optimal": optimal,
        "recommendation": recommendation,
        "components": components,
        "high_n_rows": high_n_rows,
        "output_root": root,
        "doc_path": Path(doc_path),
    }


__all__ = [
    "MODE2U_STAGE",
    "MODE2U_DEFAULT_OUTPUT_ROOT",
    "MODE2U_DOC_PATH",
    "MODE2U_SUBDIRS",
    "mode2u_visual_acceptability",
    "write_mode2u_master_highres_audit",
]
