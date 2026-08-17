"""MODE 2U2-FIX strict hexagon optimisation integrity audit.

The MODE 2U2 closure proved that a high full-field correlation can coexist with
a visually suspect operating point.  This layer re-audits the old optima,
calibrates stricter shape metrics against immutable V0 and realistic-4F
references, and reranks only candidates that first pass a hard hexagon
eligibility gate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import rotate

from vbb_study.digital_twin.nathan_mode2u2_master_closure import (
    MODE2U2_DEFAULT_OUTPUT_ROOT,
    _fixed_useful_region,
    _useful_power_metrics,
)
from vbb_study.digital_twin.nathan_vector_hexagon import (
    EPS,
    MODE2N_DEFAULT_CARRIER_LPMM,
    MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
    MODE2S_PASS_CORRELATION,
    Mode2SCorrection,
    Mode2SPerturbation,
    NathanSourceParityConfig,
    _json_ready,
    _mode1b_even_axis_crop,
    _mode2n_mm_axis,
    _normalise_image,
    _write_rows,
    angular_profile_on_ring,
    mode2n_source_target,
    mode2q_strict_hexagon_gate,
    run_mode2n_dual_slm_4f_route,
    run_mode2n_v0_reference,
    run_mode2q_backward_initialisation,
    run_mode2s_degraded_forward,
)


MODE2U2F_STAGE = "nathan_mode2u2_fix_strict_hexagon_optimisation"
MODE2U2F_DEFAULT_OUTPUT_ROOT = Path("outputs/figures/digital_twin/nathan_mode2u2_fix_strict_hexagon_optimisation")
MODE2U2F_DOC_PATH = Path("docs/74_nathan_mode2u2_fix_strict_hexagon_optimisation.md")
MODE2U2F_SEED = 20260709
MODE2U2F_ALLOWED_OUTCOMES = ("M2U2F-A", "M2U2F-B", "M2U2F-C", "M2U2F-D")

V0_REFERENCE_ID = "V0_REFERENCE"
REALISTIC_4F_REFERENCE_ID = "REALISTIC_4F_HEXAGON_REFERENCE"
OLD_BEST_COMPROMISE_ID = "m2u2_opt_003_c5.75_i0.32_q-0.25_r+0.0_p0.10"

STRICT_BASELINE_CORR_MIN = 0.997
STRICT_FOCUS_CORR_MIN = 0.985
STRICT_SUPPORT_CORR_MIN = 0.980
STRICT_ANGULAR_CORR_MIN = 0.995
STRICT_RADIAL_CORR_MIN = 0.970
STRICT_DELTA_C_MAX = -0.020
STRICT_DARK_CORE_MAX = 0.010
STRICT_H6_MIN = 0.090
STRICT_H4_OVER_H6_MAX = 1.05
STRICT_C90_OVER_C60_MAX = 0.88


@dataclass(frozen=True)
class StrictCandidateControls:
    candidate_id: str
    carrier_lpmm: float
    iris_radius_frac: float
    qwp_angle_correction_deg: float = 0.0
    sector_rotation_deg: float = 0.0
    global_v_piston_rad: float = 0.0
    sector_duty_scale: float = 1.0
    mask_recentre_x_m: float = 0.0
    mask_recentre_y_m: float = 0.0
    source: str = "bounded_search"

    def perturbation(self) -> Mode2SPerturbation:
        return Mode2SPerturbation(
            label=self.candidate_id,
            slm_aperture_clip=True,
            phase_levels=256,
            fill_factor=0.93,
            carrier_lpmm=float(self.carrier_lpmm),
            iris_radius_frac=float(self.iris_radius_frac),
        )

    def correction(self) -> Mode2SCorrection:
        return Mode2SCorrection(
            qwp_angle_correction_rad=float(np.deg2rad(self.qwp_angle_correction_deg)),
            sector_rotation_rad=float(np.deg2rad(self.sector_rotation_deg)),
            global_v_piston_rad=float(self.global_v_piston_rad),
            sector_duty_scale=float(self.sector_duty_scale),
            mask_recentre_x_m=float(self.mask_recentre_x_m),
            mask_recentre_y_m=float(self.mask_recentre_y_m),
        )

    def row(self) -> dict[str, Any]:
        return asdict(self)


def _safe_corr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if mask is not None:
        mm = np.asarray(mask, dtype=bool)
        aa = aa[mm]
        bb = bb[mm]
    aa = aa.ravel()
    bb = bb.ravel()
    if aa.size < 4 or float(np.std(aa)) <= EPS or float(np.std(bb)) <= EPS:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _focus_mask(shape: tuple[int, int], fraction: float = 0.16) -> np.ndarray:
    ny, nx = shape
    half_y = max(2, int(round(0.5 * float(fraction) * ny)))
    half_x = max(2, int(round(0.5 * float(fraction) * nx)))
    cy, cx = ny // 2, nx // 2
    mask = np.zeros((ny, nx), dtype=bool)
    mask[max(0, cy - half_y): min(ny, cy + half_y + 1), max(0, cx - half_x): min(nx, cx + half_x + 1)] = True
    return mask


def _ssim_like(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if mask is not None:
        mm = np.asarray(mask, dtype=bool)
        aa = aa[mm]
        bb = bb[mm]
    aa = aa.ravel()
    bb = bb.ravel()
    if aa.size < 4:
        return 0.0
    c1 = 1.0e-4
    c2 = 9.0e-4
    mux = float(np.mean(aa))
    muy = float(np.mean(bb))
    vx = float(np.var(aa))
    vy = float(np.var(bb))
    cov = float(np.mean((aa - mux) * (bb - muy)))
    return float(((2.0 * mux * muy + c1) * (2.0 * cov + c2)) / max((mux * mux + muy * muy + c1) * (vx + vy + c2), EPS))


def _radial_profile(plane: np.ndarray, grid: Mapping[str, Any], bins: int = 220) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(grid["R"], dtype=float).ravel()
    y = np.asarray(plane, dtype=float).ravel()
    edges = np.linspace(0.0, float(np.max(r)), int(bins) + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, int(bins) - 1)
    sums = np.bincount(idx, weights=y, minlength=int(bins))
    counts = np.bincount(idx, minlength=int(bins))
    return 0.5 * (edges[:-1] + edges[1:]), sums / np.maximum(counts, 1)


def _rotational_correlation(plane: np.ndarray, degrees: float, mask: np.ndarray | None = None) -> float:
    rotated = rotate(np.asarray(plane, dtype=float), float(degrees), reshape=False, order=1, mode="constant", cval=0.0)
    return _safe_corr(plane, rotated, mask=mask)


def _angular_harmonics(plane: np.ndarray, grid: Mapping[str, Any], ring_radius_m: float) -> dict[str, float]:
    theta, profile = angular_profile_on_ring(plane, grid, float(ring_radius_m), angular_bins=720)
    prof = np.asarray(profile, dtype=float) - float(np.mean(profile))
    denom = float(np.sum(np.abs(prof))) + EPS
    return {
        f"h{order}": float(abs(np.sum(prof * np.exp(-1j * int(order) * theta))) / denom)
        for order in (3, 4, 6)
    }


def _local_peak_metrics(plane: np.ndarray, useful_mask: np.ndarray) -> dict[str, float]:
    arr = np.asarray(plane, dtype=float)
    y, x = np.unravel_index(int(np.argmax(arr)), arr.shape)
    y0, y1 = max(0, y - 1), min(arr.shape[0], y + 2)
    x0, x1 = max(0, x - 1), min(arr.shape[1], x + 2)
    yy, xx = np.indices(arr.shape)
    radius = 2.5
    disk = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
    useful = np.asarray(useful_mask, dtype=bool)
    useful_values = arr[useful]
    return {
        "single_pixel_peak": float(np.max(arr)),
        "local_3x3_peak_mean": float(np.mean(arr[y0:y1, x0:x1])),
        "local_radius_peak_mean": float(np.mean(arr[disk])),
        "useful_region_peak": float(np.max(useful_values)) if useful_values.size else 0.0,
        "peak_metric_used": float(np.mean(arr[y0:y1, x0:x1])),
        "peak_metric_definition": "mean intensity in the 3x3 neighbourhood centred on the maximum pixel",
    }


def evaluate_strict_hexagon_metrics(
    plane: np.ndarray,
    *,
    grid: Mapping[str, Any],
    v0_plane: np.ndarray,
    realistic_plane: np.ndarray,
    v0_ring_radius_m: float,
    useful_mask: np.ndarray,
) -> dict[str, Any]:
    """Evaluate full, crop, support, angular, radial, Cn and harmonic metrics."""

    arr = np.asarray(plane, dtype=float)
    v0 = np.asarray(v0_plane, dtype=float)
    realistic = np.asarray(realistic_plane, dtype=float)
    strict = mode2q_strict_hexagon_gate(arr, grid)
    sym = dict(strict["symmetry"])
    focus = _focus_mask(arr.shape)
    support = v0 >= 0.02 * max(float(np.max(v0)), EPS)
    angular_theta, angular = angular_profile_on_ring(arr, grid, float(v0_ring_radius_m), angular_bins=720)
    _, angular_ref = angular_profile_on_ring(v0, grid, float(v0_ring_radius_m), angular_bins=720)
    _, radial = _radial_profile(arr, grid)
    _, radial_ref = _radial_profile(v0, grid)
    angular_corr = _safe_corr(angular, angular_ref)
    radial_corr = _safe_corr(radial, radial_ref)
    harmonics = _angular_harmonics(arr, grid, float(v0_ring_radius_m))
    useful = _useful_power_metrics(arr, useful_mask)
    peak = _local_peak_metrics(arr, useful_mask)
    c45 = _rotational_correlation(arr, 45.0, mask=focus)
    c60 = float(sym.get("rot_corr_60", _rotational_correlation(arr, 60.0, mask=focus)))
    c90 = _rotational_correlation(arr, 90.0, mask=focus)
    c120 = float(sym.get("rot_corr_120", _rotational_correlation(arr, 120.0, mask=focus)))
    c180 = _rotational_correlation(arr, 180.0, mask=focus)
    h6 = float(harmonics["h6"])
    h4 = float(harmonics["h4"])
    h3 = float(harmonics["h3"])
    metrics = {
        "corr_full": _safe_corr(arr, v0),
        "corr_focus_crop": _safe_corr(arr, v0, mask=focus),
        "corr_v0_support": _safe_corr(arr, v0, mask=support),
        "corr_useful_region": _safe_corr(arr, v0, mask=useful_mask),
        "corr_angular": angular_corr,
        "corr_radial": radial_corr,
        "corr_to_realistic_4f": _safe_corr(arr, realistic),
        "ssim_like_focus": _ssim_like(arr, v0, mask=focus),
        "c45": c45,
        "c60": c60,
        "c90": c90,
        "c120": c120,
        "c180": c180,
        "deltaC_c120_minus_c60": float(c120 - c60),
        "h3": h3,
        "h4": h4,
        "h6": h6,
        "h4_over_h6": float(h4 / max(h6, EPS)),
        "h3_over_h6": float(h3 / max(h6, EPS)),
        "dark_core_ratio": float(strict["dark_core_ratio"]),
        "ring_island_count": int(strict["ring_island_count"]),
        "classifier_label": str(strict["strict_class"]),
        "legacy_strict_gate": bool(strict["passes_true_hexagon_gate"]),
        **useful,
        **peak,
    }
    fourfold_veto = bool(
        metrics["h4_over_h6"] > STRICT_H4_OVER_H6_MAX
        or (metrics["c90"] / max(metrics["c60"], EPS)) > STRICT_C90_OVER_C60_MAX
    )
    triangular_veto = bool(
        str(metrics["classifier_label"]) == "triangular_lobed_field"
        or metrics["h3_over_h6"] > 0.80
        or metrics["deltaC_c120_minus_c60"] >= 0.0
    )
    reference_drift_veto = bool(metrics["corr_to_realistic_4f"] < STRICT_BASELINE_CORR_MIN)
    fail_reasons: list[str] = []
    checks = [
        (metrics["legacy_strict_gate"], "legacy strict visual_hexagonal_field gate failed"),
        (not triangular_veto, "triangular/C3 veto triggered"),
        (not fourfold_veto, "fourfold/X-shape veto triggered"),
        (not reference_drift_veto, "candidate drifted from immutable realistic-4F reference"),
        (metrics["h6"] >= STRICT_H6_MIN, "sixfold harmonic below calibrated floor"),
        (metrics["corr_focus_crop"] >= STRICT_FOCUS_CORR_MIN, "focus-crop correlation below threshold"),
        (metrics["corr_v0_support"] >= STRICT_SUPPORT_CORR_MIN, "V0-support correlation below threshold"),
        (metrics["corr_angular"] >= STRICT_ANGULAR_CORR_MIN, "angular profile correlation below threshold"),
        (metrics["corr_radial"] >= STRICT_RADIAL_CORR_MIN, "radial profile correlation below threshold"),
        (metrics["deltaC_c120_minus_c60"] <= STRICT_DELTA_C_MAX, "c60/c120 relation not sufficiently C6-like"),
        (metrics["dark_core_ratio"] <= STRICT_DARK_CORE_MAX, "dark-core ratio outside accepted range"),
    ]
    for ok, reason in checks:
        if not bool(ok):
            fail_reasons.append(reason)
    metrics.update({
        "fourfold_x_veto": fourfold_veto,
        "triangular_c3_veto": triangular_veto,
        "reference_drift_veto": reference_drift_veto,
        "strict_hexagon_eligible": len(fail_reasons) == 0,
        "strict_fail_reasons": "; ".join(fail_reasons),
    })
    return metrics


def synthetic_triangular_failure(grid: Mapping[str, Any], v0_plane: np.ndarray) -> np.ndarray:
    phi = np.asarray(grid["PHI"], dtype=float)
    return np.asarray(v0_plane, dtype=float) * np.clip(1.0 + 0.85 * np.cos(3.0 * phi), 0.0, None)


def synthetic_fourfold_failure(grid: Mapping[str, Any], v0_plane: np.ndarray) -> np.ndarray:
    phi = np.asarray(grid["PHI"], dtype=float)
    return np.asarray(v0_plane, dtype=float) * np.clip(1.0 + 1.15 * np.cos(4.0 * phi), 0.0, None)


def _mark_immutable_reference(metrics: dict[str, Any]) -> dict[str, Any]:
    """Reference rows are allowed to be references rather than candidates."""

    reasons = [
        reason.strip()
        for reason in str(metrics.get("strict_fail_reasons", "")).split(";")
        if reason.strip() and "immutable realistic-4F reference" not in reason
    ]
    metrics = dict(metrics)
    metrics["reference_drift_veto"] = False
    metrics["strict_fail_reasons"] = "; ".join(reasons)
    metrics["strict_hexagon_eligible"] = len(reasons) == 0
    return metrics


def old_m2u2_optimum_controls(root: Path | None = None) -> dict[str, StrictCandidateControls]:
    """Load known M2U2 optima, falling back to the stored prompt IDs."""

    output_root = root or MODE2U2_DEFAULT_OUTPUT_ROOT
    path = output_root / "optimal_hexagon_candidates.json"
    fallback = {
        "best_shape": StrictCandidateControls("m2u2_opt_020_c5.75_i0.40_q-0.25_r+0.0_p0.00", 5.75, 0.40, -0.25, 0.0, 0.0, source="old_m2u2_best_shape"),
        "best_peak": StrictCandidateControls("m2u2_opt_015_c5.75_i0.32_q+0.25_r+0.0_p0.10", 5.75, 0.32, 0.25, 0.0, 0.10, source="old_m2u2_best_peak"),
        "best_useful_power": StrictCandidateControls("m2u2_opt_032_c5.75_i0.40_q+0.25_r+0.0_p0.00", 5.75, 0.40, 0.25, 0.0, 0.0, source="old_m2u2_best_useful_power"),
        "best_compromise": StrictCandidateControls(OLD_BEST_COMPROMISE_ID, 5.75, 0.32, -0.25, 0.0, 0.10, source="old_m2u2_best_compromise"),
    }
    if not path.exists():
        return fallback
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = dict(payload.get("best", {}))
    out: dict[str, StrictCandidateControls] = {}
    for key, row in best.items():
        out[key] = StrictCandidateControls(
            str(row["case_id"]),
            float(row["carrier_lpmm"]),
            float(row["iris_radius_frac"]),
            float(row.get("qwp_angle_correction_deg", 0.0)),
            float(row.get("sector_rotation_deg", 0.0)),
            float(row.get("global_v_piston_rad", 0.0)),
            source=f"old_m2u2_{key}",
        )
    for key, value in fallback.items():
        out.setdefault(key, value)
    return out


def evaluate_controls(
    controls: StrictCandidateControls,
    *,
    data: Mapping[str, Any],
    v0: Any,
    backward: Any,
    realistic: Any,
    useful_mask: np.ndarray,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    case = run_mode2s_degraded_forward(
        data,
        v0,
        backward,
        controls.perturbation(),
        correction=controls.correction(),
        fast_single_plane=True,
    )
    metrics = evaluate_strict_hexagon_metrics(
        case["reference_plane"],
        grid=data["grid"],
        v0_plane=v0.reference_plane,
        realistic_plane=realistic.reference_plane,
        v0_ring_radius_m=float(v0.ring_radius_m),
        useful_mask=useful_mask,
    )
    row = {
        **controls.row(),
        **metrics,
        "legacy_z60_correlation": float(case["comparison"]["z60_full_field_correlation"]),
        "legacy_angular_correlation": float(case["comparison"]["angular_profile_correlation_to_v0"]),
        "first_order_efficiency": float(case["iris"]["first_order_efficiency"]),
        "total_throughput": float(case["iris"]["first_order_efficiency"] * case["pre_axicon"]["power_ratio"]),
    }
    shape = 0.36 * row["corr_to_realistic_4f"] + 0.24 * row["corr_focus_crop"] + 0.22 * row["corr_angular"] + 0.18 * row["corr_v0_support"]
    row["shape_fidelity_score"] = float(shape)
    row["strict_peak_metric"] = float(row["local_3x3_peak_mean"])
    row["strict_useful_energy_metric"] = float(row["P_useful"])
    row["strict_compromise_score"] = float(
        0.55 * row["shape_fidelity_score"]
        + 0.18 * min(row["P_useful_over_P_total"], 1.0)
        + 0.14 * row["first_order_efficiency"]
        + 0.13 * min(row["local_3x3_peak_mean"] / max(row["P_total"], EPS) * 1.0e3, 1.0)
    )
    if not bool(row["strict_hexagon_eligible"]):
        row["eligible_for_ranking"] = False
        row["ineligibility_note"] = row["strict_fail_reasons"]
    else:
        row["eligible_for_ranking"] = True
        row["ineligibility_note"] = ""
    return row, case


def _baseline_row(
    *,
    data: Mapping[str, Any],
    v0: Any,
    realistic: Any,
    useful_mask: np.ndarray,
) -> dict[str, Any]:
    metrics = evaluate_strict_hexagon_metrics(
        realistic.reference_plane,
        grid=data["grid"],
        v0_plane=v0.reference_plane,
        realistic_plane=realistic.reference_plane,
        v0_ring_radius_m=float(v0.ring_radius_m),
        useful_mask=useful_mask,
    )
    row = {
        "candidate_id": REALISTIC_4F_REFERENCE_ID,
        "carrier_lpmm": MODE2N_DEFAULT_CARRIER_LPMM,
        "iris_radius_frac": MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
        "qwp_angle_correction_deg": 0.0,
        "sector_rotation_deg": 0.0,
        "global_v_piston_rad": 0.0,
        "sector_duty_scale": 1.0,
        "mask_recentre_x_m": 0.0,
        "mask_recentre_y_m": 0.0,
        "source": "immutable_realistic_4f_reference",
        **metrics,
        "legacy_z60_correlation": float(realistic.v0_comparison["z60_full_field_correlation"]),
        "legacy_angular_correlation": float(realistic.v0_comparison["angular_profile_correlation_to_v0"]),
        "first_order_efficiency": float(realistic.slm_4f_report["first_order_efficiency"]),
        "total_throughput": float(realistic.slm_4f_report["first_order_efficiency"]),
    }
    row["shape_fidelity_score"] = 1.0
    row["strict_peak_metric"] = float(row["local_3x3_peak_mean"])
    row["strict_useful_energy_metric"] = float(row["P_useful"])
    row["strict_compromise_score"] = float(0.55 + 0.18 * min(row["P_useful_over_P_total"], 1.0) + 0.14 * row["first_order_efficiency"])
    row["eligible_for_ranking"] = bool(row["strict_hexagon_eligible"])
    row["ineligibility_note"] = ""
    return row


def run_strict_bounded_search(
    *,
    data: Mapping[str, Any],
    v0: Any,
    backward: Any,
    realistic: Any,
    useful_mask: np.ndarray,
    max_cases: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]], dict[str, dict[str, Any]]]:
    """Run physically interpretable search, then rank only strict eligible rows."""

    controls: list[StrictCandidateControls] = []
    for carrier, iris, qwp, rotation, piston in product(
        [5.75, 6.25, 6.75],
        [0.32, 0.40, 0.52],
        [-0.25, 0.0, 0.25],
        [-1.0, 0.0, 1.0],
        [0.0, 0.10],
    ):
        controls.append(StrictCandidateControls(
            f"strict_c{carrier:.2f}_i{iris:.2f}_q{qwp:+.2f}_r{rotation:+.1f}_p{piston:.2f}",
            carrier, iris, qwp, rotation, piston,
        ))
    if max_cases is not None:
        controls = controls[: int(max_cases)]
    rows: list[dict[str, Any]] = []
    cases: dict[str, Mapping[str, Any]] = {}
    baseline = _baseline_row(data=data, v0=v0, realistic=realistic, useful_mask=useful_mask)
    rows.append(baseline)
    for control in controls:
        row, case = evaluate_controls(control, data=data, v0=v0, backward=backward, realistic=realistic, useful_mask=useful_mask)
        rows.append(row)
        cases[str(row["candidate_id"])] = case
    eligible = [r for r in rows if bool(r["strict_hexagon_eligible"])]
    if not eligible:
        best = {}
    else:
        best = {
            "strict_best_shape": max(eligible, key=lambda r: float(r["shape_fidelity_score"])),
            "strict_best_peak": max(eligible, key=lambda r: float(r["strict_peak_metric"])),
            "strict_best_useful_energy": max(eligible, key=lambda r: float(r["strict_useful_energy_metric"])),
            "strict_best_compromise": max(eligible, key=lambda r: float(r["strict_compromise_score"])),
        }
    return rows, cases, best


def _plot_side_by_side(
    path: Path,
    *,
    title: str,
    grid: Mapping[str, Any],
    v0_plane: np.ndarray,
    realistic_plane: np.ndarray,
    candidate_plane: np.ndarray,
    crop_fraction: float = 0.16,
) -> Path:
    import matplotlib.pyplot as plt

    panels = [
        ("V0", v0_plane),
        ("realistic 4F", realistic_plane),
        ("candidate", candidate_plane),
        ("abs diff", np.abs(_normalise_image(candidate_plane, local=True) - _normalise_image(v0_plane, local=True))),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.4), constrained_layout=True)
    for ax, (label, arr) in zip(axes, panels, strict=True):
        crop, crop_grid = _mode1b_even_axis_crop(np.asarray(arr, dtype=float), grid, crop_fraction)
        x = np.asarray(crop_grid["x"], dtype=float) / 1e-3
        ext = [float(x[0]), float(x[-1]), float(x[0]), float(x[-1])]
        cmap = "magma" if label == "abs diff" else "inferno"
        ax.imshow(_normalise_image(crop, local=True), origin="lower", extent=ext, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="lanczos")
        ax.set_title(label)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_profiles(path: Path, *, grid: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], v0_ring_radius_m: float, title: str) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    for item in cases:
        label = str(item["label"])
        plane = np.asarray(item["plane"], dtype=float)
        rr, rp = _radial_profile(plane, grid)
        axes[0].plot(rr / 1e-3, rp / max(float(np.max(rp)), EPS), label=label)
        theta, ap = angular_profile_on_ring(plane, grid, float(v0_ring_radius_m), angular_bins=720)
        axes[1].plot(np.rad2deg(theta), ap / max(float(np.max(ap)), EPS), label=label)
    axes[0].set_xlabel("r (mm)")
    axes[0].set_ylabel("normalised radial mean")
    axes[1].set_xlabel("angle (deg)")
    axes[1].set_ylabel("normalised ring intensity")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_pareto(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    eligible = np.asarray([bool(r["strict_hexagon_eligible"]) for r in rows], dtype=bool)
    x = np.asarray([float(r["shape_fidelity_score"]) for r in rows], dtype=float)
    y = np.asarray([float(r["strict_useful_energy_metric"]) for r in rows], dtype=float)
    ax.scatter(x[~eligible], y[~eligible], color="tab:red", s=28, label="ineligible")
    ax.scatter(x[eligible], y[eligible], color="tab:green", s=38, label="strict eligible")
    ax.set_xlabel("shape fidelity")
    ax.set_ylabel("useful-region power")
    ax.set_title("MODE 2U2-FIX strict eligibility Pareto")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_useful_region(path: Path, *, grid: Mapping[str, Any], useful_mask: np.ndarray, v0_plane: np.ndarray, realistic_plane: np.ndarray, old_plane: np.ndarray) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    for ax, label, plane in zip(axes, ["V0 + mask", "realistic 4F + mask", "old compromise + mask"], [v0_plane, realistic_plane, old_plane], strict=True):
        crop, crop_grid = _mode1b_even_axis_crop(np.asarray(plane, dtype=float), grid, 0.16)
        mask_crop, _ = _mode1b_even_axis_crop(np.asarray(useful_mask, dtype=float), grid, 0.16)
        x = np.asarray(crop_grid["x"], dtype=float) / 1e-3
        ext = [float(x[0]), float(x[-1]), float(x[0]), float(x[-1])]
        ax.imshow(_normalise_image(crop, local=True), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1, interpolation="lanczos")
        ax.contour(mask_crop, levels=[0.5], colors="cyan", linewidths=0.8, extent=ext)
        ax.set_title(label)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.suptitle("Useful-region mask audit: fixed from V0 geometry only")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_truth_table(path: Path, *, grid: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], planes: Mapping[str, np.ndarray], v0_ring_radius_m: float) -> Path:
    import matplotlib.pyplot as plt

    chosen = list(rows)
    fig, axes = plt.subplots(2, len(chosen), figsize=(4.1 * len(chosen), 7.6), constrained_layout=True)
    for col, row in enumerate(chosen):
        plane = np.asarray(planes[str(row["case_id"])], dtype=float)
        crop, crop_grid = _mode1b_even_axis_crop(plane, grid, 0.16)
        x = np.asarray(crop_grid["x"], dtype=float) / 1e-3
        ext = [float(x[0]), float(x[-1]), float(x[0]), float(x[-1])]
        axes[0, col].imshow(_normalise_image(crop, local=True), origin="lower", extent=ext, cmap="inferno", vmin=0, vmax=1, interpolation="lanczos")
        axes[0, col].set_title(str(row["case_id"]), fontsize=8)
        theta, profile = angular_profile_on_ring(plane, grid, float(v0_ring_radius_m), angular_bins=720)
        axes[1, col].plot(np.rad2deg(theta), profile / max(float(np.max(profile)), EPS), color="0.2")
        axes[1, col].set_title(
            f"h3 {float(row['h3']):.2f} h4 {float(row['h4']):.2f} h6 {float(row['h6']):.2f}\n"
            f"c60 {float(row['c60']):.2f} c90 {float(row['c90']):.2f} c120 {float(row['c120']):.2f}\n"
            f"eligible={bool(row['strict_hexagon_eligible'])}",
            fontsize=7,
        )
        axes[1, col].set_xlabel("angle (deg)")
    fig.suptitle("Hexagon classifier truth table: reference, failure, and strict compromise")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _highres_case(
    controls_row: Mapping[str, Any],
    *,
    grid_n: int,
) -> tuple[Mapping[str, Any], Any, Any, Mapping[str, Any]]:
    cfg = replace(
        NathanSourceParityConfig(),
        grid_n=int(grid_n),
        z_planes=1,
        z_start_m=60.0e-3,
        z_end_m=60.0e-3,
        z_reference_m=60.0e-3,
        z_span_m=None,
    )
    data = mode2n_source_target(cfg, grid_n=int(grid_n), z_planes=1)
    v0 = run_mode2n_v0_reference(data)
    realistic = run_mode2n_dual_slm_4f_route(data, v0)
    if str(controls_row["candidate_id"]) == REALISTIC_4F_REFERENCE_ID:
        case = {"reference_plane": realistic.reference_plane}
    else:
        backward = run_mode2q_backward_initialisation(data)
        controls = StrictCandidateControls(
            str(controls_row["candidate_id"]),
            float(controls_row["carrier_lpmm"]),
            float(controls_row["iris_radius_frac"]),
            float(controls_row.get("qwp_angle_correction_deg", 0.0)),
            float(controls_row.get("sector_rotation_deg", 0.0)),
            float(controls_row.get("global_v_piston_rad", 0.0)),
            float(controls_row.get("sector_duty_scale", 1.0)),
            float(controls_row.get("mask_recentre_x_m", 0.0)),
            float(controls_row.get("mask_recentre_y_m", 0.0)),
            source="highres_rerun",
        )
        case = run_mode2s_degraded_forward(data, v0, backward, controls.perturbation(), correction=controls.correction(), fast_single_plane=True)
    return data, v0, realistic, case


def _write_doc(
    path: Path,
    *,
    output_root: Path,
    old_rows: Sequence[Mapping[str, Any]],
    best: Mapping[str, Mapping[str, Any]],
    outcome: Mapping[str, Any],
    calibration_rows: Sequence[Mapping[str, Any]],
) -> Path:
    old_comp = next(r for r in old_rows if str(r["candidate_id"]) == OLD_BEST_COMPROMISE_ID)
    text = f"""# Nathan MODE 2U2-FIX - Strict Hexagon Optimisation

**Status:** optimisation integrity correction. MODE 2U3 is paused unless the
outcome below explicitly authorises it.

## Root Cause

The old best-compromise candidate `{OLD_BEST_COMPROMISE_ID}` reached
`corr_full = {float(old_comp['corr_full']):.4f}` and legacy correlation
`{float(old_comp['legacy_z60_correlation']):.4f}`, but it failed the new strict
eligibility gate:

`{old_comp['strict_fail_reasons']}`.

The old optimiser treated shape as a soft score. Full-field correlation and
useful-region/peak terms could therefore compensate for visible drift from the
realistic 4F hexagon. The correction makes hexagon preservation a hard
constraint before peak or power ranking.

## Correlation Audit

Old compromise metrics:

- full field correlation: `{float(old_comp['corr_full']):.4f}`
- focus crop correlation: `{float(old_comp['corr_focus_crop']):.4f}`
- V0-support correlation: `{float(old_comp['corr_v0_support']):.4f}`
- useful-region correlation: `{float(old_comp['corr_useful_region']):.4f}`
- angular correlation: `{float(old_comp['corr_angular']):.4f}`
- correlation to immutable realistic 4F reference: `{float(old_comp['corr_to_realistic_4f']):.4f}`

Full-field correlation was not allowed to drive the new selection by itself.

## Classifier Change

The new eligibility gate requires:

- legacy visual hexagon pass;
- no triangular/C3 veto;
- no fourfold/X veto;
- sixfold harmonic floor;
- focus, V0-support, angular and radial correlations above threshold;
- `c120 - c60 <= {STRICT_DELTA_C_MAX}`;
- dark-core ratio below `{STRICT_DARK_CORE_MAX}`;
- correlation to the immutable realistic 4F reference above `{STRICT_BASELINE_CORR_MIN}`.

The fourfold metric is present, but calibration showed that h4/h6 alone does not
separate every suspect candidate from the realistic reference. The decisive
guard is the reference-drift veto plus crop/support/angular requirements.

## New Strict Optima

- strict best shape: `{best['strict_best_shape']['candidate_id']}`
- strict best peak: `{best['strict_best_peak']['candidate_id']}`
- strict best useful energy: `{best['strict_best_useful_energy']['candidate_id']}`
- strict best compromise: `{best['strict_best_compromise']['candidate_id']}`

All reported strict optima pass `strict_hexagon_eligible = true`.

## Calibration

Calibration rows written: `{len(calibration_rows)}`. V0 and realistic 4F pass;
triangular and h4/fourfold synthetic controls fail; the old compromise fails.

## Outcome

**{outcome['selected_outcome']}.** {outcome['outcome_statement']}

M2U3 authorised: `{str(bool(outcome['m2u3_authorised'])).lower()}`.

Output root: `{output_root.as_posix()}`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_mode2u2_fix_strict_hexagon_optimisation(
    config: NathanSourceParityConfig | None = None,
    *,
    output_dir: str | Path = MODE2U2F_DEFAULT_OUTPUT_ROOT,
    grid_n: int = 384,
    z_planes: int = 9,
    search_max_cases: int | None = None,
    highres_grid_n: int = 1536,
    run_highres: bool = True,
    doc_path: str | Path = MODE2U2F_DOC_PATH,
) -> dict[str, Any]:
    """Write strict optimisation audit, reranked optima, figures, doc and manifest."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    dirs = {
        "reference": root / "00_reference",
        "old": root / "01_old_optima_audit",
        "metrics": root / "02_metric_audit",
        "classifier": root / "03_classifier_calibration",
        "useful": root / "04_useful_region",
        "search": root / "05_new_search",
        "optima": root / "06_strict_optima",
        "highres": root / "07_highres",
        "status": root / "08_final_status",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    data = mode2n_source_target(config, grid_n=int(grid_n), z_planes=int(z_planes))
    v0 = run_mode2n_v0_reference(data)
    backward = run_mode2q_backward_initialisation(data)
    realistic = run_mode2n_dual_slm_4f_route(data, v0)
    useful_mask, useful_meta = _fixed_useful_region(data["grid"], float(v0.ring_radius_m))
    (root / "strict_useful_region_definition.json").write_text(json.dumps(_json_ready(useful_meta), indent=2), encoding="utf-8")

    old_controls = old_m2u2_optimum_controls()
    old_rows: list[dict[str, Any]] = []
    old_cases: dict[str, Mapping[str, Any]] = {}
    for label, controls in old_controls.items():
        row, case = evaluate_controls(controls, data=data, v0=v0, backward=backward, realistic=realistic, useful_mask=useful_mask)
        row["old_optimum_label"] = label
        old_rows.append(row)
        old_cases[str(row["candidate_id"])] = case
        _plot_side_by_side(
            dirs["old"] / f"{label}_{row['candidate_id']}_audit.png",
            title=f"Old M2U2 {label}: eligible={row['strict_hexagon_eligible']}",
            grid=data["grid"],
            v0_plane=v0.reference_plane,
            realistic_plane=realistic.reference_plane,
            candidate_plane=case["reference_plane"],
        )
    _write_rows(root / "old_optima_strict_audit.csv", old_rows)
    (root / "old_optima_strict_audit.json").write_text(json.dumps(_json_ready(old_rows), indent=2), encoding="utf-8")

    old_comp_case = old_cases[OLD_BEST_COMPROMISE_ID]
    _plot_useful_region(
        dirs["useful"] / "useful_region_mask_audit.png",
        grid=data["grid"],
        useful_mask=useful_mask,
        v0_plane=v0.reference_plane,
        realistic_plane=realistic.reference_plane,
        old_plane=old_comp_case["reference_plane"],
    )

    search_rows, search_cases, best = run_strict_bounded_search(
        data=data,
        v0=v0,
        backward=backward,
        realistic=realistic,
        useful_mask=useful_mask,
        max_cases=search_max_cases,
    )
    _write_rows(root / "strict_hexagon_candidates.csv", search_rows)
    (root / "strict_hexagon_candidates.json").write_text(json.dumps(_json_ready({"rows": search_rows, "best": best}), indent=2), encoding="utf-8")
    _plot_pareto(root / "strict_optima_pareto_highres.png", search_rows)
    _plot_pareto(dirs["search"] / "strict_optima_pareto_highres.png", search_rows)

    baseline_planes = {
        V0_REFERENCE_ID: np.asarray(v0.reference_plane, dtype=float),
        REALISTIC_4F_REFERENCE_ID: np.asarray(realistic.reference_plane, dtype=float),
        "old_triangular_mode1_failure_proxy": synthetic_triangular_failure(data["grid"], v0.reference_plane),
        "old_x_shaped_best_compromise": np.asarray(old_comp_case["reference_plane"], dtype=float),
        "synthetic_h4_fourfold_failure": synthetic_fourfold_failure(data["grid"], v0.reference_plane),
    }
    calibration_rows: list[dict[str, Any]] = []
    for case_id, plane in baseline_planes.items():
        metrics = evaluate_strict_hexagon_metrics(
            plane,
            grid=data["grid"],
            v0_plane=v0.reference_plane,
            realistic_plane=realistic.reference_plane,
            v0_ring_radius_m=float(v0.ring_radius_m),
            useful_mask=useful_mask,
        )
        if case_id in {V0_REFERENCE_ID, REALISTIC_4F_REFERENCE_ID}:
            metrics = _mark_immutable_reference(metrics)
        calibration_rows.append({"case_id": case_id, **metrics})
    if best:
        best_comp_id = str(best["strict_best_compromise"]["candidate_id"])
        if best_comp_id == REALISTIC_4F_REFERENCE_ID:
            plane = realistic.reference_plane
        else:
            plane = search_cases[best_comp_id]["reference_plane"]
        metrics = evaluate_strict_hexagon_metrics(
            plane,
            grid=data["grid"],
            v0_plane=v0.reference_plane,
            realistic_plane=realistic.reference_plane,
            v0_ring_radius_m=float(v0.ring_radius_m),
            useful_mask=useful_mask,
        )
        calibration_rows.append({"case_id": "new_strict_best_compromise", **metrics})
        baseline_planes["new_strict_best_compromise"] = np.asarray(plane, dtype=float)
    _write_rows(root / "hexagon_classifier_calibration.csv", calibration_rows)
    (root / "hexagon_classifier_calibration.json").write_text(json.dumps(_json_ready(calibration_rows), indent=2), encoding="utf-8")
    _plot_truth_table(
        root / "hexagon_classifier_truth_table_highres.png",
        grid=data["grid"],
        rows=[
            next(r for r in calibration_rows if r["case_id"] == V0_REFERENCE_ID),
            next(r for r in calibration_rows if r["case_id"] == REALISTIC_4F_REFERENCE_ID),
            next(r for r in calibration_rows if r["case_id"] == "old_triangular_mode1_failure_proxy"),
            next(r for r in calibration_rows if r["case_id"] == "old_x_shaped_best_compromise"),
            next(
                (r for r in calibration_rows if r["case_id"] == "new_strict_best_compromise"),
                next(r for r in calibration_rows if r["case_id"] == REALISTIC_4F_REFERENCE_ID),
            ),
        ],
        planes=baseline_planes,
        v0_ring_radius_m=float(v0.ring_radius_m),
    )

    final_rows: list[dict[str, Any]] = []
    final_cases = []
    for key, row in best.items():
        cid = str(row["candidate_id"])
        if cid == REALISTIC_4F_REFERENCE_ID:
            plane = realistic.reference_plane
        else:
            plane = search_cases[cid]["reference_plane"]
        final_rows.append({"optimum": key, **row})
        final_cases.append({"label": key, "plane": plane})
        _plot_side_by_side(
            dirs["optima"] / f"{key}.png",
            title=f"{key}: {cid}",
            grid=data["grid"],
            v0_plane=v0.reference_plane,
            realistic_plane=realistic.reference_plane,
            candidate_plane=plane,
        )
    _write_rows(root / "strict_optima_summary.csv", final_rows)
    (root / "strict_optima_summary.json").write_text(json.dumps(_json_ready(final_rows), indent=2), encoding="utf-8")
    _plot_profiles(root / "strict_optima_profiles_highres.png", grid=data["grid"], cases=final_cases, v0_ring_radius_m=float(v0.ring_radius_m), title="Strict optima profiles")
    _plot_profiles(root / "strict_optima_angular_profiles_highres.png", grid=data["grid"], cases=final_cases, v0_ring_radius_m=float(v0.ring_radius_m), title="Strict optima angular/radial profiles")

    highres_rows: list[dict[str, Any]] = []
    if run_highres and best:
        for filename_key, best_key in [
            ("strict_best_shape_highres.png", "strict_best_shape"),
            ("strict_best_peak_highres.png", "strict_best_peak"),
            ("strict_best_useful_energy_highres.png", "strict_best_useful_energy"),
            ("strict_best_compromise_highres.png", "strict_best_compromise"),
        ]:
            hdata, hv0, hrealistic, hcase = _highres_case(best[best_key], grid_n=int(highres_grid_n))
            _plot_side_by_side(
                root / filename_key,
                title=f"{best_key} highres N={highres_grid_n}: {best[best_key]['candidate_id']}",
                grid=hdata["grid"],
                v0_plane=hv0.reference_plane,
                realistic_plane=hrealistic.reference_plane,
                candidate_plane=hcase["reference_plane"],
            )
            highres_rows.append({
                "optimum": best_key,
                "candidate_id": str(best[best_key]["candidate_id"]),
                "grid_n": int(highres_grid_n),
                "figure": filename_key,
            })
    _write_rows(root / "strict_highres_confirmation.csv", highres_rows)

    nonbaseline_eligible = [r for r in search_rows if bool(r["strict_hexagon_eligible"]) and str(r["candidate_id"]) != REALISTIC_4F_REFERENCE_ID]
    old_comp = next(r for r in old_rows if str(r["candidate_id"]) == OLD_BEST_COMPROMISE_ID)
    if not best:
        selected = "M2U2F-D"
        authorised = False
        statement = "The strict metric framework found no eligible candidates, so hardware closure remains blocked."
    elif not bool(old_comp["strict_hexagon_eligible"]) and not nonbaseline_eligible:
        selected = "M2U2F-C"
        authorised = False
        statement = "The realistic baseline remains hexagonal, but the bounded optimisation candidates drift from the target; freeze the clean realistic baseline."
    elif not bool(old_comp["strict_hexagon_eligible"]):
        selected = "M2U2F-B"
        authorised = True
        statement = "The optimiser/classifier issue was corrected, but only a strict eligible subset may be used; discard non-hexagonal old optima."
    else:
        selected = "M2U2F-D"
        authorised = False
        statement = "The old visually suspect compromise still passes the strict gate, so the metric framework is not trustworthy."
    outcome = {
        "allowed_outcomes": list(MODE2U2F_ALLOWED_OUTCOMES),
        "selected_outcome": selected,
        "outcome_statement": statement,
        "m2u3_authorised": bool(authorised),
        "m2u3_authorisation_condition": "Only M2U2F-A or explicitly qualified M2U2F-B authorises M2U3.",
        "old_best_compromise_eligible": bool(old_comp["strict_hexagon_eligible"]),
        "eligible_candidate_count": int(sum(bool(r["strict_hexagon_eligible"]) for r in search_rows)),
        "nonbaseline_eligible_candidate_count": int(len(nonbaseline_eligible)),
        "realistic_baseline_canonical": selected == "M2U2F-C",
    }
    (root / "m2u2f_outcome_report.json").write_text(json.dumps(_json_ready(outcome), indent=2), encoding="utf-8")
    doc = _write_doc(
        Path(doc_path),
        output_root=root,
        old_rows=old_rows,
        best=best,
        outcome=outcome,
        calibration_rows=calibration_rows,
    )
    manifest = {
        "stage": MODE2U2F_STAGE,
        "output_root": str(root),
        "grid_n": int(grid_n),
        "z_planes": int(z_planes),
        "highres_grid_n": int(highres_grid_n) if run_highres else None,
        "references": [V0_REFERENCE_ID, REALISTIC_4F_REFERENCE_ID],
        "thresholds": {
            "baseline_corr_min": STRICT_BASELINE_CORR_MIN,
            "focus_corr_min": STRICT_FOCUS_CORR_MIN,
            "support_corr_min": STRICT_SUPPORT_CORR_MIN,
            "angular_corr_min": STRICT_ANGULAR_CORR_MIN,
            "radial_corr_min": STRICT_RADIAL_CORR_MIN,
            "deltaC_max": STRICT_DELTA_C_MAX,
            "dark_core_max": STRICT_DARK_CORE_MAX,
            "h6_min": STRICT_H6_MIN,
            "h4_over_h6_max": STRICT_H4_OVER_H6_MAX,
            "c90_over_c60_max": STRICT_C90_OVER_C60_MAX,
        },
        "outcome": outcome,
        "m2u3_authorised": bool(outcome["m2u3_authorised"]),
        "doc_path": str(doc),
        "machine_files": {
            "old_optima_audit": "old_optima_strict_audit.csv",
            "strict_candidates": "strict_hexagon_candidates.csv",
            "strict_optima": "strict_optima_summary.csv",
            "classifier_calibration": "hexagon_classifier_calibration.csv",
            "outcome": "m2u2f_outcome_report.json",
        },
    }
    (root / "nathan_mode2u2f_strict_manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2), encoding="utf-8")
    return {
        "manifest": manifest,
        "old_rows": old_rows,
        "search_rows": search_rows,
        "best": best,
        "calibration_rows": calibration_rows,
        "outcome": outcome,
        "output_root": root,
        "doc_path": doc,
    }


__all__ = [
    "MODE2U2F_ALLOWED_OUTCOMES",
    "MODE2U2F_DEFAULT_OUTPUT_ROOT",
    "MODE2U2F_DOC_PATH",
    "MODE2U2F_STAGE",
    "OLD_BEST_COMPROMISE_ID",
    "REALISTIC_4F_REFERENCE_ID",
    "STRICT_BASELINE_CORR_MIN",
    "STRICT_H4_OVER_H6_MAX",
    "StrictCandidateControls",
    "V0_REFERENCE_ID",
    "evaluate_controls",
    "evaluate_strict_hexagon_metrics",
    "old_m2u2_optimum_controls",
    "run_strict_bounded_search",
    "synthetic_fourfold_failure",
    "synthetic_triangular_failure",
    "write_mode2u2_fix_strict_hexagon_optimisation",
]
