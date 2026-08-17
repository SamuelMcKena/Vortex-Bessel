"""MODE 2W-FIX: sequential architecture and readable master figure pack.

This is a presentation/source-audit correction mode, not a new physics branch.
It proves that the intended collinear sequential two-SLM implementation is
Jones-equivalent to the already validated abstract H/V synthesis, then rebuilds
the figure pack around the sequential architecture and audited numerical
sources.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.equations.propagation import scalable_angular_spectrum_propagate
from vbb_study.vector_field import propagate_vector_asm
from vbb_study.digital_twin.nathan_mode2u2_fix_strict_hexagon_optimisation import (
    OLD_BEST_COMPROMISE_ID,
    STRICT_BASELINE_CORR_MIN,
    evaluate_strict_hexagon_metrics,
)
from vbb_study.digital_twin.nathan_mode2u3_hardware_closure import (
    CANONICAL_OPERATING_POINT_ID,
    STRICT_COMPROMISE_ID,
    assert_not_forbidden,
)
from vbb_study.digital_twin.nathan_mode2v_lab_ready_build import (
    LAB_WAVELENGTH_M,
    build_native_masks,
    load_operating_points,
)
from vbb_study.digital_twin.nathan_vector_hexagon import (
    EPS,
    MODE2N_DEFAULT_CARRIER_LPMM,
    MODE2N_DEFAULT_IRIS_RADIUS_FRAC,
    Mode2SCorrection,
    Mode2SPerturbation,
    NathanSourceParityConfig,
    _json_ready,
    _normalise_image,
    _safe_corr,
    _apply_free_space_vector_axicon,
    _mode2n_vector_field,
    angular_profile_on_ring,
    apply_uniform_jones,
    circular_profile_correlation,
    complex_vector_overlap,
    jones_metric_row,
    linear_retarder,
    mode2n_source_target,
    mode2s_combined_cases,
    run_mode2n_dual_slm_4f_route,
    run_mode2n_dual_slm_qwp_route,
    run_mode2n_v0_reference,
    run_mode2q_backward_initialisation,
    run_mode2s_degraded_forward,
    wrap_2pi,
)

MODE2WF_STAGE = "nathan_mode2w_fix_sequential_master"
MODE2WF_DEFAULT_OUTPUT_ROOT = Path("outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master")
MODE2WF_DOC_PATH = Path("docs/83_nathan_mode2w_fix_sequential_architecture.md")
MODE2WF_ALLOWED_OUTCOMES = ("M2WF-A", "M2WF-B", "M2WF-C", "M2WF-D")
PUBLICATION_HIGHN_ROOT = Path("outputs/figures/digital_twin/nathan_mode2u_master_highres_audit")


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches

    plt.rcParams.update({
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 8.8,
        "xtick.labelsize": 7.8,
        "ytick.labelsize": 7.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })
    return plt, patches


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _savefig(fig: Any, png: Path, pdf: Path, *, dpi: int = 280) -> tuple[Path, Path]:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    return png, pdf


def _source_config(
    *,
    grid_n: int,
    z_planes: int,
    z_start_m: float,
    z_end_m: float,
) -> NathanSourceParityConfig:
    return replace(
        NathanSourceParityConfig(),
        wavelength_m=LAB_WAVELENGTH_M,
        grid_n=int(grid_n),
        z_planes=int(z_planes),
        z_start_m=float(z_start_m),
        z_end_m=float(z_end_m),
        z_span_m=None,
    )


def _bench_from_config(cfg: NathanSourceParityConfig) -> dict[str, Any]:
    data = mode2n_source_target(cfg, grid_n=int(cfg.grid_n), z_planes=int(cfg.z_planes))
    v0 = run_mode2n_v0_reference(data)
    realistic = run_mode2n_dual_slm_4f_route(data, v0)
    backward = run_mode2q_backward_initialisation(data)
    useful_mask, useful_meta = _fixed_useful_region(data["grid"], float(v0.ring_radius_m))
    return {
        "config": cfg,
        "data": data,
        "v0": v0,
        "realistic": realistic,
        "backward": backward,
        "useful_mask": useful_mask,
        "useful_meta": useful_meta,
    }


def _fixed_useful_region(grid: Mapping[str, Any], ring_radius_m: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Small local duplicate of the M2U2-FIX useful-region geometry."""

    radius = 2.65 * float(ring_radius_m)
    x = np.asarray(grid["X"], dtype=float)
    y = np.asarray(grid["Y"], dtype=float)
    q1 = np.abs(x)
    q2 = np.abs(0.5 * x + np.sqrt(3.0) / 2.0 * y)
    q3 = np.abs(-0.5 * x + np.sqrt(3.0) / 2.0 * y)
    mask = (q1 <= radius) & (q2 <= radius) & (q3 <= radius)
    return mask, {
        "region_id": "fixed_regular_hexagon_radius_2p65_v0_ring",
        "hex_radius_m": radius,
        "v0_ring_radius_m": float(ring_radius_m),
    }


def _crop(arr: np.ndarray, fraction: float = 0.30) -> tuple[np.ndarray, tuple[slice, slice]]:
    a = np.asarray(arr)
    n, m = a.shape[-2:]
    side = max(8, int(round(min(n, m) * float(fraction))))
    if side % 2:
        side += 1
    cy, cx = n // 2, m // 2
    sy = slice(max(0, cy - side // 2), min(n, cy + side // 2))
    sx = slice(max(0, cx - side // 2), min(m, cx + side // 2))
    return np.asarray(a[sy, sx]), (sy, sx)


def _extent_mm(grid: Mapping[str, Any], slices: tuple[slice, slice] | None = None) -> list[float]:
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    if slices is not None:
        sy, sx = slices
        x = x[sx]
        y = y[sy]
    return [float(x[0] / 1e-3), float(x[-1] / 1e-3), float(y[0] / 1e-3), float(y[-1] / 1e-3)]


def _equal_power(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a / max(float(np.sum(a)), EPS)


def _samples_per_radial_fringe(cfg: NathanSourceParityConfig) -> float:
    period = 2.0 * np.pi / max(float(cfg.k_r_m_inv), EPS)
    dx = float(cfg.window_m) / float(cfg.grid_n)
    return float(period / dx)


def _samples_per_radial_fringe_from_dx(cfg: NathanSourceParityConfig, dx_m: float) -> float:
    period = 2.0 * np.pi / max(float(cfg.k_r_m_inv), EPS)
    return float(period / max(float(dx_m), EPS))


def _sas_output_dx(cfg: NathanSourceParityConfig, z_m: float, pad_factor: int) -> float:
    return float(cfg.wavelength_m) * abs(float(z_m)) / max(float(pad_factor) * float(cfg.window_m), EPS)


def _source_row(
    *,
    figure_id: str,
    panel_id: str,
    route: str,
    cfg: NathanSourceParityConfig,
    module: str,
    source: str,
    display_interpolation: bool,
    native_metrics: bool = True,
    higher_n_exists: bool = False,
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "panel_id": panel_id,
        "route_or_case": route,
        "originating_module": module,
        "originating_output_file_or_function": source,
        "numerical_N": int(cfg.grid_n),
        "physical_window_m": float(cfg.window_m),
        "z_planes": int(cfg.z_planes),
        "z_start_mm": float(cfg.z_start_m / 1e-3),
        "z_end_mm": float(cfg.z_end_m / 1e-3),
        "numerical_dx_m": float(cfg.window_m / cfg.grid_n),
        "samples_per_radial_fringe": _samples_per_radial_fringe(cfg),
        "display_interpolation_used": bool(display_interpolation),
        "metrics_computed_on_native_data": bool(native_metrics),
        "higher_N_validated_source_exists": bool(higher_n_exists),
    }


def _sas_zoom_plane(
    Ex: Any,
    Ey: Any,
    bench: Mapping[str, Any],
    *,
    z_m: float,
    pad_factor: int,
) -> dict[str, Any]:
    """Render a propagated plane on the SAS-scaled output grid.

    Metrics in MODE 2W-FIX remain tied to the native fixed-grid ASM arrays.  This helper
    is for publication close-ups: it uses scalable angular-spectrum propagation to return
    a finer physical output grid for the central field, avoiding low-sample full-window
    crops masquerading as high-resolution images.
    """

    cfg: NathanSourceParityConfig = bench["config"]
    data = bench["data"]
    field = _mode2n_vector_field(Ex, Ey, data)
    after, axicon_meta = _apply_free_space_vector_axicon(
        field,
        n_axicon=float(cfg.axicon_n),
        n_medium=float(cfg.medium_n),
        base_angle_rad=float(cfg.axicon_base_angle_rad),
    )
    # Match the vector ASM entry condition, then use SAS per component for the scaled grid.
    projected = propagate_vector_asm(after, 0.0)
    components = []
    metas = []
    out_grid = None
    for comp in (projected.ex, projected.ey, projected.ez):
        out, grid, meta = scalable_angular_spectrum_propagate(
            comp,
            dict(projected.grid),
            float(projected.wavelength_m),
            float(z_m),
            n_medium=float(projected.medium_index),
            pad_factor=int(pad_factor),
            bandlimit=True,
            skip_final_phase=True,
            allow_invalid=False,
        )
        components.append(out)
        metas.append(meta)
        out_grid = grid
    intensity = sum(np.abs(comp) ** 2 for comp in components)
    meta0 = dict(metas[0])
    return {
        "intensity": np.asarray(intensity, dtype=np.float32),
        "grid": out_grid,
        "z_m": float(z_m),
        "pad_factor": int(pad_factor),
        "method": "scalable_angular_spectrum_zoom",
        "input_N": int(projected.ex.shape[0]),
        "input_dx_m": float(projected.grid["dx"]),
        "output_N": int(np.asarray(intensity).shape[0]),
        "output_dx_m": float(out_grid["dx"]),
        "output_window_m": float(out_grid["dx"]) * int(np.asarray(intensity).shape[0]),
        "samples_per_radial_fringe": _samples_per_radial_fringe_from_dx(cfg, float(out_grid["dx"])),
        "sas_valid": bool(meta0["valid"]),
        "sas_z_limit_m": float(meta0["z_limit_m"]),
        "sas_z_over_limit": float(meta0["z_over_limit"]),
        "axicon_meta": axicon_meta,
        "component_meta": metas,
    }


def sequential_architecture_rows() -> list[dict[str, Any]]:
    """Canonical sequential single-beam route, with no PBS split/recombination."""

    return [
        {"order": 1, "component": "PHAROS source", "role": "1029 nm Gaussian, w0=2 mm", "status": "fixed_validated"},
        {"order": 2, "component": "POL1 / input HWP", "role": "prepare equal coherent H/V components", "status": "routine_calibration"},
        {"order": 3, "component": "SLM1", "role": "LC-compatible selective phase: phi_H = +alpha + carrier", "status": "fixed_convention"},
        {"order": 4, "component": "swap HWP", "role": "conditional same-orientation panel route; swap H/V before SLM2", "status": "conditional"},
        {"order": 5, "component": "SLM2", "role": "selective phase: phi_V = -alpha + pi/2 + carrier", "status": "fixed_convention"},
        {"order": 6, "component": "swap-back HWP", "role": "conditional same-orientation panel route; restore H/V order", "status": "conditional"},
        {"order": 7, "component": "common 4F", "role": "single first-order filter, f=300 mm, carrier=6.25 lp/mm, iris D~1.54 mm", "status": "routine_calibration"},
        {"order": 8, "component": "QWP", "role": "nominal code -45 deg; mount sign bench-checked", "status": "routine_calibration"},
        {"order": 9, "component": "axicon", "role": "2 deg base, n=1.458", "status": "fixed_validated"},
        {"order": 10, "component": "camera z stage", "role": "source-scale z scan, reference z=60 mm", "status": "routine_calibration"},
    ]


def sequential_variant_rows() -> list[dict[str, Any]]:
    return [
        {
            "variant": "A_same_panel_orientation",
            "swap_hwps_required": True,
            "rotated_panel_valid": False,
            "description": "SLM1 and SLM2 share LC-director orientation; use swap and swap-back HWPs so each original component is selectively modulated.",
            "status": "valid_pending_director_orientation_check",
        },
        {
            "variant": "B_orthogonally_mounted_slm2",
            "swap_hwps_required": False,
            "rotated_panel_valid": True,
            "description": "Mount SLM2 with orthogonal LC director so the second original component is addressed without swap HWPs.",
            "status": "valid_if_mechanical_mount_and_panel_director_test_confirm",
        },
    ]


def sequential_phase_convention_rows() -> list[dict[str, Any]]:
    return [
        {"element": "SLM1", "phase_convention": "phi_H = +alpha + carrier", "component_modulated": "first/original H component"},
        {"element": "SLM2", "phase_convention": "phi_V = -alpha + pi/2 + carrier", "component_modulated": "second/original V component"},
    ]


def sequential_jones_equivalence(bench: Mapping[str, Any]) -> dict[str, Any]:
    """Numerically verify the sequential route against the validated abstract route."""

    data = bench["data"]
    a = np.asarray(data["A"], dtype=float)
    alpha = np.asarray(data["alpha"], dtype=float)
    mask = np.asarray(data["metric_mask"], dtype=bool)
    phi_h = alpha
    phi_v = -alpha + 0.5 * np.pi
    e0 = (a / np.sqrt(2.0), a / np.sqrt(2.0))
    e1 = ((a / np.sqrt(2.0)) * np.exp(1j * phi_h), e0[1])
    e2 = (e1[1], e1[0])
    e3 = ((a / np.sqrt(2.0)) * np.exp(1j * phi_v), e2[1])
    e4 = (e3[1], e3[0])
    abstract_pre_qwp = (
        (a / np.sqrt(2.0)) * np.exp(1j * phi_h),
        (a / np.sqrt(2.0)) * np.exp(1j * phi_v),
    )
    qwp = linear_retarder(0.5 * np.pi, -0.25 * np.pi)
    post = apply_uniform_jones(qwp, e4[0], e4[1])
    target = data["target"]
    ideal = run_mode2n_dual_slm_qwp_route(data, bench["v0"])
    metrics = evaluate_strict_hexagon_metrics(
        ideal.reference_plane,
        grid=data["grid"],
        v0_plane=bench["v0"].reference_plane,
        realistic_plane=bench["realistic"].reference_plane,
        v0_ring_radius_m=float(bench["v0"].ring_radius_m),
        useful_mask=bench["useful_mask"],
    )
    real_metrics = evaluate_strict_hexagon_metrics(
        bench["realistic"].reference_plane,
        grid=data["grid"],
        v0_plane=bench["v0"].reference_plane,
        realistic_plane=bench["realistic"].reference_plane,
        v0_ring_radius_m=float(bench["v0"].ring_radius_m),
        useful_mask=bench["useful_mask"],
    )
    return {
        "stage": MODE2WF_STAGE,
        "input_jones": "E0 = A/sqrt(2) [1, 1]^T",
        "slm1_phase": "phi_H = +alpha + carrier",
        "slm2_phase": "phi_V = -alpha + pi/2 + carrier",
        "carrier_note": "carrier is common and removed by the validated 4F first-order selection for propagation equivalence",
        "sequential_pre_qwp_overlap_to_abstract": complex_vector_overlap(e4, abstract_pre_qwp, mask),
        "sequential_post_qwp_overlap_to_target": complex_vector_overlap(post, target, mask),
        "sequential_post_qwp_jones_metrics": jones_metric_row("sequential_post_qwp", post, target, mask=mask),
        "ideal_sequential_z60_corr_to_v0": float(ideal.v0_comparison["z60_full_field_correlation"]),
        "ideal_sequential_strict_class": str(metrics["classifier_label"]),
        "ideal_sequential_strict_hexagon_candidate_gate": bool(metrics["strict_hexagon_eligible"]),
        "realistic_sequential_z60_corr_to_v0": float(bench["realistic"].v0_comparison["z60_full_field_correlation"]),
        "realistic_sequential_corr_to_realistic": float(real_metrics["corr_to_realistic_4f"]),
        "realistic_sequential_strict_class": str(real_metrics["classifier_label"]),
        "realistic_sequential_strict_hexagon": bool(real_metrics["strict_hexagon_eligible"]),
        "swap_hwps_conditional": True,
        "rotated_panel_route_valid_if_director_test_confirms": True,
    }


def sequential_power_ledger(bench: Mapping[str, Any], canonical: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Sequential single-beam power ledger without split-arm bookkeeping."""

    fill = 0.93
    effective_modulated = fill * fill
    eff = float(bench["realistic"].slm_4f_report["first_order_efficiency"])
    axicon_t = 0.932404  # same uncoated source axicon transmission measured by M2V ledger to within plotting precision
    after_4f = effective_modulated * eff
    after_axicon = after_4f * axicon_t
    useful_frac = float(canonical["P_useful_over_P_total"])
    stages = [
        ("01_laser_input", 1.0, "input reference"),
        ("02_after_pol_hwp_prep", 1.0, "lossless model; vendor polariser/HWP transmission not invented"),
        ("03_after_slm1", 1.0, "phase-only selective modulation; panel reflectivity unknown and not applied"),
        ("04_after_first_swap_if_used", 1.0, "conditional HWP is unitary in the model"),
        ("05_after_slm2_effective_modulated_field", effective_modulated, "effective validated fill-factor modulated fraction, 0.93^2"),
        ("06_after_swap_back_if_used", effective_modulated, "conditional HWP is unitary in the model"),
        ("07_selected_plus1_after_common_4f", after_4f, f"first-order efficiency {eff:.4f}; rejects carrier tails / DC"),
        ("08_after_qwp", after_4f, "QWP unitary"),
        ("09_after_axicon", after_axicon, "uncoated source axicon Fresnel/conical p-s model"),
        ("10_total_power_at_z60", after_axicon, "free-space propagation conserves integrated model power"),
        ("11_useful_hexagon_region_power", after_axicon * useful_frac, f"fixed useful-region fraction {useful_frac:.4f}"),
        ("12_outside_useful_region_power", after_axicon * (1.0 - useful_frac), "outer rings / side lobes"),
    ]
    rows = []
    for order, (stage, frac, note) in enumerate(stages, start=1):
        rows.append({
            "order": order,
            "stage": stage,
            "model_fraction_of_input": float(frac),
            "example_at_1W_input_W": float(frac),
            "example_at_10W_input_W": 10.0 * float(frac),
            "note": note,
            "split_arm_stage": False,
            "scaling_disclaimer": "linear scaling example only; not a damage-threshold or power-rating claim",
        })
    return rows


def _route_metrics(route_id: str, plane: np.ndarray, bench: Mapping[str, Any], *, role: str, first_order_efficiency: float | None = None) -> dict[str, Any]:
    metrics = evaluate_strict_hexagon_metrics(
        plane,
        grid=bench["data"]["grid"],
        v0_plane=bench["v0"].reference_plane,
        realistic_plane=bench["realistic"].reference_plane,
        v0_ring_radius_m=float(bench["v0"].ring_radius_m),
        useful_mask=bench["useful_mask"],
    )
    return {
        "route_id": route_id,
        "role": role,
        "classifier_label": metrics["classifier_label"],
        "strict_hexagon_eligible": bool(metrics["strict_hexagon_eligible"]),
        "corr_full": float(metrics["corr_full"]),
        "corr_to_realistic_4f": float(metrics["corr_to_realistic_4f"]),
        "corr_angular": float(metrics["corr_angular"]),
        "c60": float(metrics["c60"]),
        "c90": float(metrics["c90"]),
        "c120": float(metrics["c120"]),
        "h4": float(metrics["h4"]),
        "h6": float(metrics["h6"]),
        "dark_core_ratio": float(metrics["dark_core_ratio"]),
        "P_useful": float(metrics["P_useful"]),
        "P_useful_over_P_total": float(metrics["P_useful_over_P_total"]),
        "strict_peak_metric": float(metrics.get("strict_peak_metric", metrics.get("local_3x3_peak_mean", np.nan))),
        "first_order_efficiency": first_order_efficiency,
        "strict_fail_reasons": metrics.get("strict_fail_reasons", ""),
    }


def _ideal_cases(bench: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ideal = run_mode2n_dual_slm_qwp_route(bench["data"], bench["v0"])
    eff = float(bench["realistic"].slm_4f_report["first_order_efficiency"])
    cases = [
        {"route_id": "V0_reference", "role": "source-scale reference", "plane": np.asarray(bench["v0"].reference_plane, dtype=float), "ref": np.asarray(bench["v0"].reference_plane, dtype=float), "eta": None, "route_result": bench["v0"]},
        {"route_id": "ideal_sequential_dual_slm", "role": "Jones-equivalent sequential ideal", "plane": np.asarray(ideal.reference_plane, dtype=float), "ref": np.asarray(bench["v0"].reference_plane, dtype=float), "eta": None, "route_result": ideal},
        {"route_id": "realistic_sequential_dual_slm_4f", "role": "validated realistic sequential equivalent", "plane": np.asarray(bench["realistic"].reference_plane, dtype=float), "ref": np.asarray(bench["v0"].reference_plane, dtype=float), "eta": eff, "route_result": bench["realistic"]},
    ]
    metrics = [_route_metrics(c["route_id"], c["plane"], bench, role=c["role"], first_order_efficiency=c["eta"]) for c in cases]
    return cases, metrics


def _realism_cases(bench: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = bench["data"]
    v0 = bench["v0"]
    backward = bench["backward"]
    moderate = run_mode2s_degraded_forward(data, v0, backward, mode2s_combined_cases()[1], fast_single_plane=True)
    bad = run_mode2s_degraded_forward(data, v0, backward, mode2s_combined_cases()[2], fast_single_plane=True)
    clean = {
        "reference_plane": np.asarray(bench["realistic"].reference_plane, dtype=float),
        "iris": bench["realistic"].slm_4f_report,
    }
    cases = [
        {"route_id": "clean_realistic", "label": "clean realistic", "plane": np.asarray(clean["reference_plane"], dtype=float), "eta": float(bench["realistic"].slm_4f_report["first_order_efficiency"])},
        {"route_id": "moderate_realism", "label": "moderate", "plane": np.asarray(moderate["reference_plane"], dtype=float), "eta": float(moderate["iris"]["first_order_efficiency"])},
        {"route_id": "bad_realism", "label": "bad", "plane": np.asarray(bad["reference_plane"], dtype=float), "eta": float(bad["iris"]["first_order_efficiency"])},
    ]
    metrics = [_route_metrics(c["route_id"], c["plane"], bench, role=c["label"], first_order_efficiency=c["eta"]) for c in cases]
    return cases, metrics


def _correction_cases(bench: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = bench["data"]
    v0 = bench["v0"]
    backward = bench["backward"]
    degraded_pert = Mode2SPerturbation(
        label="axicon_decentre_0p5mm",
        slm_aperture_clip=True,
        phase_levels=256,
        fill_factor=0.93,
        axicon_decentre_x_m=0.5e-3,
    )
    corrected = Mode2SCorrection(mask_recentre_x_m=0.5e-3)
    degraded = run_mode2s_degraded_forward(data, v0, backward, degraded_pert, fast_single_plane=True)
    recovered = run_mode2s_degraded_forward(data, v0, backward, degraded_pert, correction=corrected, fast_single_plane=True)
    cases = [
        {"route_id": "target_realistic", "label": "target", "plane": np.asarray(bench["realistic"].reference_plane, dtype=float), "eta": float(bench["realistic"].slm_4f_report["first_order_efficiency"])},
        {"route_id": "degraded_axicon_0p5mm", "label": "degraded", "plane": np.asarray(degraded["reference_plane"], dtype=float), "eta": float(degraded["iris"]["first_order_efficiency"])},
        {"route_id": "corrected_axicon_0p5mm", "label": "corrected", "plane": np.asarray(recovered["reference_plane"], dtype=float), "eta": float(recovered["iris"]["first_order_efficiency"])},
    ]
    metrics = [_route_metrics(c["route_id"], c["plane"], bench, role=c["label"], first_order_efficiency=c["eta"]) for c in cases]
    return cases, metrics, {"correction": corrected.as_row(), "truth": {"axicon_decentre_x_um": 500.0}}


def tolerance_limit_rows() -> list[dict[str, Any]]:
    rows = _read_csv(Path("outputs/figures/digital_twin/nathan_mode2s_lab_realism_tolerance/mode2s_single_parameter_tolerances.csv"))
    wanted = [
        ("phase_levels", "phase quantisation", "levels"),
        ("fill_factor", "fill factor", "fraction"),
        ("hv_amplitude_ratio", "H/V imbalance", "ratio"),
        ("qwp_angle_error_deg", "QWP angle", "deg"),
        ("qwp_retardance_error_deg", "QWP retardance", "deg"),
        ("iris_radius_frac", "iris radius", "fraction of carrier"),
        ("iris_decentre_fx_lpmm", "iris decentre", "lp/mm"),
        ("hv_shift_um", "H/V registration", "um"),
        ("axicon_decentre_mm", "axicon decentre", "mm"),
        ("z_offset_mm", "z offset", "mm"),
    ]
    by_param: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_param.setdefault(row.get("sweep_parameter", ""), []).append(row)
    out: list[dict[str, Any]] = []
    for key, label, units in wanted:
        cases = by_param.get(key, [])
        if key == "fill_factor" and not cases:
            out.append({
                "parameter": key,
                "label": label,
                "tested_range": "0.93 fixed validated hardware-realism setting",
                "worst_tested_passing_value": "0.93",
                "first_failing_value": "not reached",
                "minimum_correlation": "",
                "strict_class_status": "visual_hexagonal_field at fixed setting",
                "units": units,
            })
            continue
        values = [float(r["sweep_value"]) for r in cases if r.get("sweep_value", "nan").lower() != "inf"]
        passing = [r for r in cases if _as_bool(r.get("passes"))]
        failing = [r for r in cases if not _as_bool(r.get("passes"))]
        pass_values = [float(r["sweep_value"]) for r in passing if r.get("sweep_value", "nan").lower() != "inf"]
        corr_values = [float(r["z60_full_field_correlation"]) for r in cases if r.get("z60_full_field_correlation")]
        first_fail = failing[0]["sweep_value"] if failing else "not reached"
        strict_classes = sorted({r.get("strict_class", "") for r in cases if r.get("strict_class", "")})
        out.append({
            "parameter": key,
            "label": label,
            "tested_range": "" if not values else f"{min(values):g} to {max(values):g}",
            "worst_tested_passing_value": "" if not pass_values else f"{max(pass_values, key=lambda v: abs(v)):g}",
            "first_failing_value": first_fail,
            "minimum_correlation": "" if not corr_values else float(min(corr_values)),
            "strict_class_status": ", ".join(strict_classes),
            "units": units,
        })
    return out


def combined_correction_summary_rows() -> list[dict[str, Any]]:
    combined = _read_csv(Path("outputs/figures/digital_twin/nathan_mode2s_lab_realism_tolerance/mode2s_combined_cases.csv"))
    loops = _read_csv(Path("outputs/figures/digital_twin/nathan_mode2v_lab_ready_build/08_closed_loop/mode2v_closed_loop_results.csv"))
    rows: list[dict[str, Any]] = []
    for row in combined:
        if row.get("label") in {"combined_mild_lab", "combined_moderate_lab", "combined_bad_lab"}:
            rows.append({
                "case_id": row["label"],
                "case_group": "combined_realism",
                "before_corr_to_reference": float(row["z60_full_field_correlation"]),
                "after_corr_to_reference": "",
                "before_strict_eligible": _as_bool(row.get("passes")),
                "after_strict_eligible": "",
                "strict_class": row.get("strict_class", ""),
                "note": row.get("failure_mode", ""),
            })
    for row in loops:
        if row.get("case_id", "").startswith(("A_", "D_", "E_")):
            rows.append({
                "case_id": row["case_id"],
                "case_group": "closed_loop_recovery",
                "before_corr_to_reference": float(row["initial_corr_to_realistic"]),
                "after_corr_to_reference": float(row["final_corr_to_realistic"]),
                "before_strict_eligible": _as_bool(row["initial_strict_eligible"]),
                "after_strict_eligible": _as_bool(row["final_strict_eligible"]),
                "strict_class": row.get("final_classifier", ""),
                "note": f"{row.get('n_forward_evaluations', '')} measured-image evaluations; search_received_truth=false",
            })
    return rows


def _plot_sequential_architecture(path_png: Path, path_pdf: Path) -> tuple[Path, Path]:
    plt, patches = _mpl()
    fig, ax = plt.subplots(figsize=(17.5, 5.8), constrained_layout=True)
    ax.set_xlim(0, 18)
    ax.set_ylim(-1.8, 1.8)
    ax.axis("off")
    blocks = [
        (0.3, "PHAROS\n1029 nm\nGaussian\nw0=2 mm", "#dbeafe"),
        (2.0, "POL1 / HWP\ncoherent H/V\n50/50 prep", "#fff1c7"),
        (3.9, "SLM1\n1920x1080\nphi_H=+alpha\n+ carrier", "#d8f0df"),
        (5.9, "swap HWP\nif same panel\norientation", "#fff1c7"),
        (7.7, "SLM2\n1920x1080\nphi_V=-alpha+pi/2\n+ carrier", "#d8f0df"),
        (9.9, "swap-back HWP\nif required", "#fff1c7"),
        (11.6, "common 4F\nf=300 mm\n+1=1.929 mm\niris D~1.54 mm", "#e8e8ff"),
        (14.0, "QWP\ncode -45 deg", "#fff1c7"),
        (15.3, "axicon\n2 deg\nn=1.458", "#d8f0df"),
        (16.7, "hexagonal\nBessel zone\ncamera z stage", "#eeeeee"),
    ]
    for x, label, color in blocks:
        rect = patches.FancyBboxPatch((x, -0.55), 1.35, 1.1, boxstyle="round,pad=0.04,rounding_size=0.08",
                                      facecolor=color, edgecolor="#333333", linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x + 0.675, 0, label, ha="center", va="center", fontsize=8.1)
    for idx in range(len(blocks) - 1):
        x0 = blocks[idx][0] + 1.35
        x1 = blocks[idx + 1][0]
        ax.annotate("", xy=(x1, 0), xytext=(x0, 0), arrowprops={"arrowstyle": "->", "lw": 1.3, "color": "#222222"})
    ax.text(0.3, 1.35, "MODE 2W-FIX Figure 1: sequential single-beam architecture (no PBS split, no H/V interferometer arms)",
            fontsize=13, weight="bold", ha="left")
    ax.text(6.2, -1.45, "Alternate valid branch: if SLM2 is mounted with orthogonal LC director, swap/swap-back HWPs may be omitted after the panel-orientation test.",
            fontsize=9.5, ha="center", bbox={"facecolor": "#f8f8f8", "edgecolor": "#777777", "pad": 5})
    return _savefig(fig, path_png, path_pdf)


def _plot_target_masks(path_png: Path, path_pdf: Path, bench: Mapping[str, Any], masks: Mapping[str, Any]) -> tuple[Path, Path]:
    plt, _ = _mpl()
    data = bench["data"]
    grid = data["grid"]
    ex, ey = data["target"]
    ex = np.asarray(ex)
    ey = np.asarray(ey)
    s0 = np.abs(ex) ** 2 + np.abs(ey) ** 2
    s1 = np.abs(ex) ** 2 - np.abs(ey) ** 2
    s2 = 2.0 * np.real(ex * np.conj(ey))
    s3 = -2.0 * np.imag(ex * np.conj(ey))
    fig = plt.figure(figsize=(17.5, 14.2), constrained_layout=False)
    gs = fig.add_gridspec(
        4, 6,
        height_ratios=[1.0, 1.05, 0.78, 0.78],
        hspace=0.52,
        wspace=0.58,
        left=0.045,
        right=0.965,
        top=0.925,
        bottom=0.06,
    )
    top = [("sector mask", data["radial_mask"], "viridis"), ("alpha(theta)", data["alpha"], "twilight"), ("orientation map", np.mod(data["alpha"], np.pi), "twilight")]
    for i, (title, arr, cmap) in enumerate(top):
        ax = fig.add_subplot(gs[0, i * 2:(i + 1) * 2])
        im = ax.imshow(arr, origin="lower", extent=_extent_mm(grid), cmap=cmap, interpolation="bilinear")
        ax.set_title(title)
        ax.set_xlabel("x mm")
        ax.set_ylabel("y mm")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    for i, (key, title) in enumerate([("phi_H", "SLM1 native wrapped phase"), ("phi_V", "SLM2 native wrapped phase")]):
        ax = fig.add_subplot(gs[1, i * 3:(i + 1) * 3])
        im = ax.imshow(masks[key], origin="lower", extent=[-7.68, 7.68, -4.32, 4.32], cmap="twilight",
                       vmin=0, vmax=2 * np.pi, interpolation="nearest", aspect="auto")
        ax.plot([0], [0], marker="+", color="white", ms=10, mew=1.5)
        ax.set_title(title)
        ax.set_xlabel("panel x mm")
        ax.set_ylabel("panel y mm")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="phase rad")
    bottom = [("S0", s0, "inferno"), ("S1", s1, "coolwarm"), ("S2", s2, "coolwarm"), ("S3", s3, "coolwarm"), ("|Ex|", np.abs(ex), "magma"), ("|Ey|", np.abs(ey), "magma")]
    for i, (title, arr, cmap) in enumerate(bottom):
        ax = fig.add_subplot(gs[2 + i // 3, 2 * (i % 3):2 * (i % 3) + 2])
        im = ax.imshow(arr, origin="lower", extent=_extent_mm(grid), cmap=cmap, interpolation="bilinear")
        ax.set_title(title)
        ax.set_xlabel("x mm")
        ax.set_ylabel("y mm")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    meta = masks["metadata"]
    fig.text(
        0.5, 0.495,
        (
            f"Native masks: {meta['panel']['width_px']} x {meta['panel']['height_px']} @ "
            f"{meta['panel']['pixel_pitch_m'] / 1e-6:.0f} um, centre pixel {tuple(meta['panel']['centre_pixel_col_row'])}; "
            f"carrier period {meta['carrier_period_slm_pixels']:.1f} px; phi_H=+alpha+carrier; "
            "phi_V=-alpha+pi/2+carrier; LUT not applied."
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        bbox={"facecolor": "#f7f7f7", "edgecolor": "#777777", "pad": 4},
    )
    fig.suptitle("MODE 2W-FIX Figure 2: target vector field and native sequential SLM1/SLM2 masks; LUT not applied", fontsize=14, weight="bold", y=0.982)
    return _savefig(fig, path_png, path_pdf)


def _plot_three_route_comparison(path_png: Path, path_pdf: Path, bench: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    plt, _ = _mpl()
    z_ref = float(bench["config"].z_reference_m)
    zooms = [
        _sas_zoom_plane(
            case["route_result"].pre_axicon_field[0],
            case["route_result"].pre_axicon_field[1],
            bench,
            z_m=z_ref,
            pad_factor=2,
        )
        for case in cases
    ]
    grid = zooms[0]["grid"]
    ring = float(bench["v0"].ring_radius_m)
    fig = plt.figure(figsize=(15.8, 13.5), constrained_layout=True)
    gs = fig.add_gridspec(5, 3, height_ratios=[1.25, 1.25, 1.25, 0.8, 0.8])
    crops = [_crop(z["intensity"], 0.56) for z in zooms]
    vmax = max(float(np.percentile(crop, 99.8)) for crop, _ in crops)
    for col, case in enumerate(cases):
        plane = np.asarray(zooms[col]["intensity"], dtype=float)
        zgrid = zooms[col]["grid"]
        crop, sl = crops[col]
        ref_crop = np.asarray(zooms[0]["intensity"], dtype=float)[sl]
        row = next(r for r in metrics if r["route_id"] == case["route_id"])
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(_normalise_image(crop, local=False, vmax=vmax), origin="lower", extent=_extent_mm(zgrid, sl),
                  cmap="inferno", vmin=0, vmax=1, interpolation="lanczos", resample=True)
        ax.set_title(f"{case['route_id']}\nlinear focus crop")
        ax.set_xlabel("x mm")
        ax.set_ylabel("y mm")
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(np.log10(_normalise_image(crop, local=True) + 1e-4), origin="lower", extent=_extent_mm(zgrid, sl),
                  cmap="magma", vmin=-4, vmax=0, interpolation="lanczos", resample=True)
        ax.set_title("log focus crop")
        ax.set_xlabel("x mm")
        ax.set_ylabel("y mm")
        ax = fig.add_subplot(gs[2, col])
        diff = _equal_power(crop) - _equal_power(ref_crop)
        lim = max(float(np.max(np.abs(diff))), EPS)
        ax.imshow(diff, origin="lower", extent=_extent_mm(zgrid, sl), cmap="coolwarm", vmin=-lim, vmax=lim, interpolation="lanczos", resample=True)
        ax.set_title("equal-power difference to V0")
        ax.set_xlabel("x mm")
        ax.set_ylabel("y mm")
        ax = fig.add_subplot(gs[3, col])
        px = crop[crop.shape[0] // 2, :]
        py = crop[:, crop.shape[1] // 2]
        xx = np.linspace(_extent_mm(zgrid, sl)[0], _extent_mm(zgrid, sl)[1], px.size)
        ax.plot(xx, px / max(float(np.max(px)), EPS), label="x")
        ax.plot(xx, py / max(float(np.max(py)), EPS), label="y", ls="--")
        ax.set_title("centre x/y profiles")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
        ax = fig.add_subplot(gs[4, col])
        angles, prof = angular_profile_on_ring(plane, zgrid, ring)
        ax.plot(np.rad2deg(angles), prof / max(float(np.max(prof)), EPS), lw=1.0)
        ax.set_title(
            f"SAS dx={float(zooms[col]['output_dx_m']) / 1e-6:.2f} um | native N={int(bench['config'].grid_n)} | corrV0={float(row['corr_full']):.4f}\n"
            f"c60/c90/c120={float(row['c60']):.3f}/{float(row['c90']):.3f}/{float(row['c120']):.3f}; "
            f"h4/h6={float(row['h4']):.3f}/{float(row['h6']):.3f}; dark={float(row['dark_core_ratio']):.4f}",
            fontsize=7.3,
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("angle deg")
    fig.suptitle("MODE 2W-FIX Figure 3A: V0 vs ideal sequential vs realistic sequential + 4F (SAS-scaled focus crops)", fontsize=14, weight="bold")
    return _savefig(fig, path_png, path_pdf)


def _plot_realism(path_png: Path, path_pdf: Path, bench: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    plt, _ = _mpl()
    grid = bench["data"]["grid"]
    fig = plt.figure(figsize=(15.0, 9.3), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    crops = [_crop(c["plane"], 0.30) for c in cases]
    clean_crop = crops[0][0]
    vmax = max(float(np.percentile(crop, 99.8)) for crop, _ in crops)
    for col, case in enumerate(cases):
        crop, sl = crops[col]
        row = next(r for r in metrics if r["route_id"] == case["route_id"])
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(_normalise_image(crop, local=False, vmax=vmax), origin="lower", extent=_extent_mm(grid, sl), cmap="inferno", interpolation="bilinear")
        ax.set_title(f"{case['label']} linear\nclass={row['classifier_label']} corr={float(row['corr_full']):.4f}")
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(np.log10(_normalise_image(crop, local=True) + 1e-4), origin="lower", extent=_extent_mm(grid, sl), cmap="magma", vmin=-4, vmax=0, interpolation="bilinear")
        ax.set_title("log crop")
        ax = fig.add_subplot(gs[2, col])
        diff = _equal_power(crop) - _equal_power(clean_crop)
        lim = max(float(np.max(np.abs(diff))), EPS)
        ax.imshow(diff, origin="lower", extent=_extent_mm(grid, sl), cmap="coolwarm", vmin=-lim, vmax=lim, interpolation="bilinear")
        ax.set_title(f"diff to clean | strict={row['strict_hexagon_eligible']}")
    fig.suptitle("MODE 2W-FIX Figure 3B: clean vs moderate vs bad realism, identical crop/normalisation", fontsize=14, weight="bold")
    return _savefig(fig, path_png, path_pdf)


def _plot_correction(path_png: Path, path_pdf: Path, bench: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], metrics: Sequence[Mapping[str, Any]], correction_meta: Mapping[str, Any]) -> tuple[Path, Path]:
    plt, _ = _mpl()
    grid = bench["data"]["grid"]
    fig = plt.figure(figsize=(16.0, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.32])
    crops = [_crop(c["plane"], 0.30) for c in cases]
    target_crop = crops[0][0]
    vmax = max(float(np.percentile(crop, 99.8)) for crop, _ in crops)
    for col, case in enumerate(cases):
        crop, sl = crops[col]
        row = next(r for r in metrics if r["route_id"] == case["route_id"])
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(_normalise_image(crop, local=False, vmax=vmax), origin="lower", extent=_extent_mm(grid, sl), cmap="inferno", interpolation="bilinear")
        ax.set_title(f"{case['label']}\ncorrR={float(row['corr_to_realistic_4f']):.4f}, strict={row['strict_hexagon_eligible']}")
    diff = _equal_power(crops[-1][0]) - _equal_power(target_crop)
    lim = max(float(np.max(np.abs(diff))), EPS)
    ax = fig.add_subplot(gs[0, 3])
    ax.imshow(diff, origin="lower", extent=_extent_mm(grid, crops[-1][1]), cmap="coolwarm", vmin=-lim, vmax=lim, interpolation="bilinear")
    ax.set_title("corrected residual difference")
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    corr = correction_meta["correction"]
    ax.text(0.01, 0.9, (
        "0.5 mm axicon/mask offset: degraded -> corrected by seeded digital recentre. "
        f"inferred mask_recentre_x={float(corr['mask_recentre_x_um']):.1f} um, y={float(corr['mask_recentre_y_um']):.1f} um. "
        "This is a correction demonstration, not a new physics route."
    ), va="top", ha="left", fontsize=10, bbox={"facecolor": "#f6f6f6", "edgecolor": "#777", "pad": 5})
    fig.suptitle("MODE 2W-FIX Figure 3C: degraded vs corrected recovery", fontsize=14, weight="bold")
    return _savefig(fig, path_png, path_pdf)


def _plot_transverse_evolution(path_png: Path, path_pdf: Path, bench: Mapping[str, Any]) -> tuple[Path, Path]:
    plt, _ = _mpl()
    # Reuse the already stored realistic x-z planes for z evolution.  The route result stores xz/yz and reference;
    # for exact xy planes at requested z values this mode reruns a compact propagation through the realistic field.
    from vbb_study.digital_twin.nathan_vector_hexagon import mode2n_propagate_through_source_axicon

    prop = mode2n_propagate_through_source_axicon(bench["realistic"].pre_axicon_field[0], bench["realistic"].pre_axicon_field[1], bench["data"])
    stack = np.asarray(prop["intensity_stack"], dtype=float)
    z = np.asarray(prop["z_values_m"], dtype=float)
    grid = bench["data"]["grid"]
    wanted_mm = [0.1, 30, 60, 90, 150, 200]
    fixed_idx = int(np.argmin(np.abs(z / 1e-3 - wanted_mm[0])))
    sas_zoom = [
        _sas_zoom_plane(
            bench["realistic"].pre_axicon_field[0],
            bench["realistic"].pre_axicon_field[1],
            bench,
            z_m=value * 1e-3,
            pad_factor=3,
        )
        for value in wanted_mm[1:]
    ]
    fig = plt.figure(figsize=(16.2, 8.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0])
    ref = int(bench["realistic"].reference_index)
    ax.imshow(_normalise_image(stack[ref], local=True), origin="lower", extent=_extent_mm(grid), cmap="inferno", interpolation="bilinear")
    ax.set_title("physical 10 mm context at z=60 mm")
    ax.set_xlabel("x mm")
    ax.set_ylabel("y mm")
    native_crop, native_sl = _crop(stack[fixed_idx], 0.24)
    panels = [{
        "plane": native_crop,
        "grid": grid,
        "slices": native_sl,
        "title": f"native crop z={z[fixed_idx] / 1e-3:.1f} mm",
        "dx_um": float(bench["config"].window_m / bench["config"].grid_n / 1e-6),
    }]
    for zoom in sas_zoom:
        crop, sl = _crop(zoom["intensity"], 0.72)
        panels.append({
            "plane": crop,
            "grid": zoom["grid"],
            "slices": sl,
            "title": f"SAS zoom z={zoom['z_m'] / 1e-3:.0f} mm",
            "dx_um": float(zoom["output_dx_m"] / 1e-6),
        })
    vmax = max(float(np.percentile(panel["plane"], 99.8)) for panel in panels)
    for k, panel in enumerate(panels):
        ax = fig.add_subplot(gs[(k + 1) // 4, (k + 1) % 4])
        ax.imshow(
            _normalise_image(panel["plane"], local=False, vmax=vmax),
            origin="lower",
            extent=_extent_mm(panel["grid"], panel["slices"]),
            cmap="inferno",
            interpolation="lanczos",
            resample=True,
        )
        ax.set_title(f"{panel['title']}\ndx={panel['dx_um']:.2f} um")
        ax.set_xlabel("x mm")
        ax.set_ylabel("y mm")
    fig.suptitle("MODE 2W-FIX Figure 4A: transverse evolution with physical context plus SAS-scaled focus crops", fontsize=14, weight="bold")
    return _savefig(fig, path_png, path_pdf)


def _plot_propagation(path_png: Path, path_pdf: Path, bench: Mapping[str, Any]) -> tuple[Path, Path]:
    plt, _ = _mpl()
    prop = bench["realistic"]
    z = np.asarray(prop.z_values_m, dtype=float)
    grid = bench["data"]["grid"]
    xz = np.asarray(prop.xz_map, dtype=float)
    yz = np.asarray(prop.yz_map, dtype=float)
    useful = np.asarray(bench["useful_mask"], dtype=bool)
    # Recompute stack summary at the stored z values for useful power/ring peak.
    from vbb_study.digital_twin.nathan_vector_hexagon import mode2n_propagate_through_source_axicon

    full = mode2n_propagate_through_source_axicon(prop.pre_axicon_field[0], prop.pre_axicon_field[1], bench["data"])["intensity_stack"]
    mid = full.shape[1] // 2
    on_axis = full[:, mid, mid]
    ring_peak = np.asarray([float(np.max(p[useful])) for p in full])
    useful_power = np.asarray([float(np.sum(p[useful])) for p in full])
    fig = plt.figure(figsize=(16.0, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.8])
    extent_xz = [_extent_mm(grid)[0], _extent_mm(grid)[1], float(z[0] / 1e-3), float(z[-1] / 1e-3)]
    extent_yz = [_extent_mm(grid)[2], _extent_mm(grid)[3], float(z[0] / 1e-3), float(z[-1] / 1e-3)]
    for ax, arr, title, ext in [(fig.add_subplot(gs[0, 0]), xz, "x-z centre slice", extent_xz), (fig.add_subplot(gs[0, 1]), yz, "y-z centre slice", extent_yz)]:
        ax.imshow(_normalise_image(arr, local=True), origin="lower", aspect="auto", extent=ext, cmap="inferno", interpolation="bilinear")
        ax.axhline(60.0, color="white", lw=1.0, ls="--")
        ax.axhspan(30.0, 90.0, color="white", alpha=0.08)
        ax.set_title(title)
        ax.set_xlabel("transverse mm")
        ax.set_ylabel("z mm")
    ax = fig.add_subplot(gs[1, :])
    ax.plot(z / 1e-3, on_axis / max(float(np.max(on_axis)), EPS), label="on-axis")
    ax.plot(z / 1e-3, ring_peak / max(float(np.max(ring_peak)), EPS), label="ring peak")
    ax.plot(z / 1e-3, useful_power / max(float(np.max(useful_power)), EPS), label="useful power")
    ax.axvline(60.0, color="0.25", lw=1.0, ls="--", label="z=60 mm")
    ax.axvspan(30.0, 90.0, color="#dbeafe", alpha=0.45, label="publication focus zone")
    ax.set_xlabel("z mm")
    ax.set_ylabel("normalised diagnostic")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=4)
    ax.set_title("z diagnostics")
    fig.suptitle("MODE 2W-FIX Figure 4B: propagation maps and z diagnostics", fontsize=14, weight="bold")
    return _savefig(fig, path_png, path_pdf)


def _plot_power(path_png: Path, path_pdf: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    plt, _ = _mpl()
    keep = [row for row in rows if row["stage"] in {
        "01_laser_input", "02_after_pol_hwp_prep", "03_after_slm1",
        "05_after_slm2_effective_modulated_field", "07_selected_plus1_after_common_4f",
        "08_after_qwp", "09_after_axicon", "10_total_power_at_z60", "11_useful_hexagon_region_power",
    }]
    labels = [row["stage"].replace("_", "\n") for row in keep]
    vals = [float(row["model_fraction_of_input"]) for row in keep]
    fig, ax = plt.subplots(figsize=(15.5, 6.2), constrained_layout=True)
    bars = ax.bar(np.arange(len(vals)), vals, color="#5c8fbb")
    ax.set_ylim(0, 1.08)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylabel("fraction of input")
    ax.set_title("MODE 2W-FIX Figure 4C: sequential-route power flow (no split-arm stages)")
    for bar, value in zip(bars, vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{100*value:.1f}%", ha="center", va="bottom", fontsize=8)
    ax.text(0.02, 0.03, "1 W example equals the fraction in W; 10 W example is 10x. Linear scaling only, not a damage-threshold claim.",
            transform=ax.transAxes, fontsize=9, bbox={"facecolor": "#f6f6f6", "edgecolor": "#777", "pad": 5})
    return _savefig(fig, path_png, path_pdf)


def _plot_tolerances(path_png: Path, path_pdf: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    plt, _ = _mpl()
    fig, ax = plt.subplots(figsize=(15.5, 8.2), constrained_layout=True)
    ax.axis("off")
    display = [[r["label"], r["tested_range"], r["worst_tested_passing_value"], r["first_failing_value"], "" if r["minimum_correlation"] == "" else f"{float(r['minimum_correlation']):.4f}", r["strict_class_status"]] for r in rows]
    table = ax.table(
        cellText=display,
        colLabels=["parameter", "tested range", "worst passing", "first failure", "min corr", "strict class"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.16, 0.18, 0.16, 0.14, 0.10, 0.26],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.7)
    ax.set_title("MODE 2W-FIX Figure 5A: tolerance limits, not pass-fraction bars", fontsize=14, weight="bold")
    return _savefig(fig, path_png, path_pdf)


def _plot_combined_correction(path_png: Path, path_pdf: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    plt, _ = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), constrained_layout=True)
    combined = [r for r in rows if r["case_group"] == "combined_realism"]
    loops = [r for r in rows if r["case_group"] == "closed_loop_recovery"]
    axes[0].bar([r["case_id"].replace("combined_", "").replace("_lab", "") for r in combined], [float(r["before_corr_to_reference"]) for r in combined],
                color=["#70ad78" if r["before_strict_eligible"] else "#c35b5b" for r in combined])
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("combined realism cases")
    axes[0].set_ylabel("correlation to reference")
    idx = np.arange(len(loops))
    axes[1].bar(idx - 0.18, [float(r["before_corr_to_reference"]) for r in loops], width=0.36, label="before")
    axes[1].bar(idx + 0.18, [float(r["after_corr_to_reference"]) for r in loops], width=0.36, label="after")
    axes[1].axhline(STRICT_BASELINE_CORR_MIN, color="0.3", lw=0.9, ls="--", label="strict floor")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels([r["case_id"].replace("_", "\n") for r in loops], fontsize=7)
    axes[1].set_ylim(0.75, 1.005)
    axes[1].set_title("measured-image correction recovery")
    axes[1].legend()
    fig.suptitle("MODE 2W-FIX Figure 5B: combined realism and correction results", fontsize=14, weight="bold")
    return _savefig(fig, path_png, path_pdf)


def _plot_supplementary(path_png: Path, path_pdf: Path, rows: Sequence[Mapping[str, Any]]) -> tuple[Path, Path]:
    plt, _ = _mpl()
    fig, ax = plt.subplots(figsize=(14.0, 5.4), constrained_layout=False)
    ax.axis("off")
    cells = [[r["section"], r["item"], r["value"], r["status"]] for r in rows]
    table = ax.table(
        cellText=cells,
        colLabels=["section", "item", "value", "status"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.02, 0.08, 0.96, 0.78],
        colWidths=[0.18, 0.25, 0.38, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.55)
    ax.set_title("MODE 2W-FIX Figure A1: compact supplementary parameter/provenance table", fontsize=14, weight="bold", y=0.96)
    return _savefig(fig, path_png, path_pdf)


def supplementary_rows() -> list[dict[str, Any]]:
    return [
        {"section": "architecture", "item": "canonical route", "value": "SLM1 -> swap? -> SLM2 -> 4F -> QWP -> axicon", "status": "corrected"},
        {"section": "operating point", "item": "canonical", "value": CANONICAL_OPERATING_POINT_ID, "status": "strict eligible"},
        {"section": "operating point", "item": "secondary", "value": STRICT_COMPROMISE_ID, "status": "strict eligible"},
        {"section": "forbidden", "item": "old compromise", "value": OLD_BEST_COMPROMISE_ID, "status": "forbidden"},
        {"section": "hardware", "item": "SLMs", "value": "PLUTO-2.1 NIR-149; 1920x1080 @ 8 um; identity external", "status": "label/manual pending"},
        {"section": "hardware", "item": "carrier / 4F", "value": "6.25 lp/mm, f=300 mm, +1=1.929 mm, iris D~1.54 mm", "status": "routine calibration"},
        {"section": "calibration", "item": "remaining", "value": "LUT; iris/focal; camera/z; parity; QWP; centring", "status": "routine"},
        {"section": "claim boundary", "item": "microfabrication/sample plane", "value": "no success claim", "status": "blocked separate branch"},
    ]


def _write_doc(path: Path, eq: Mapping[str, Any], readiness: Mapping[str, Any], root: Path) -> Path:
    text = f"""# Nathan MODE 2W-FIX - Sequential Architecture and Master Figure Repair

**Status:** MODE 2W-FIX presentation/source-audit correction. The original MODE 2W pack is superseded
and must not be treated as the publication/presentation pass.

## Sequential Architecture

The accepted architecture is collinear and sequential:

`1029 nm Gaussian -> POL1/HWP equal H/V prep -> SLM1(phi_H=+alpha+carrier) -> optional swap HWP
-> SLM2(phi_V=-alpha+pi/2+carrier) -> optional swap-back -> common 4F -> QWP -> axicon -> camera`.

No canonical PBS split/recombine H/V interferometer arms are used.

## Equivalence Results

- sequential pre-QWP overlap to abstract H/V synthesis: `{float(eq['sequential_pre_qwp_overlap_to_abstract']):.12f}`
- sequential post-QWP overlap to validated target: `{float(eq['sequential_post_qwp_overlap_to_target']):.12f}`
- ideal sequential z=60 correlation to V0: `{float(eq['ideal_sequential_z60_corr_to_v0']):.12f}`
- realistic sequential z=60 correlation to V0: `{float(eq['realistic_sequential_z60_corr_to_v0']):.12f}`
- realistic strict class: `{eq['realistic_sequential_strict_class']}`
- realistic strict hexagon: `{eq['realistic_sequential_strict_hexagon']}`

Swap HWPs are required only for the same-panel-orientation implementation. The rotated/orthogonal
SLM2 route is valid if the LC-director orientation and mount geometry are confirmed on the bench.

## Source Audit

Every primary beam panel is recorded in `01_source_audit/mode2w_fix_numerical_source_audit.csv`.
The hero ideal-vs-realistic comparison uses N=1536 native numerical data. Interpolation is display-only;
metrics are computed on native arrays. N=384 data is not used for primary hero comparison figures.
The close-up beam panels in Figures 3A and 4A use scalable angular-spectrum (SAS) zoom rendering:
the audit records both the native input dx and the scaled output dx, so zoom fidelity is separated
from classifier/metric provenance.

## Corrected Power Model

The sequential power ledger removes split-arm H/V and PBS-recombination stages. It tracks a single
beam through preparation, SLM1, optional swap, SLM2, common 4F, QWP, axicon, z=60 and useful-region
power.

## Outcome

Outcome **{readiness['selected_outcome']}**: {readiness['outcome_statement']}

No microfabrication/sample-plane success claim is made.

Output root: `{root}`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_mode2w_fix_sequential_master(
    *,
    output_dir: str | Path = MODE2WF_DEFAULT_OUTPUT_ROOT,
    doc_path: str | Path = MODE2WF_DOC_PATH,
    primary_grid_n: int = 1536,
    primary_z_planes: int = 41,
    secondary_grid_n: int = 1024,
    secondary_z_planes: int = 41,
) -> dict[str, Any]:
    """Write the corrected sequential figure pack and all audit tables."""

    root = Path(output_dir)
    dirs = {name: root / name for name in (
        "00_architecture", "01_source_audit", "02_target_masks", "03_ideal_vs_realistic",
        "04_realism", "05_correction", "06_propagation", "07_power", "08_tolerances",
        "09_supplementary", "10_final_status",
    )}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    assert_not_forbidden(CANONICAL_OPERATING_POINT_ID)
    assert_not_forbidden(STRICT_COMPROMISE_ID)
    canonical, _ = load_operating_points()

    primary_cfg = _source_config(grid_n=primary_grid_n, z_planes=primary_z_planes, z_start_m=30e-3, z_end_m=90e-3)
    secondary_cfg = _source_config(grid_n=secondary_grid_n, z_planes=secondary_z_planes, z_start_m=0.1e-3, z_end_m=200e-3)
    primary = _bench_from_config(primary_cfg)
    secondary = _bench_from_config(secondary_cfg)
    masks = build_native_masks(canonical)
    eq = sequential_jones_equivalence(primary)
    ideal_cases, ideal_metrics = _ideal_cases(primary)
    realism_cases, realism_metrics = _realism_cases(secondary)
    correction_cases, correction_metrics, correction_meta = _correction_cases(secondary)
    power_rows = sequential_power_ledger(primary, canonical)
    tol_rows = tolerance_limit_rows()
    combined_rows = combined_correction_summary_rows()
    supp_rows = supplementary_rows()

    source_rows: list[dict[str, Any]] = []
    for panel in ("sector_mask", "alpha_theta", "orientation_map", "S0", "S1", "S2", "S3", "abs_Ex", "abs_Ey"):
        source_rows.append(_source_row(
            figure_id="fig2",
            panel_id=panel,
            route="source_scale_target_vector_field",
            cfg=primary_cfg,
            module="nathan_vector_hexagon.mode2n_source_target",
            source="fresh N=1536 source-grid target/Stokes field",
            display_interpolation=True,
            higher_n_exists=False,
        ))
    panel_meta = dict(masks["metadata"]["panel"])
    for panel, phase_name in (("SLM1_native_wrapped_phase", "phi_H = +alpha + carrier"), ("SLM2_native_wrapped_phase", "phi_V = -alpha + pi/2 + carrier")):
        source_rows.append({
            "figure_id": "fig2",
            "panel_id": panel,
            "route_or_case": "native_panel_phase_mask",
            "originating_module": "nathan_mode2v_lab_ready_build.build_native_masks",
            "originating_output_file_or_function": "build_native_masks",
            "numerical_N": int(panel_meta["width_px"]),
            "numerical_M_y": int(panel_meta["height_px"]),
            "physical_window_m": float(panel_meta["active_width_m"]),
            "physical_window_y_m": float(panel_meta["active_height_m"]),
            "z_planes": 0,
            "z_start_mm": "",
            "z_end_mm": "",
            "numerical_dx_m": float(panel_meta["pixel_pitch_m"]),
            "samples_per_radial_fringe": "",
            "samples_per_carrier_period": float(masks["metadata"]["carrier_period_slm_pixels"]),
            "display_interpolation_used": False,
            "metrics_computed_on_native_data": True,
            "higher_N_validated_source_exists": False,
            "phase_convention": phase_name,
            "lut_applied": bool(masks["metadata"]["lut_applied"]),
        })
    for case in ideal_cases:
        for panel in ("linear_crop", "log_crop", "difference", "profiles", "angular_profile"):
            row = _source_row(
                figure_id="fig3A",
                panel_id=f"{case['route_id']}_{panel}",
                route=case["route_id"],
                cfg=primary_cfg,
                module="nathan_mode2w_fix_sequential_master",
                source="fresh N=1536 source-scale field rendered with SAS-scaled z=60 zoom",
                display_interpolation=panel in {"linear_crop", "log_crop", "difference"},
                higher_n_exists=False,
            )
            sas_dx = _sas_output_dx(primary_cfg, float(primary_cfg.z_reference_m), 2)
            row.update({
                "render_method": "scalable_angular_spectrum_zoom",
                "sas_pad_factor": 2,
                "native_input_N": int(primary_cfg.grid_n),
                "native_input_dx_m": float(primary_cfg.window_m / primary_cfg.grid_n),
                "native_input_samples_per_radial_fringe": _samples_per_radial_fringe(primary_cfg),
                "numerical_dx_m": sas_dx,
                "physical_window_m": float(primary_cfg.grid_n) * sas_dx,
                "samples_per_radial_fringe": _samples_per_radial_fringe_from_dx(primary_cfg, sas_dx),
                "metrics_native_grid_N": int(primary_cfg.grid_n),
                "metrics_native_dx_m": float(primary_cfg.window_m / primary_cfg.grid_n),
            })
            source_rows.append(row)
    for case in realism_cases:
        source_rows.append(_source_row(
            figure_id="fig3B",
            panel_id=case["route_id"],
            route=case["route_id"],
            cfg=secondary_cfg,
            module="nathan_mode2w_fix_sequential_master",
            source="fresh N=1024 realism propagation; N=384 tolerance CSV used only for summary tables",
            display_interpolation=True,
            higher_n_exists=case["route_id"] == "clean_realistic",
        ))
    for case in correction_cases:
        source_rows.append(_source_row(
            figure_id="fig3C",
            panel_id=case["route_id"],
            route=case["route_id"],
            cfg=secondary_cfg,
            module="nathan_mode2w_fix_sequential_master",
            source="fresh N=1024 correction visualisation; M2V closed-loop CSV used only for metrics summary",
            display_interpolation=True,
            higher_n_exists=case["route_id"] == "target_realistic",
        ))
    source_rows.append(_source_row(
        figure_id="fig4",
        panel_id="physical_context",
        route="realistic_sequential_dual_slm_4f",
        cfg=secondary_cfg,
        module="nathan_mode2w_fix_sequential_master",
        source="fresh N=1024 broad-z fixed-grid ASM context",
        display_interpolation=True,
        higher_n_exists=True,
    ))
    native_near = _source_row(
        figure_id="fig4",
        panel_id="native_focus_crop_z0p1mm",
        route="realistic_sequential_dual_slm_4f",
        cfg=secondary_cfg,
        module="nathan_mode2w_fix_sequential_master",
        source="fresh N=1024 fixed-grid ASM near-axicon crop",
        display_interpolation=True,
        higher_n_exists=False,
    )
    native_near["render_method"] = "native_fixed_grid_crop"
    source_rows.append(native_near)
    for value in (30.0, 60.0, 90.0, 150.0, 200.0):
        sas_dx = _sas_output_dx(secondary_cfg, value * 1e-3, 3)
        row = _source_row(
            figure_id="fig4",
            panel_id=f"sas_focus_crop_z{value:g}mm",
            route="realistic_sequential_dual_slm_4f",
            cfg=secondary_cfg,
            module="nathan_mode2w_fix_sequential_master",
            source="fresh N=1024 source-scale field rendered with SAS-scaled focus zoom",
            display_interpolation=True,
            higher_n_exists=True,
        )
        row.update({
            "render_method": "scalable_angular_spectrum_zoom",
            "sas_pad_factor": 3,
            "native_input_N": int(secondary_cfg.grid_n),
            "native_input_dx_m": float(secondary_cfg.window_m / secondary_cfg.grid_n),
            "native_input_samples_per_radial_fringe": _samples_per_radial_fringe(secondary_cfg),
            "z_render_mm": value,
            "numerical_dx_m": sas_dx,
            "physical_window_m": float(secondary_cfg.grid_n) * sas_dx,
            "samples_per_radial_fringe": _samples_per_radial_fringe_from_dx(secondary_cfg, sas_dx),
            "metrics_native_grid_N": int(secondary_cfg.grid_n),
            "metrics_native_dx_m": float(secondary_cfg.window_m / secondary_cfg.grid_n),
        })
        source_rows.append(row)
    for panel in ("xz", "yz", "z_diagnostics"):
        source_rows.append(_source_row(
            figure_id="fig4",
            panel_id=panel,
            route="realistic_sequential_dual_slm_4f",
            cfg=secondary_cfg,
            module="nathan_mode2w_fix_sequential_master",
            source="fresh N=1024 broad-z fixed-grid ASM propagation diagnostics",
            display_interpolation=panel in {"xz", "yz"},
            higher_n_exists=True,
        ))

    _write_csv(dirs["00_architecture"] / "mode2w_fix_sequential_architecture.csv", sequential_architecture_rows())
    _write_json(dirs["00_architecture"] / "mode2w_fix_sequential_architecture.json", sequential_architecture_rows())
    _write_csv(dirs["00_architecture"] / "mode2w_fix_sequential_variants.csv", sequential_variant_rows())
    _write_json(dirs["00_architecture"] / "mode2w_fix_sequential_variants.json", sequential_variant_rows())
    _write_csv(dirs["00_architecture"] / "mode2w_fix_phase_conventions.csv", sequential_phase_convention_rows())
    _write_json(dirs["00_architecture"] / "mode2w_fix_phase_conventions.json", sequential_phase_convention_rows())
    _write_json(dirs["00_architecture"] / "mode2w_fix_sequential_equivalence.json", eq)
    _write_csv(dirs["01_source_audit"] / "mode2w_fix_numerical_source_audit.csv", source_rows)
    _write_json(dirs["01_source_audit"] / "mode2w_fix_numerical_source_audit.json", source_rows)
    _write_json(dirs["02_target_masks"] / "mode2w_fix_slm_mask_metadata.json", masks["metadata"])
    _write_csv(dirs["03_ideal_vs_realistic"] / "mode2w_fix_ideal_vs_realistic_metrics.csv", ideal_metrics)
    _write_json(dirs["03_ideal_vs_realistic"] / "mode2w_fix_ideal_vs_realistic_metrics.json", ideal_metrics)
    _write_csv(dirs["04_realism"] / "mode2w_fix_realism_metrics.csv", realism_metrics)
    _write_json(dirs["04_realism"] / "mode2w_fix_realism_metrics.json", realism_metrics)
    _write_csv(dirs["05_correction"] / "mode2w_fix_correction_metrics.csv", correction_metrics)
    _write_json(dirs["05_correction"] / "mode2w_fix_correction_metrics.json", {"rows": correction_metrics, "correction": correction_meta})
    _write_csv(dirs["07_power"] / "mode2w_fix_sequential_power_ledger.csv", power_rows)
    _write_json(dirs["07_power"] / "mode2w_fix_sequential_power_ledger.json", power_rows)
    _write_csv(dirs["08_tolerances"] / "mode2w_fix_tolerance_limits.csv", tol_rows)
    _write_json(dirs["08_tolerances"] / "mode2w_fix_tolerance_limits.json", tol_rows)
    _write_csv(dirs["08_tolerances"] / "mode2w_fix_combined_correction_summary.csv", combined_rows)
    _write_json(dirs["08_tolerances"] / "mode2w_fix_combined_correction_summary.json", combined_rows)
    _write_csv(dirs["09_supplementary"] / "mode2w_fix_supplementary_table.csv", supp_rows)
    _write_json(dirs["09_supplementary"] / "mode2w_fix_supplementary_table.json", supp_rows)

    figures = {
        "fig1": _plot_sequential_architecture(dirs["00_architecture"] / "fig1_sequential_optical_architecture.png", dirs["00_architecture"] / "fig1_sequential_optical_architecture.pdf"),
        "fig2": _plot_target_masks(dirs["02_target_masks"] / "fig2_target_and_sequential_masks.png", dirs["02_target_masks"] / "fig2_target_and_sequential_masks.pdf", primary, masks),
        "fig3A": _plot_three_route_comparison(dirs["03_ideal_vs_realistic"] / "fig3A_v0_ideal_realistic_sequential.png", dirs["03_ideal_vs_realistic"] / "fig3A_v0_ideal_realistic_sequential.pdf", primary, ideal_cases, ideal_metrics),
        "fig3B": _plot_realism(dirs["04_realism"] / "fig3B_clean_moderate_bad_realism.png", dirs["04_realism"] / "fig3B_clean_moderate_bad_realism.pdf", secondary, realism_cases, realism_metrics),
        "fig3C": _plot_correction(dirs["05_correction"] / "fig3C_degraded_corrected_recovery.png", dirs["05_correction"] / "fig3C_degraded_corrected_recovery.pdf", secondary, correction_cases, correction_metrics, correction_meta),
        "fig4A": _plot_transverse_evolution(dirs["06_propagation"] / "fig4A_transverse_evolution.png", dirs["06_propagation"] / "fig4A_transverse_evolution.pdf", secondary),
        "fig4B": _plot_propagation(dirs["06_propagation"] / "fig4B_xz_yz_z_diagnostics.png", dirs["06_propagation"] / "fig4B_xz_yz_z_diagnostics.pdf", secondary),
        "fig4C": _plot_power(dirs["07_power"] / "fig4C_sequential_power_flow.png", dirs["07_power"] / "fig4C_sequential_power_flow.pdf", power_rows),
        "fig5A": _plot_tolerances(dirs["08_tolerances"] / "fig5A_tolerance_limits.png", dirs["08_tolerances"] / "fig5A_tolerance_limits.pdf", tol_rows),
        "fig5B": _plot_combined_correction(dirs["08_tolerances"] / "fig5B_combined_correction_results.png", dirs["08_tolerances"] / "fig5B_combined_correction_results.pdf", combined_rows),
        "figA1": _plot_supplementary(dirs["09_supplementary"] / "figA1_supplementary_parameter_table.png", dirs["09_supplementary"] / "figA1_supplementary_parameter_table.pdf", supp_rows),
    }
    low_res_primary = [row for row in source_rows if row["figure_id"] == "fig3A" and int(row["numerical_N"]) < 1536]
    if float(eq["sequential_pre_qwp_overlap_to_abstract"]) < 0.999999 or float(eq["sequential_post_qwp_overlap_to_target"]) < 0.999999:
        outcome = "M2WF-C"
        statement = "The sequential architecture cannot realise the validated H/V target under the current SLM polarisation assumptions."
    elif low_res_primary:
        outcome = "M2WF-D"
        statement = "The figure source audit found insufficient numerical resolution in a primary hero comparison panel."
    elif not bool(eq["realistic_sequential_strict_hexagon"]):
        outcome = "M2WF-B"
        statement = "Sequential ideal equivalence passes, but realistic sequential hardware terms materially change the output."
    else:
        outcome = "M2WF-A"
        statement = (
            "Sequential physical implementation is numerically equivalent to the validated abstract H/V synthesis; "
            "the realistic sequential route preserves the strict hexagon; the power ledger is corrected; and the "
            "redesigned figure pack is readable and scientifically coherent."
        )
    readiness = {
        "stage": MODE2WF_STAGE,
        "selected_outcome": outcome,
        "allowed_outcomes": MODE2WF_ALLOWED_OUTCOMES,
        "outcome_statement": statement,
        "sequential_architecture_valid": outcome in {"M2WF-A", "M2WF-B"},
        "swap_hwps_required_same_orientation": True,
        "rotated_panel_route_valid": True,
        "sequential_pre_qwp_overlap": float(eq["sequential_pre_qwp_overlap_to_abstract"]),
        "sequential_post_qwp_overlap": float(eq["sequential_post_qwp_overlap_to_target"]),
        "z60_correlation": float(eq["realistic_sequential_z60_corr_to_v0"]),
        "strict_hexagon_result": bool(eq["realistic_sequential_strict_hexagon"]),
        "low_resolution_primary_sources": low_res_primary,
        "sas_zoom_rendering_enabled": True,
        "sas_zoom_native_metrics_policy": "render on scaled ASM output grids; keep classifier/strict metrics on native validated arrays",
        "microfabrication_sample_plane_claim": False,
    }
    _write_json(dirs["10_final_status"] / "m2wf_outcome_report.json", readiness)
    doc = _write_doc(Path(doc_path), eq, readiness, root)
    manifest = {
        "stage": MODE2WF_STAGE,
        "output_root": str(root),
        "doc": str(doc),
        "figures": {key: [str(p) for p in value] for key, value in figures.items()},
        "selected_outcome": outcome,
        "canonical_operating_point": CANONICAL_OPERATING_POINT_ID,
        "secondary_operating_point": STRICT_COMPROMISE_ID,
        "microfabrication_sample_plane_claim": False,
    }
    manifest_path = _write_json(dirs["10_final_status"] / "mode2w_fix_manifest.json", manifest)
    return {
        "root": root,
        "figures": figures,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "doc": doc,
        "equivalence": eq,
        "readiness": readiness,
        "source_audit": source_rows,
        "power_rows": power_rows,
    }


__all__ = [
    "MODE2WF_STAGE",
    "MODE2WF_DEFAULT_OUTPUT_ROOT",
    "MODE2WF_DOC_PATH",
    "MODE2WF_ALLOWED_OUTCOMES",
    "sequential_architecture_rows",
    "sequential_variant_rows",
    "sequential_phase_convention_rows",
    "sequential_jones_equivalence",
    "sequential_power_ledger",
    "tolerance_limit_rows",
    "combined_correction_summary_rows",
    "write_mode2w_fix_sequential_master",
]
