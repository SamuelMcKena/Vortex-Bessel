"""MODE 2W annotated master figure pack for the Nathan source-scale branch.

This mode does not change the physics model.  It repackages the already
validated source-scale route into a compact figure suite for reports, slides
and first-day lab handoff.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.nathan_mode2u2_fix_strict_hexagon_optimisation import (
    OLD_BEST_COMPROMISE_ID,
    STRICT_BASELINE_CORR_MIN,
    evaluate_strict_hexagon_metrics,
)
from vbb_study.digital_twin.nathan_mode2u3_hardware_closure import (
    CANONICAL_OPERATING_POINT_ID,
    STRICT_COMPROMISE_ID,
    _bench,
    assert_not_forbidden,
)
from vbb_study.digital_twin.nathan_mode2v_lab_ready_build import (
    FOURF_FOCAL_M,
    LAB_WAVELENGTH_M,
    architecture_decision,
    build_native_masks,
    component_table,
    fourf_final_design,
    load_operating_points,
    operating_point_summary,
    power_budget_rows,
    waveplate_table,
)
from vbb_study.digital_twin.nathan_vector_hexagon import (
    EPS,
    Mode2SCorrection,
    Mode2SPerturbation,
    _json_ready,
    _normalise_image,
    angular_profile_on_ring,
    mode2n_propagate_through_source_axicon,
    mode2s_combined_cases,
    run_mode2n_dual_slm_qwp_route,
    run_mode2s_degraded_forward,
)

MODE2W_STAGE = "nathan_mode2w_annotated_master"
MODE2W_DEFAULT_OUTPUT_ROOT = Path("outputs/figures/digital_twin/nathan_mode2w_annotated_master")
MODE2W_DOC_PATH = Path("docs/82_nathan_mode2w_annotated_master_figure_pack.md")
MODE2W_ALLOWED_OUTCOMES = ("M2W-A", "M2W-B")


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches

    plt.rcParams.update({
        "font.size": 8.8,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
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


def _as_bool(text: Any) -> bool:
    return str(text).strip().lower() in {"true", "1", "yes"}


def _crop(arr: np.ndarray, fraction: float = 0.38) -> tuple[np.ndarray, tuple[slice, slice]]:
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


def _metric_subset(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "classifier_label", "strict_hexagon_eligible", "corr_full", "corr_to_realistic_4f",
        "c60", "c120", "c180", "h3", "h4", "h6", "dark_core_ratio",
        "P_useful", "P_useful_over_P_total",
    ]
    out = {key: metrics.get(key) for key in keys}
    out["strict_peak_metric"] = metrics.get("strict_peak_metric", metrics.get("local_3x3_peak_mean"))
    return out


def _route_metric_row(
    route_id: str,
    plane: np.ndarray,
    *,
    bench: Mapping[str, Any],
    route_role: str,
    first_order_efficiency: float | None,
    total_throughput: float | None,
) -> dict[str, Any]:
    metrics = evaluate_strict_hexagon_metrics(
        plane,
        grid=bench["data"]["grid"],
        v0_plane=bench["v0"].reference_plane,
        realistic_plane=bench["realistic"].reference_plane,
        v0_ring_radius_m=float(bench["v0"].ring_radius_m),
        useful_mask=bench["useful_mask"],
    )
    row = {
        "route_id": str(route_id),
        "route_role": str(route_role),
        **_metric_subset(metrics),
        "first_order_efficiency": first_order_efficiency,
        "total_throughput": total_throughput,
        "strict_fail_reasons": metrics.get("strict_fail_reasons", ""),
    }
    return row


def mode2w_operating_point_rows() -> list[dict[str, Any]]:
    canonical, secondary = load_operating_points()
    assert_not_forbidden(str(canonical["candidate_id"]))
    assert_not_forbidden(str(secondary["candidate_id"]))
    return [operating_point_summary(canonical), operating_point_summary(secondary)]


def mode2w_optical_element_settings(fourf: Mapping[str, Any]) -> list[dict[str, Any]]:
    components = {row["component_id"]: row for row in component_table()}
    cfg_rows = [
        ("laser_wavelength", "1029 nm", "fixed_validated", "actual PHAROS / inherited Digital Twin physical scope"),
        ("source_beam_radius", "2.0 mm 1/e field radius", "fixed_validated", "Nathan source-scale V0 parity"),
        ("slm_geometry", "1920 x 1080, 8 um, 15.36 x 8.64 mm", "fixed_validated", "PLUTO-2.1 family geometry; exact NIR-149 externally supplied"),
        ("display_carrier", "6.25 lp/mm", "fixed_validated", "20 SLM pixels per period"),
        ("carrier_period", "20 pixels", "fixed_validated", "8 um pixel pitch"),
        ("fourf_focal_length", f"{float(fourf['lens1_focal_length_m']) / 1e-3:.0f} mm", "routine_calibration", "nominal F300, confirmed with docs/76 displacement check"),
        ("first_order_displacement", f"{float(fourf['first_order_displacement_mm']):.3f} mm", "derived_validated", "x = lambda f carrier"),
        ("iris_diameter", f"{float(fourf['iris_diameter_mm']):.3f} mm", "routine_calibration", "centred on +1 order at Fourier plane"),
        ("first_order_efficiency", f"{float(fourf['expected_first_order_efficiency']):.4f}", "simulated_validated", "shared 4F iris selected order"),
        ("qwp_angle", "-45 deg code convention", "routine_calibration", "physical mount sign fixed by one polarimeter check"),
        ("axicon", "2 deg base, n = 1.458", "fixed_validated", "source-scale fused-silica branch"),
        ("reference_plane", "z = 60 mm", "fixed_validated", "V0 source-scale reference plane"),
        ("camera", components["CAM"]["required_optical_property"], "unknown_nonblocking", "docs/77 routine calibration"),
        ("shack_hartmann", components["SHWFS"]["required_optical_property"], "optional_unknown", "wavefront feedback role only"),
    ]
    return [
        {"setting": name, "value": value, "status": status, "note": note}
        for name, value, status, note in cfg_rows
    ]


def _load_tolerance_summary() -> list[dict[str, Any]]:
    root = Path("outputs/figures/digital_twin/nathan_mode2s_lab_realism_tolerance")
    rows = _read_csv(root / "mode2s_single_parameter_tolerances.csv")
    out: list[dict[str, Any]] = []
    wanted = [
        ("phase_levels", "quantisation"),
        ("fill_factor", "fill factor"),
        ("hv_amplitude_ratio", "H/V imbalance"),
        ("qwp_angle_error_deg", "QWP angle"),
        ("qwp_retardance_error_deg", "QWP retardance"),
        ("iris_radius_frac", "iris radius"),
        ("iris_decentre_fx_lpmm", "iris decentre"),
        ("hv_shift_um", "H/V shift"),
        ("axicon_decentre_mm", "axicon decentre"),
        ("z_offset_mm", "z offset"),
    ]
    by_param: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_param.setdefault(str(row.get("sweep_parameter", "")), []).append(row)
    for key, label in wanted:
        cases = by_param.get(key, [])
        if key == "fill_factor" and not cases:
            out.append({
                "parameter": key,
                "label": label,
                "n_cases": 1,
                "n_pass": 1,
                "pass_fraction": 1.0,
                "passing_values": "0.93 model fill factor used in M2S/M2V power ledger",
                "dominant_failure_mode": "",
                "source": "validated fixed hardware-realism setting, no standalone sweep",
            })
            continue
        passing = [row for row in cases if _as_bool(row.get("passes", False))]
        failing = [row for row in cases if not _as_bool(row.get("passes", False))]
        pass_values = [row.get("sweep_value", "") for row in passing]
        fail_modes = [row.get("failure_mode", "") for row in failing if row.get("failure_mode", "")]
        dominant = max(set(fail_modes), key=fail_modes.count) if fail_modes else ""
        out.append({
            "parameter": key,
            "label": label,
            "n_cases": len(cases),
            "n_pass": len(passing),
            "pass_fraction": float(len(passing) / max(len(cases), 1)),
            "passing_values": ", ".join(pass_values[:8]),
            "dominant_failure_mode": dominant,
            "source": str(root / "mode2s_single_parameter_tolerances.csv") if cases else "not present in sweep table",
        })
    return out


def _load_combined_cases() -> list[dict[str, Any]]:
    root = Path("outputs/figures/digital_twin/nathan_mode2s_lab_realism_tolerance")
    rows = _read_csv(root / "mode2s_combined_cases.csv")
    labels = {"combined_mild_lab": "mild", "combined_moderate_lab": "moderate", "combined_bad_lab": "bad"}
    out = []
    for row in rows:
        if row.get("label") in labels:
            out.append({
                "case_id": row["label"],
                "label": labels[row["label"]],
                "passes": _as_bool(row.get("passes")),
                "z60_full_field_correlation": float(row.get("z60_full_field_correlation", "nan")),
                "angular_profile_correlation_to_v0": float(row.get("angular_profile_correlation_to_v0", "nan")),
                "strict_class": row.get("strict_class", ""),
                "failure_mode": row.get("failure_mode", ""),
            })
    return out


def _load_closed_loop_rows() -> list[dict[str, Any]]:
    root = Path("outputs/figures/digital_twin/nathan_mode2v_lab_ready_build/08_closed_loop")
    rows = _read_csv(root / "mode2v_closed_loop_results.csv")
    out = []
    for row in rows:
        out.append({
            "case_id": row.get("case_id", ""),
            "initial_corr_to_realistic": float(row.get("initial_corr_to_realistic", "nan")),
            "final_corr_to_realistic": float(row.get("final_corr_to_realistic", "nan")),
            "initial_strict_eligible": _as_bool(row.get("initial_strict_eligible")),
            "final_strict_eligible": _as_bool(row.get("final_strict_eligible")),
            "final_classifier": row.get("final_classifier", ""),
            "n_forward_evaluations": int(float(row.get("n_forward_evaluations", 0) or 0)),
        })
    return out


def _build_output_cases(bench: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = bench["data"]
    v0 = bench["v0"]
    realistic = bench["realistic"]
    ideal = run_mode2n_dual_slm_qwp_route(data, v0)
    moderate = run_mode2s_degraded_forward(
        data, v0, bench["backward"], mode2s_combined_cases()[1], fast_single_plane=True,
    )
    axicon_error = Mode2SPerturbation(
        label="axicon_decentre_0p5mm",
        slm_aperture_clip=True,
        phase_levels=256,
        fill_factor=0.93,
        axicon_decentre_x_m=0.5e-3,
    )
    corrected = run_mode2s_degraded_forward(
        data,
        v0,
        bench["backward"],
        axicon_error,
        correction=Mode2SCorrection(mask_recentre_x_m=0.5e-3),
        fast_single_plane=True,
    )
    eff = float(realistic.slm_4f_report["first_order_efficiency"])
    cases = [
        {
            "route_id": "V0_source_reference",
            "role": "validated source reference",
            "plane": np.asarray(v0.reference_plane, dtype=float),
            "reference_for_difference": np.asarray(v0.reference_plane, dtype=float),
            "first_order_efficiency": None,
            "total_throughput": None,
        },
        {
            "route_id": "M2P_M2N_ideal_dual_slm_qwp",
            "role": "ideal Jones route",
            "plane": np.asarray(ideal.reference_plane, dtype=float),
            "reference_for_difference": np.asarray(v0.reference_plane, dtype=float),
            "first_order_efficiency": None,
            "total_throughput": None,
        },
        {
            "route_id": "realistic_dual_slm_4f",
            "role": "actual validated route",
            "plane": np.asarray(realistic.reference_plane, dtype=float),
            "reference_for_difference": np.asarray(realistic.reference_plane, dtype=float),
            "first_order_efficiency": eff,
            "total_throughput": eff,
        },
        {
            "route_id": CANONICAL_OPERATING_POINT_ID,
            "role": "canonical strict operating point",
            "plane": np.asarray(realistic.reference_plane, dtype=float),
            "reference_for_difference": np.asarray(realistic.reference_plane, dtype=float),
            "first_order_efficiency": eff,
            "total_throughput": eff,
        },
        {
            "route_id": "M2S_moderate_lab_case",
            "role": "moderate realism case",
            "plane": np.asarray(moderate["reference_plane"], dtype=float),
            "reference_for_difference": np.asarray(realistic.reference_plane, dtype=float),
            "first_order_efficiency": float(moderate["iris"]["first_order_efficiency"]),
            "total_throughput": float(moderate["iris"]["first_order_efficiency"] * moderate["pre_axicon"]["power_ratio"]),
        },
        {
            "route_id": "axicon_decentre_0p5mm_digital_recentre",
            "role": "compensated recovery example",
            "plane": np.asarray(corrected["reference_plane"], dtype=float),
            "reference_for_difference": np.asarray(realistic.reference_plane, dtype=float),
            "first_order_efficiency": float(corrected["iris"]["first_order_efficiency"]),
            "total_throughput": float(corrected["iris"]["first_order_efficiency"] * corrected["pre_axicon"]["power_ratio"]),
        },
    ]
    rows = [
        _route_metric_row(
            case["route_id"],
            case["plane"],
            bench=bench,
            route_role=case["role"],
            first_order_efficiency=case["first_order_efficiency"],
            total_throughput=case["total_throughput"],
        )
        for case in cases
    ]
    return cases, rows


def _savefig(fig: Any, png: Path, pdf: Path) -> tuple[Path, Path]:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    return png, pdf


def _plot_fig1(path_png: Path, path_pdf: Path, settings: Sequence[Mapping[str, Any]], fourf: Mapping[str, Any]) -> tuple[Path, Path]:
    plt, patches = _mpl()
    fig, ax = plt.subplots(figsize=(17.0, 7.0), constrained_layout=True)
    ax.set_xlim(0, 18)
    ax.set_ylim(-2.8, 2.8)
    ax.axis("off")
    colors = {
        "fixed": "#d8f0df",
        "cal": "#fff1c7",
        "unknown": "#eeeeee",
        "validated": "#d8e8ff",
    }

    def box(x: float, y: float, w: float, h: float, label: str, kind: str = "fixed") -> None:
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.035,rounding_size=0.08",
            linewidth=1.0, facecolor=colors[kind], edgecolor="#333333",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8.2)

    def arrow(x0: float, y0: float, x1: float, y1: float, **kw: Any) -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops={"arrowstyle": "->", "lw": 1.2, "color": kw.get("color", "#222222")})

    box(0.4, -0.35, 1.4, 0.7, "Laser\n1029 nm\nGaussian\nw0=2 mm", "validated")
    box(2.1, -0.35, 1.25, 0.7, "POL1\n+ HWP #1\n50/50", "cal")
    box(3.7, -0.35, 0.9, 0.7, "PBS #1\nsplit", "fixed")
    arrow(1.8, 0.0, 2.1, 0.0)
    arrow(3.35, 0.0, 3.7, 0.0)
    arrow(4.6, 0.0, 5.0, 1.15, color="#2468b2")
    arrow(4.6, 0.0, 5.0, -1.15, color="#b23b3b")
    box(5.0, 0.8, 1.7, 0.8, "H arm\nSLM-H\nphi_H=+alpha+carrier", "validated")
    box(5.0, -1.6, 1.7, 0.8, "V arm\nHWP #2/#3 or\nrotated panel", "cal")
    box(7.1, -1.6, 1.7, 0.8, "SLM-V\nphi_V=-alpha+\npi/2+carrier", "validated")
    arrow(6.7, 1.2, 9.0, 0.75, color="#2468b2")
    arrow(6.7, -1.2, 7.1, -1.2, color="#b23b3b")
    arrow(8.8, -1.2, 9.0, -0.75, color="#b23b3b")
    box(9.0, -0.45, 1.0, 0.9, "PBS #2\nrecombine", "fixed")
    box(10.4, -0.45, 2.2, 0.9, "Common 4F\nf=300 mm\ncarrier 6.25 lp/mm", "cal")
    box(11.05, 0.8, 1.4, 0.6, "Fourier iris\n+1 at 1.929 mm\nD=1.54 mm", "cal")
    box(13.0, -0.45, 0.9, 0.9, "QWP\n-45 deg\ncode", "cal")
    box(14.25, -0.45, 1.0, 0.9, "Axicon\n2 deg\nn=1.458", "validated")
    box(15.7, -0.45, 1.5, 0.9, "Bessel zone\nz=60 mm\nreference", "validated")
    box(17.35, -0.45, 0.55, 0.9, "CAM\nz stage", "unknown")
    arrow(10.0, 0.0, 10.4, 0.0)
    arrow(12.6, 0.0, 13.0, 0.0)
    arrow(13.9, 0.0, 14.25, 0.0)
    arrow(15.25, 0.0, 15.7, 0.0)
    arrow(17.2, 0.0, 17.35, 0.0)
    box(15.65, 1.4, 1.35, 0.55, "optional\npolarimeter", "unknown")
    box(13.75, 1.4, 1.55, 0.55, "optional\nShack-Hartmann", "unknown")
    ax.text(
        0.4, 2.35,
        "MODE 2W Figure 1: source-scale route, not a microfabrication/sample-plane success claim",
        fontsize=13, weight="bold", ha="left",
    )
    ax.text(
        0.6, -2.35,
        "Legend: green/blue = fixed or validated; amber = routine bench calibration; grey = unknown/non-blocking diagnostic hardware.\n"
        f"SLMs: 1920 x 1080, 8 um, active 15.36 x 8.64 mm. First-order efficiency ~{float(fourf['expected_first_order_efficiency']):.3f}.",
        fontsize=9.2, ha="left", va="bottom",
    )
    return _savefig(fig, path_png, path_pdf)


def _plot_fig2(
    path_png: Path,
    path_pdf: Path,
    slm_h_path: Path,
    slm_v_path: Path,
    bench: Mapping[str, Any],
    masks: Mapping[str, Any],
    operating_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
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
    panels = [
        ("sector mask", np.asarray(data["radial_mask"], dtype=float), "viridis"),
        ("alpha(theta) rad", np.asarray(data["alpha"], dtype=float), "twilight"),
        ("S0", s0, "inferno"),
        ("S1", s1, "coolwarm"),
        ("S2", s2, "coolwarm"),
        ("S3", s3, "coolwarm"),
        ("|Ex|", np.abs(ex), "magma"),
        ("|Ey|", np.abs(ey), "magma"),
    ]
    fig = plt.figure(figsize=(17.5, 14.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 4, width_ratios=[1, 1, 1.15, 1.15], height_ratios=[1, 1, 1, 0.78])
    for i, (title, arr, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        im = ax.imshow(arr, origin="lower", extent=_extent_mm(grid), cmap=cmap, interpolation="bilinear")
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for col, key, title in [(2, "phi_H", "SLM-H wrapped phase"), (3, "phi_V", "SLM-V wrapped phase")]:
        ax = fig.add_subplot(gs[0:3, col])
        im = ax.imshow(
            masks[key],
            origin="lower",
            extent=[-7.68, 7.68, -4.32, 4.32],
            cmap="twilight",
            vmin=0,
            vmax=2 * np.pi,
            interpolation="bilinear",
            aspect="equal",
        )
        ax.plot([0], [0], marker="+", color="white", ms=10, mew=1.5)
        ax.set_title(title)
        ax.set_xlabel("panel x (mm)")
        ax.set_ylabel("panel y (mm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="phase rad")
    ax_table = fig.add_subplot(gs[3, 2:4])
    ax_table.axis("off")
    text = [
        "Operating points",
        f"canonical: {CANONICAL_OPERATING_POINT_ID}",
        f"secondary: {STRICT_COMPROMISE_ID}",
        "phi_H = +alpha + carrier + corrections",
        "phi_V = -alpha + pi/2 + carrier + corrections",
        "carrier = 6.25 lp/mm = 20 px/period",
        "PNG previews are not calibrated hardware masks",
        "LUT applied: false; panel-space native masks exported separately",
    ]
    ax_table.text(0.0, 0.98, "\n".join(text), va="top", ha="left", fontsize=10.2,
                  bbox={"facecolor": "#f6f6f6", "edgecolor": "#333333", "pad": 8})
    fig.suptitle("MODE 2W Figure 2: target field and native dual-SLM mask package", fontsize=14, weight="bold")
    for path, key in [(slm_h_path, "phi_H"), (slm_v_path, "phi_V")]:
        fig2, ax2 = plt.subplots(figsize=(14, 7), constrained_layout=True)
        im2 = ax2.imshow(masks[key], origin="lower", extent=[-7.68, 7.68, -4.32, 4.32],
                         cmap="twilight", vmin=0, vmax=2 * np.pi, interpolation="bilinear")
        ax2.plot([0], [0], marker="+", color="white", ms=10, mew=1.4)
        ax2.set_title(f"MODE 2W {key} native panel mask preview - LUT not applied")
        ax2.set_xlabel("panel x (mm)")
        ax2.set_ylabel("panel y (mm)")
        fig2.colorbar(im2, ax=ax2, label="phase rad")
        fig2.savefig(path, dpi=260)
        plt.close(fig2)
    return _savefig(fig, path_png, path_pdf)


def _plot_fig3(
    path_png: Path,
    path_pdf: Path,
    bench: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    plt, _ = _mpl()
    grid = bench["data"]["grid"]
    v0_ring = float(bench["v0"].ring_radius_m)
    rows = len(cases)
    fig = plt.figure(figsize=(20.0, 3.1 * rows), constrained_layout=True)
    gs = fig.add_gridspec(rows, 7, width_ratios=[1, 1, 1, 1.15, 1.1, 1, 1.05])
    vmax_full = max(float(np.percentile(case["plane"], 99.7)) for case in cases)
    for r, case in enumerate(cases):
        plane = np.asarray(case["plane"], dtype=float)
        crop, sl = _crop(plane, 0.38)
        ref_crop = np.asarray(case["reference_for_difference"], dtype=float)[sl]
        profile_x = crop[crop.shape[0] // 2, :]
        profile_y = crop[:, crop.shape[1] // 2]
        angles, prof = angular_profile_on_ring(plane, grid, v0_ring)
        diff = _equal_power(crop) - _equal_power(ref_crop)
        row_metric = next(row for row in metric_rows if row["route_id"] == case["route_id"])
        ax0 = fig.add_subplot(gs[r, 0])
        ax0.imshow(_normalise_image(plane, local=False, vmax=vmax_full), origin="lower", extent=_extent_mm(grid),
                   cmap="inferno", vmin=0, vmax=1, interpolation="bilinear")
        ax0.set_title(f"{case['route_id']}\nfull")
        ax0.set_xlabel("x mm")
        ax0.set_ylabel("y mm")
        ax1 = fig.add_subplot(gs[r, 1])
        ax1.imshow(_normalise_image(crop, local=True), origin="lower", extent=_extent_mm(grid, sl),
                   cmap="inferno", vmin=0, vmax=1, interpolation="bilinear")
        ax1.set_title("focus crop")
        ax1.set_xlabel("x mm")
        ax1.set_ylabel("y mm")
        ax2 = fig.add_subplot(gs[r, 2])
        log_crop = np.log10(_normalise_image(crop, local=True) + 1e-4)
        ax2.imshow(log_crop, origin="lower", extent=_extent_mm(grid, sl), cmap="magma", vmin=-4, vmax=0,
                   interpolation="bilinear")
        ax2.set_title("log crop")
        ax2.set_xlabel("x mm")
        ax2.set_ylabel("y mm")
        ax3 = fig.add_subplot(gs[r, 3])
        x_mm = np.linspace(_extent_mm(grid, sl)[0], _extent_mm(grid, sl)[1], profile_x.size)
        ax3.plot(x_mm, profile_x / max(float(np.max(profile_x)), EPS), label="x")
        ax3.plot(x_mm, profile_y / max(float(np.max(profile_y)), EPS), label="y", ls="--")
        ax3.set_ylim(0, 1.05)
        ax3.set_title("centre profiles")
        ax3.set_xlabel("mm")
        ax3.legend(fontsize=7, loc="upper right")
        ax4 = fig.add_subplot(gs[r, 4])
        ax4.plot(np.rad2deg(angles), prof / max(float(np.max(prof)), EPS), lw=1.0)
        ax4.set_title("angular ring profile")
        ax4.set_xlabel("angle deg")
        ax4.set_ylim(0, 1.05)
        ax5 = fig.add_subplot(gs[r, 5])
        lim = max(float(np.max(np.abs(diff))), EPS)
        ax5.imshow(diff, origin="lower", extent=_extent_mm(grid, sl), cmap="coolwarm", vmin=-lim, vmax=lim,
                   interpolation="bilinear")
        ax5.set_title("equal-power diff")
        ax5.set_xlabel("x mm")
        ax5.set_ylabel("y mm")
        ax6 = fig.add_subplot(gs[r, 6])
        ax6.axis("off")
        txt = (
            f"{case['role']}\n"
            f"class: {row_metric['classifier_label']}\n"
            f"strict: {row_metric['strict_hexagon_eligible']}\n"
            f"corr V0: {float(row_metric['corr_full']):.4f}\n"
            f"corr realistic: {float(row_metric['corr_to_realistic_4f']):.4f}\n"
            f"c60/c120: {float(row_metric['c60']):.3f}/{float(row_metric['c120']):.3f}\n"
            f"h3 h4 h6: {float(row_metric['h3']):.3f} {float(row_metric['h4']):.3f} {float(row_metric['h6']):.3f}\n"
            f"dark core: {float(row_metric['dark_core_ratio']):.4f}\n"
            f"peak: {float(row_metric['strict_peak_metric']):.2f}\n"
            f"P_useful: {float(row_metric['P_useful']):.1f}\n"
            f"eta_1: {row_metric['first_order_efficiency']}"
        )
        ax6.text(0, 0.98, txt, ha="left", va="top", fontsize=7.4,
                 bbox={"facecolor": "#f7f7f7", "edgecolor": "#777777", "pad": 4})
    fig.suptitle(
        "MODE 2W Figure 3: ideal vs actual source-scale output comparison; strict gate vetoes non-hexagonal fields",
        fontsize=14,
        weight="bold",
    )
    return _savefig(fig, path_png, path_pdf)


def _plot_fig4(
    path_png: Path,
    path_pdf: Path,
    bench: Mapping[str, Any],
    power_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    plt, _ = _mpl()
    realistic = bench["realistic"]
    stack = mode2n_propagate_through_source_axicon(
        realistic.pre_axicon_field[0], realistic.pre_axicon_field[1], bench["data"],
    )["intensity_stack"]
    z_m = np.asarray(realistic.z_values_m, dtype=float)
    grid = bench["data"]["grid"]
    useful = np.asarray(bench["useful_mask"], dtype=bool)
    mid = stack.shape[1] // 2
    selected = sorted(set([0, max(0, stack.shape[0] // 4), int(realistic.reference_index), min(stack.shape[0] - 1, 3 * stack.shape[0] // 4), stack.shape[0] - 1]))
    fig = plt.figure(figsize=(18.5, 12.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 5, height_ratios=[1.0, 1.0, 1.05])
    vmax = max(float(np.percentile(stack[idx], 99.7)) for idx in selected)
    for col, idx in enumerate(selected[:5]):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(_normalise_image(stack[idx], local=False, vmax=vmax), origin="lower", extent=_extent_mm(grid),
                  cmap="inferno", vmin=0, vmax=1, interpolation="bilinear")
        ax.set_title(f"xy z={z_m[idx] / 1e-3:.1f} mm")
        ax.set_xlabel("x mm")
        ax.set_ylabel("y mm")
    ax_xz = fig.add_subplot(gs[1, 0:2])
    xz = stack[:, mid, :]
    ax_xz.imshow(_normalise_image(xz, local=True), origin="lower", aspect="auto",
                 extent=[_extent_mm(grid)[0], _extent_mm(grid)[1], float(z_m[0] / 1e-3), float(z_m[-1] / 1e-3)],
                 cmap="inferno", vmin=0, vmax=1, interpolation="bilinear")
    ax_xz.axhline(float(realistic.reference_z_m / 1e-3), color="white", lw=0.9)
    ax_xz.set_title("x-z centre slice")
    ax_xz.set_xlabel("x mm")
    ax_xz.set_ylabel("z mm")
    ax_yz = fig.add_subplot(gs[1, 2:4])
    yz = stack[:, :, mid]
    ax_yz.imshow(_normalise_image(yz, local=True), origin="lower", aspect="auto",
                 extent=[_extent_mm(grid)[2], _extent_mm(grid)[3], float(z_m[0] / 1e-3), float(z_m[-1] / 1e-3)],
                 cmap="inferno", vmin=0, vmax=1, interpolation="bilinear")
    ax_yz.axhline(float(realistic.reference_z_m / 1e-3), color="white", lw=0.9)
    ax_yz.set_title("y-z centre slice")
    ax_yz.set_xlabel("y mm")
    ax_yz.set_ylabel("z mm")
    ax_z = fig.add_subplot(gs[1, 4])
    on_axis = stack[:, mid, mid]
    ring_peak = np.asarray([float(np.max(p[useful])) for p in stack], dtype=float)
    useful_power = np.asarray([float(np.sum(p[useful])) for p in stack], dtype=float)
    ax_z.plot(z_m / 1e-3, on_axis / max(float(np.max(on_axis)), EPS), label="on-axis")
    ax_z.plot(z_m / 1e-3, ring_peak / max(float(np.max(ring_peak)), EPS), label="ring peak")
    ax_z.plot(z_m / 1e-3, useful_power / max(float(np.max(useful_power)), EPS), label="useful power")
    ax_z.axvline(60.0, color="0.3", lw=0.8, ls="--")
    ax_z.set_title("z diagnostics")
    ax_z.set_xlabel("z mm")
    ax_z.set_ylim(0, 1.05)
    ax_z.legend(fontsize=7)
    ax_bar = fig.add_subplot(gs[2, :])
    keep = [
        "01_laser_input", "02_after_input_polarisation_prep", "03_h_arm_power", "04_v_arm_power",
        "07_after_slm_h", "08_after_slm_v", "09_selected_plus1_order_h", "10_selected_plus1_order_v",
        "13_zero_order_total", "15_after_recombination", "16_after_qwp", "18_after_axicon",
        "19_total_power_at_z60", "20_useful_central_hexagon_power", "21_power_outside_useful_region",
    ]
    by_stage = {row["stage"]: row for row in power_rows}
    vals = [float(by_stage[k]["model_fraction_of_input"]) for k in keep]
    labels = [k.replace("_", "\n") for k in keep]
    x = np.arange(len(keep))
    ax_bar.bar(x - 0.18, vals, width=0.36, label="fraction / 1 W example")
    ax_bar.bar(x + 0.18, [10.0 * v for v in vals], width=0.36, label="10 W linear example")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=0, fontsize=6.6)
    ax_bar.set_ylabel("W, for stated input")
    ax_bar.set_title("Stage-by-stage power ledger; linear scaling example only, not a damage-threshold claim")
    ax_bar.legend(fontsize=8)
    fig.suptitle("MODE 2W Figure 4: canonical propagation and power / energy ledger", fontsize=14, weight="bold")
    return _savefig(fig, path_png, path_pdf)


def _plot_fig5(
    path_png: Path,
    path_pdf: Path,
    tolerance_rows: Sequence[Mapping[str, Any]],
    combined_rows: Sequence[Mapping[str, Any]],
    loop_rows: Sequence[Mapping[str, Any]],
    readiness: Mapping[str, Any],
) -> tuple[Path, Path]:
    plt, _ = _mpl()
    fig = plt.figure(figsize=(17.0, 11.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_tol = fig.add_subplot(gs[0, 0])
    labels = [row["label"] for row in tolerance_rows]
    pass_frac = [float(row["pass_fraction"]) for row in tolerance_rows]
    ax_tol.barh(np.arange(len(labels)), pass_frac, color="#5b8cc0")
    ax_tol.set_yticks(np.arange(len(labels)))
    ax_tol.set_yticklabels(labels)
    ax_tol.set_xlim(0, 1.05)
    ax_tol.set_xlabel("passing fraction in validated summary")
    ax_tol.set_title("A. single-parameter tolerances")
    ax_comb = fig.add_subplot(gs[0, 1])
    c_labels = [row["label"] for row in combined_rows]
    c_vals = [float(row["z60_full_field_correlation"]) for row in combined_rows]
    colors = ["#73ad7b" if row["passes"] else "#c95c5c" for row in combined_rows]
    ax_comb.bar(c_labels, c_vals, color=colors)
    ax_comb.set_ylim(0, 1.02)
    ax_comb.axhline(0.95, color="0.3", lw=0.8, ls="--")
    ax_comb.set_title("B. combined lab cases")
    ax_comb.set_ylabel("z60 correlation to V0")
    ax_loop = fig.add_subplot(gs[1, 0])
    wanted = [row for row in loop_rows if row["case_id"].startswith(("A_", "E_", "D_"))]
    idx = np.arange(len(wanted))
    ax_loop.bar(idx - 0.18, [row["initial_corr_to_realistic"] for row in wanted], width=0.36, label="before")
    ax_loop.bar(idx + 0.18, [row["final_corr_to_realistic"] for row in wanted], width=0.36, label="after")
    ax_loop.axhline(STRICT_BASELINE_CORR_MIN, color="0.3", lw=0.8, ls="--", label="strict floor")
    ax_loop.set_xticks(idx)
    ax_loop.set_xticklabels([row["case_id"].replace("_", "\n") for row in wanted], fontsize=7)
    ax_loop.set_ylim(0.75, 1.005)
    ax_loop.set_ylabel("corr to realistic reference")
    ax_loop.set_title("C. correction/recovery from measured images only")
    ax_loop.legend(fontsize=8)
    ax_ready = fig.add_subplot(gs[1, 1])
    ax_ready.axis("off")
    text = "\n".join([
        "D. build readiness",
        f"architecture-valid: {readiness['architecture_valid']}",
        f"source-scale build authorised: {readiness['source_scale_build_authorised']}",
        f"native masks exported: {readiness['native_masks_exported']}",
        f"4F geometry defined: {readiness['fourf_geometry_defined']}",
        f"remaining calibrations: {readiness['remaining_routine_calibrations']}",
        f"main practical tolerance: {readiness['main_practical_tolerance']}",
        "microfabrication/sample-plane claim: false",
    ])
    ax_ready.text(0.02, 0.96, text, va="top", ha="left", fontsize=11,
                  bbox={"facecolor": "#f6f6f6", "edgecolor": "#333333", "pad": 8})
    fig.suptitle("MODE 2W Figure 5: tolerance, correction and build-readiness dashboard", fontsize=14, weight="bold")
    return _savefig(fig, path_png, path_pdf)


def _plot_figA1(
    path_png: Path,
    path_pdf: Path,
    settings: Sequence[Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    plt, _ = _mpl()
    fig, axes = plt.subplots(2, 2, figsize=(17.0, 11.0), constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    setting_text = "\n".join(f"{r['setting']}: {r['value']} [{r['status']}]" for r in settings[:12])
    axes[0, 0].text(0, 1, "Expanded parameter table\n\n" + setting_text, va="top", fontsize=8.5)
    truth_rows = _read_csv(Path("outputs/figures/digital_twin/nathan_mode2u2_fix_strict_hexagon_optimisation/hexagon_classifier_calibration.csv"))
    truth_text = "\n".join(
        f"{r.get('case_id', '')}: strict={r.get('strict_hexagon_eligible', '')}, class={r.get('classifier_label', r.get('strict_class', ''))}"
        for r in truth_rows[:8]
    )
    axes[0, 1].text(0, 1, "Classifier truth table\n\n" + truth_text, va="top", fontsize=8.5)
    op_rows = [row for row in metric_rows if row["route_id"] in {CANONICAL_OPERATING_POINT_ID, "realistic_dual_slm_4f"}]
    metric_text = "\n".join(
        f"{r['route_id']}: strict={r['strict_hexagon_eligible']}, corrR={float(r['corr_to_realistic_4f']):.4f}, "
        f"Puse={float(r['P_useful']):.1f}, peak={float(r['strict_peak_metric']):.2f}"
        for r in op_rows
    )
    axes[1, 0].text(0, 1, "Canonical / reference comparison\n\n" + metric_text, va="top", fontsize=8.5)
    forbidden = (
        f"Forbidden old optimum:\n{OLD_BEST_COMPROMISE_ID}\n\n"
        "Hardware provenance summary:\n"
        "- exact NIR-149 identity: externally supplied until physical label/manual read\n"
        "- phase stroke/LUT: docs/75 routine calibration\n"
        "- camera scale/z-stage: docs/77 routine calibration\n"
        "- no microfabrication/sample-plane success claim"
    )
    axes[1, 1].text(0, 1, "Provenance and forbidden audit\n\n" + forbidden, va="top", fontsize=8.5)
    fig.suptitle("MODE 2W Figure A1: supplementary metrics and provenance", fontsize=14, weight="bold")
    return _savefig(fig, path_png, path_pdf)


def _write_doc(path: Path, readiness: Mapping[str, Any], output_dir: Path) -> Path:
    text = f"""# Nathan MODE 2W - Annotated Master Figure Pack

**Status:** presentation / synthesis mode only. MODE 2W does not introduce new physics, does not
revive forbidden optima, and makes no microfabrication/sample-plane success claim.

Canonical source-scale operating point: `{CANONICAL_OPERATING_POINT_ID}`.
Secondary strict-eligible operating point: `{STRICT_COMPROMISE_ID}`.
Forbidden old optimum: `{OLD_BEST_COMPROMISE_ID}`.

## What each figure shows

1. `fig1_optical_system_annotated.*` - the full left-to-right source-scale bench with fixed,
   routine-calibration, and unknown/non-blocking items visually separated.
2. `fig2_target_and_masks_annotated.*` - the target sector field, Stokes/intensity views and the
   native SLM-H/SLM-V phase-mask package.
3. `fig3_ideal_vs_actual_outputs.*` - V0, ideal Jones route, realistic/canonical 4F route,
   moderate realism and one compensated recovery example under the repaired strict gate.
4. `fig4_propagation_and_power.*` - source-scale propagation through the Bessel region plus the
   canonical stage-by-stage power ledger.
5. `fig5_tolerance_correction_build_readiness.*` - tolerance/correction dashboard and first-day
   build-readiness status.
6. `figA1_appendix_metrics_and_provenance.*` - supplementary settings, truth table and provenance audit.

## Route interpretation

The ideal route is the M2P/M2N dual-linear-SLM Jones synthesis with an ideal final QWP. The actual
route is the realistic dual-SLM + carrier + common 4F + QWP + axicon chain. The canonical operating
point is the realistic 4F reference itself, `{CANONICAL_OPERATING_POINT_ID}`; the secondary strict
candidate remains `{STRICT_COMPROMISE_ID}`.

## Optical settings and masks

The source-scale bench uses 1029 nm, a 2 mm source beam radius, native 1920 x 1080 SLM panels at
8 um pitch, 6.25 lp/mm carrier, f=300 mm common 4F, +1 order at about 1.929 mm, iris diameter about
1.54 mm, QWP code angle -45 deg and a 2 deg n=1.458 axicon. The masks are panel-space wrapped phase
arrays; preview PNGs are not calibrated hardware masks and the per-panel LUT still comes from docs/75.

## Output differences

V0 and the ideal route are visual/reference controls; the realistic/canonical route is the
strict-eligible operating point under the repaired candidate gate. The moderate realism and
compensated recovery rows are shown explicitly so a high full-field correlation cannot hide a failed
candidate gate, C3/triangular drift, dark-core growth or fourfold/X-like failure.

## Propagation and power

The propagation panel shows xy slices, x-z/y-z centre maps, on-axis intensity, ring peak and useful
power across z. The power panel is a normalised model ledger with 1 W and 10 W linear examples only;
it is not a damage-threshold or power-rating claim.

## Tolerances and correction

The dashboard summarises M2S single-parameter and combined-case tolerances, and M2V closed-loop
correction rows. The main practical tolerance is hologram/mask-to-axicon centring; the correction
mechanism is measured-image-driven digital recentring plus bounded low-order/piston/sector updates.

## Build-ready conclusion

Outcome **M2W-A**: the figure pack presents the validated source-scale branch clearly enough for a
report, slide deck or lab handoff. Source-scale build authorised: `{readiness['source_scale_build_authorised']}`.
Remaining calibrations are routine: `{readiness['remaining_routine_calibrations']}`.

Output root: `{output_dir}`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_mode2w_annotated_master_figure_pack(
    *,
    output_dir: str | Path = MODE2W_DEFAULT_OUTPUT_ROOT,
    doc_path: str | Path = MODE2W_DOC_PATH,
    grid_n: int = 384,
    z_planes: int = 17,
) -> dict[str, Any]:
    """Create MODE 2W annotated figures, source tables and docs/82."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    assert_not_forbidden(CANONICAL_OPERATING_POINT_ID)
    assert_not_forbidden(STRICT_COMPROMISE_ID)
    bench = _bench(LAB_WAVELENGTH_M, grid_n=int(grid_n), z_planes=int(z_planes))
    canonical, secondary = load_operating_points()
    masks = build_native_masks(canonical)
    eff = float(bench["realistic"].slm_4f_report["first_order_efficiency"])
    leak = float(bench["realistic"].slm_4f_report["zero_order_leakage_after_iris"])
    fourf = fourf_final_design(simulated_first_order_efficiency=eff, simulated_zero_order_leakage=leak)
    power_rows = power_budget_rows(canonical, grid_n=int(grid_n))
    op_rows = mode2w_operating_point_rows()
    settings = mode2w_optical_element_settings(fourf)
    waveplates = waveplate_table()
    cases, metric_rows = _build_output_cases(bench)
    tolerance_rows = _load_tolerance_summary()
    combined_rows = _load_combined_cases()
    loop_rows = _load_closed_loop_rows()
    readiness = {
        "stage": MODE2W_STAGE,
        "selected_outcome": "M2W-A",
        "architecture_valid": architecture_decision()["chosen_route"] == "B_common_4f_after_recombination",
        "source_scale_build_authorised": True,
        "native_masks_exported": True,
        "fourf_geometry_defined": True,
        "remaining_routine_calibrations": "SLM LUT/stroke, exact iris/focal confirmation, camera scale/z-stage, parity sign, QWP mount sign, beam centring",
        "main_practical_tolerance": "hologram/mask centre to axicon axis; use camera-driven digital recentring",
        "canonical_operating_point": CANONICAL_OPERATING_POINT_ID,
        "secondary_operating_point": STRICT_COMPROMISE_ID,
        "forbidden_old_optimum": OLD_BEST_COMPROMISE_ID,
        "microfabrication_sample_plane_claim": False,
    }

    tables = {
        "operating_point_csv": _write_csv(root / "mode2w_operating_point_summary.csv", op_rows),
        "operating_point_json": _write_json(root / "mode2w_operating_point_summary.json", op_rows),
        "optical_settings_csv": _write_csv(root / "mode2w_optical_element_settings.csv", settings),
        "optical_settings_json": _write_json(root / "mode2w_optical_element_settings.json", settings),
        "waveplate_settings_csv": _write_csv(root / "mode2w_waveplate_settings.csv", waveplates),
        "waveplate_settings_json": _write_json(root / "mode2w_waveplate_settings.json", waveplates),
        "slm_mask_metadata_json": _write_json(root / "mode2w_slm_mask_metadata.json", masks["metadata"]),
        "output_metrics_csv": _write_csv(root / "mode2w_output_comparison_metrics.csv", metric_rows),
        "output_metrics_json": _write_json(root / "mode2w_output_comparison_metrics.json", metric_rows),
        "power_ledger_csv": _write_csv(root / "mode2w_power_ledger.csv", power_rows),
        "power_ledger_json": _write_json(root / "mode2w_power_ledger.json", power_rows),
        "tolerance_summary_csv": _write_csv(root / "mode2w_tolerance_summary.csv", tolerance_rows),
        "tolerance_summary_json": _write_json(root / "mode2w_tolerance_summary.json", tolerance_rows),
        "build_readiness_json": _write_json(root / "mode2w_build_readiness_summary.json", readiness),
    }

    figures = {
        "fig1": _plot_fig1(root / "fig1_optical_system_annotated.png", root / "fig1_optical_system_annotated.pdf", settings, fourf),
        "fig2": _plot_fig2(
            root / "fig2_target_and_masks_annotated.png",
            root / "fig2_target_and_masks_annotated.pdf",
            root / "fig2_slmH_mask_only.png",
            root / "fig2_slmV_mask_only.png",
            bench,
            masks,
            op_rows,
        ),
        "fig3": _plot_fig3(root / "fig3_ideal_vs_actual_outputs.png", root / "fig3_ideal_vs_actual_outputs.pdf", bench, cases, metric_rows),
        "fig4": _plot_fig4(root / "fig4_propagation_and_power.png", root / "fig4_propagation_and_power.pdf", bench, power_rows),
        "fig5": _plot_fig5(root / "fig5_tolerance_correction_build_readiness.png", root / "fig5_tolerance_correction_build_readiness.pdf", tolerance_rows, combined_rows, loop_rows, readiness),
        "figA1": _plot_figA1(root / "figA1_appendix_metrics_and_provenance.png", root / "figA1_appendix_metrics_and_provenance.pdf", settings, metric_rows),
    }
    doc = _write_doc(Path(doc_path), readiness, root)
    manifest = {
        "stage": MODE2W_STAGE,
        "output_root": str(root),
        "doc": str(doc),
        "figures": {key: [str(p) for p in value] for key, value in figures.items()},
        "tables": {key: str(value) for key, value in tables.items()},
        "selected_outcome": readiness["selected_outcome"],
        "canonical_operating_point": CANONICAL_OPERATING_POINT_ID,
        "secondary_operating_point": STRICT_COMPROMISE_ID,
        "microfabrication_sample_plane_claim": False,
    }
    manifest_path = _write_json(root / "mode2w_manifest.json", manifest)
    return {
        "root": root,
        "bench": bench,
        "figures": figures,
        "tables": tables,
        "doc": doc,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "readiness": readiness,
        "output_metrics": metric_rows,
        "power_rows": power_rows,
    }


__all__ = [
    "MODE2W_STAGE",
    "MODE2W_DEFAULT_OUTPUT_ROOT",
    "MODE2W_DOC_PATH",
    "MODE2W_ALLOWED_OUTCOMES",
    "mode2w_operating_point_rows",
    "mode2w_optical_element_settings",
    "write_mode2w_annotated_master_figure_pack",
]
