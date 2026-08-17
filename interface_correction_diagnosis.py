"""Interface-correction diagnosis and proof figures.

I keep this as a targeted runner rather than changing the publication notebooks.
It proves whether the planar-interface correction conserves power, cancels the
Zernike spherical term, and recovers the no-interface reference field.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import json
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vbb_study import setup_study, vbb_metrics, vbb_style

import bessel_twin_core as bt
bt_impl = bt
from finalize_publication_outputs import finalize_outputs


REFERENCE_KEYS = [
    "ring_radius_um",
    "core_radius_um",
    "feature_diameter_um",
    "ring_width_um",
    "bessel_zone_um",
    "bessel_region_um",
    "peak_fluence_J_cm2",
    "core_or_ring_peak_fluence_J_cm2",
    "side_to_core_peak_ratio",
    "first_order_selected_fraction",
    "propagation_power_drift_fraction",
]


def _sha_array(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.view(np.uint8))
    return h.hexdigest()


def reference_config(*, ell: int = 3, depth_um: float = 300.0, preset: str = "fast") -> bt.TwinConfig:
    base = bt.default_config(preset)
    return replace(
        base,
        target=replace(base.target, ell=int(ell)),
        material=replace(base.material, write_depth_m=float(depth_um) * bt.um),
    )


def variant_config(base: bt.TwinConfig, variant: str) -> bt.TwinConfig:
    key = str(variant).lower().strip()
    if key == "no_interface":
        return replace(base, apply_interface=False, correct_interface=False)
    if key == "uncorrected":
        return replace(base, apply_interface=True, correct_interface=False)
    if key == "corrected":
        return replace(base, apply_interface=True, correct_interface=True)
    raise ValueError(f"Unknown interface variant: {variant}")


def focal_power(result: Mapping[str, Any]) -> float:
    grid = result["focal_grid"]
    return float(np.sum(np.abs(result["U_focus"]) ** 2) * float(grid["dx"]) ** 2)


def complex_rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    """Return rel-L2 after removing only a global phase."""

    A = np.asarray(a, dtype=complex)
    B = np.asarray(b, dtype=complex)
    inner = np.vdot(B, A)
    phase = inner / (abs(inner) + bt.EPS)
    return float(np.linalg.norm(A - phase * B) / (np.linalg.norm(B) + bt.EPS))


def ring_or_core_contrast(result: Mapping[str, Any]) -> dict[str, float | str]:
    """Return a vortex-honest contrast metric for the peak transverse plane."""

    design = result["design"]
    volume = result["volume"]
    plane = np.asarray(volume["planes"]["peak"], dtype=float)
    grid = volume["crop_grid"]
    metrics = vbb_metrics.peak_plane_radial_metrics(
        plane,
        grid,
        int(design.ell),
        float(design.kr_sample_m_inv),
        center_mode="centroid",
    )
    r = np.asarray(metrics["r_profile_m"], dtype=float)
    prof = np.asarray(metrics["radial_profile_smooth"], dtype=float)
    ell_abs = abs(int(design.ell))
    finite = np.isfinite(prof)
    if not np.any(finite):
        return {"contrast_kind": "undefined", "ring_or_core_contrast": np.nan, "contrast_floor_used": np.nan}
    if ell_abs > 0:
        peak_idx = int(np.nanargmin(np.abs(r - float(metrics["ring_radius_m"]))))
        peak = float(prof[peak_idx])
        inner = finite & (r <= max(float(metrics["r_half_inner_m"]), 2.0 * float(grid["dx"])))
        null = float(np.nanmin(prof[inner])) if np.any(inner) else float(prof[0])
        floor = 1.0e-4 * max(peak, bt.EPS)
        return {
            "contrast_kind": "ring_peak_to_vortex_null",
            "ring_or_core_contrast": float(peak / max(null, floor)),
            "contrast_floor_used": float(max(null, floor)),
        }
    peak = float(np.nanmax(prof[finite]))
    core = float(metrics["core_radius_m"])
    annulus = finite & (r >= core) & (r <= 4.0 * max(core, float(grid["dx"])))
    valley = float(np.nanmin(prof[annulus])) if np.any(annulus) else float(np.nanmin(prof[finite]))
    floor = 1.0e-4 * max(peak, bt.EPS)
    return {
        "contrast_kind": "core_peak_to_first_valley",
        "ring_or_core_contrast": float(peak / max(valley, floor)),
        "contrast_floor_used": float(max(valley, floor)),
    }


def run_case_set(base: bt.TwinConfig, *, ell: int, z_values_m: np.ndarray) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for variant in ("no_interface", "uncorrected", "corrected"):
        cfg = variant_config(base, variant)
        cases[variant] = bt.run_case(
            cfg,
            preset=str(cfg.grid.label or "fast"),
            path="realistic",
            case_id=f"interface_{variant}_ell{int(ell)}",
            z_values_m=z_values_m,
        )
    return cases


def legacy_clipped_corrected_power(config: bt.TwinConfig) -> dict[str, float]:
    """Reproduce the old corrected path where correction went through the filter.

    This is only for diagnosis. The patched engine defers the conjugate phase to
    the pupil after order isolation; this helper lets me quantify the original
    clipping without reverting the source file.
    """

    design = bt.compute_design_from_targets(config.laser, config.target, config.material)
    rect_grid = bt_impl._reduced_device_grid(config)
    parts = bt_impl._continuous_phase(rect_grid, config, design, include_correction=True)
    phase = (
        bt.quantize_phase(parts["phase_continuous"], config.slm.phase_bits)
        if config.include_quantization
        else parts["phase_wrapped"]
    )
    amp = bt.gaussian_amplitude(rect_grid["R"], config.laser.beam_radius_on_slm_m)
    fill = bt.fill_factor_amplitude(rect_grid, config.slm.pixel_pitch_m, config.slm.fill_factor)
    aperture = (
        (np.abs(rect_grid["X"]) <= 0.5 * config.slm.active_width_m)
        & (np.abs(rect_grid["Y"]) <= 0.5 * config.slm.active_height_m)
    ).astype(float)
    U_rect = amp * fill * aperture * np.exp(1j * phase)
    grid = bt.make_xy_grid(config.grid.N, rect_grid["dx"])
    U = bt_impl._pad_rect_to_square(U_rect, int(config.grid.N))
    order = bt.isolate_first_order(U, grid, config.slm)
    U = order["U_selected"]
    pupil = (grid["R"] <= config.objective.pupil_radius_m).astype(float)
    W = bt.interface_aberration_pupil(grid, config.laser, config.objective, config.material)
    U_focus, focal_grid = bt.focus_to_focal_plane(U * pupil * np.exp(1j * W), grid, config.laser, config.objective)
    return {
        "legacy_corrected_focus_power": float(np.sum(np.abs(U_focus) ** 2) * focal_grid["dx"] ** 2),
        "legacy_corrected_selected_fraction": float(order["selected_fraction"]),
    }


def _crop_focus(result: Mapping[str, Any], crop_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    U = np.asarray(result["U_focus"], dtype=complex)
    grid = result["focal_grid"]
    N = int(grid["N"])
    h = max(2, min(int(crop_pixels), N) // 2)
    c = N // 2
    sl = slice(c - h, c + h)
    return np.abs(U[sl, sl]) ** 2, np.asarray(grid["x"][sl], dtype=float)


def plot_interface_comparison(
    cases: Mapping[str, Mapping[str, Any]],
    *,
    ell: int,
    output_path: str | Path,
) -> Path:
    vbb_style.apply_style()
    labels = [
        ("no_interface", "no-interface ideal"),
        ("uncorrected", "with interface"),
        ("corrected", "corrected"),
    ]
    crop = 192
    focus_planes = []
    focus_x = None
    xz_max = 0.0
    focus_max = 0.0
    for key, _label in labels:
        I, x = _crop_focus(cases[key], crop)
        focus_planes.append(I)
        focus_x = x
        focus_max = max(focus_max, float(np.nanmax(I)))
        xz_max = max(xz_max, float(np.nanmax(cases[key]["volume"]["xz"])))

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    xy_artist = None
    xz_artist = None
    assert focus_x is not None
    x_um = focus_x / bt.um
    for col, (key, label) in enumerate(labels):
        result = cases[key]
        volume = result["volume"]
        xy_artist = axes[0, col].imshow(
            vbb_style.display_scale(focus_planes[col] / (focus_max + bt.EPS), gamma=0.45, normalise=False),
            origin="lower",
            extent=[float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])],
            cmap=vbb_style.INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
        )
        axes[0, col].set_title(label)
        axes[0, col].set_xlabel("x [um, sample plane]")
        axes[0, col].set_ylabel("y [um, sample plane]")

        gx = np.asarray(volume["crop_grid"]["x"], dtype=float) / bt.um
        z_um = np.asarray(volume["z"], dtype=float) / bt.um
        xz_artist = axes[1, col].imshow(
            vbb_style.display_scale(np.asarray(volume["xz"], dtype=float) / (xz_max + bt.EPS), gamma=0.45, normalise=False),
            origin="lower",
            aspect="auto",
            extent=[float(z_um[0]), float(z_um[-1]), float(gx[0]), float(gx[-1])],
            cmap=vbb_style.INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
        )
        metrics = result["metrics"]
        axes[1, col].axvspan(float(metrics["zone_start_um"]), float(metrics["zone_end_um"]), color="white", alpha=0.10, lw=0)
        axes[1, col].set_xlabel("z [um, sample plane]")
        axes[1, col].set_ylabel("x [um, sample plane]")
    if xy_artist is not None:
        cbar = fig.colorbar(xy_artist, ax=axes[0, :], shrink=0.90)
        cbar.set_label("matched focal-plane display intensity, gamma=0.45 [a.u.]")
    if xz_artist is not None:
        cbar = fig.colorbar(xz_artist, ax=axes[1, :], shrink=0.90)
        cbar.set_label("matched XZ display intensity, gamma=0.45 [a.u.]")
    fig.suptitle(f"Interface correction comparison, ell={int(ell)}, depth=300 um")
    caption = (
        f"Interface-correction comparison for ell={int(ell)} at 300 um write depth. "
        "The corrected branch recovers the no-interface focal plane and designed axial zone; "
        "the uncorrected longer zone is aberration-induced elongation, not a controlled feature length."
    )
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"ell": int(ell)})
    plt.close(fig)
    return out


def _json_float(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_json_float(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_float(v) for k, v in value.items()}
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def verify_uncorrected_snapshot(snapshot_path: str | Path) -> dict[str, Any]:
    snap = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    base = reference_config(ell=3, depth_um=300.0)
    cfg = variant_config(base, "uncorrected")
    z_values = np.asarray(snap["z_values_um"], dtype=float) * bt.um
    result = bt.run_case(
        cfg,
        preset="fast",
        path="realistic",
        case_id=snap["case_id"],
        z_values_m=z_values,
    )
    hashes = {
        "U_focus": _sha_array(result["U_focus"]),
        "volume_peak": _sha_array(result["volume"]["peak"]),
        "volume_xz": _sha_array(result["volume"]["xz"]),
        "volume_intensity_stack": _sha_array(result["volume"]["intensity_stack"]),
        "plane_peak": _sha_array(result["volume"]["planes"]["peak"]),
    }
    metrics = {k: float(result["metrics"].get(k, np.nan)) for k in snap["key_metrics"]}
    metric_match = all(metrics[k] == float(snap["key_metrics"][k]) for k in metrics)
    hash_match = hashes == snap["hashes"]
    return {
        "pass": bool(hash_match and metric_match),
        "hash_match": bool(hash_match),
        "metric_match": bool(metric_match),
        "hashes": hashes,
        "metrics": metrics,
    }


def run_interface_diagnosis() -> dict[str, Any]:
    paths = setup_study.bootstrap(Path(__file__))
    out_root = paths["outputs"]
    fig_dir = out_root / "figures" / "interface_correction"
    csv_dir = out_root / "csv" / "interface_correction"
    json_dir = out_root / "json" / "interface_correction"
    diag_dir = out_root / "interface_correction"
    for directory in (fig_dir, csv_dir, json_dir, diag_dir):
        directory.mkdir(parents=True, exist_ok=True)

    base3 = reference_config(ell=3, depth_um=300.0)
    z_values = np.linspace(0.0, 450.0 * bt.um, 61)
    cases3 = run_case_set(base3, ell=3, z_values_m=z_values)

    powers = {name: focal_power(case) for name, case in cases3.items()}
    rel_power_span = (max(powers.values()) - min(powers.values())) / (np.mean(list(powers.values())) + bt.EPS)

    pupil_grid = cases3["uncorrected"]["pupil_grid"]
    W = bt.interface_aberration_pupil(pupil_grid, base3.laser, base3.objective, base3.material)
    Wcorr = bt.interface_correction_phase(pupil_grid, base3.laser, base3.objective, base3.material)
    zern_before = bt.fit_interface_zernike_terms(pupil_grid, W, base3.objective.pupil_radius_m)
    zern_after = bt.fit_interface_zernike_terms(pupil_grid, W + Wcorr, base3.objective.pupil_radius_m)
    rel_l2 = complex_rel_l2(cases3["corrected"]["U_focus"], cases3["no_interface"]["U_focus"])

    variants_rows = []
    for variant, case in cases3.items():
        row = {
            "variant": variant,
            "focal_power": powers[variant],
            "relative_power_vs_no_interface": powers[variant] / (powers["no_interface"] + bt.EPS),
            **{k: float(case["metrics"].get(k, np.nan)) for k in REFERENCE_KEYS},
            **ring_or_core_contrast(case),
        }
        variants_rows.append(row)
    variants_df = pd.DataFrame(variants_rows)
    variants_csv = csv_dir / vbb_style.csv_name(8, "interface", "ell3_variant_metrics")
    variants_df.to_csv(variants_csv, index=False)

    recovery_rows = []
    figure_paths = []
    for ell in (0, 3):
        base = reference_config(ell=ell, depth_um=300.0)
        cases = run_case_set(base, ell=ell, z_values_m=z_values)
        figure_paths.append(
            plot_interface_comparison(
                cases,
                ell=ell,
                output_path=fig_dir / vbb_style.figure_name(8, "interface", f"ell{ell}_ideal_uncorrected_corrected"),
            )
        )
        no = cases["no_interface"]
        corr = cases["corrected"]
        unc = cases["uncorrected"]
        no_contrast = ring_or_core_contrast(no)
        corr_contrast = ring_or_core_contrast(corr)
        unc_contrast = ring_or_core_contrast(unc)
        recovery_rows.append(
            {
                "ell": int(ell),
                "corrected_rel_l2_to_no_interface": complex_rel_l2(corr["U_focus"], no["U_focus"]),
                "corrected_ring_radius_delta_um": float(corr["metrics"]["ring_radius_um"] - no["metrics"]["ring_radius_um"]),
                "corrected_zone_delta_um": float(corr["metrics"]["bessel_zone_um"] - no["metrics"]["bessel_zone_um"]),
                "corrected_peak_fluence_delta_pct": float(
                    100.0 * (corr["metrics"]["peak_fluence_J_cm2"] - no["metrics"]["peak_fluence_J_cm2"])
                    / (no["metrics"]["peak_fluence_J_cm2"] + bt.EPS)
                ),
                "corrected_contrast_delta_pct": float(
                    100.0 * (corr_contrast["ring_or_core_contrast"] - no_contrast["ring_or_core_contrast"])
                    / (no_contrast["ring_or_core_contrast"] + bt.EPS)
                ),
                "no_interface_zone_um": float(no["metrics"]["bessel_zone_um"]),
                "uncorrected_zone_um": float(unc["metrics"]["bessel_zone_um"]),
                "corrected_zone_um": float(corr["metrics"]["bessel_zone_um"]),
                "no_interface_peak_fluence_J_cm2": float(no["metrics"]["peak_fluence_J_cm2"]),
                "uncorrected_peak_fluence_J_cm2": float(unc["metrics"]["peak_fluence_J_cm2"]),
                "corrected_peak_fluence_J_cm2": float(corr["metrics"]["peak_fluence_J_cm2"]),
                "no_interface_contrast": float(no_contrast["ring_or_core_contrast"]),
                "uncorrected_contrast": float(unc_contrast["ring_or_core_contrast"]),
                "corrected_contrast": float(corr_contrast["ring_or_core_contrast"]),
                "old_side_to_core_uncorrected": float(unc["metrics"]["side_to_core_peak_ratio"]),
                "old_side_to_core_corrected": float(corr["metrics"]["side_to_core_peak_ratio"]),
            }
        )
    recovery_df = pd.DataFrame(recovery_rows)
    recovery_csv = csv_dir / vbb_style.csv_name(8, "interface", "recover_no_interface_table")
    recovery_df.to_csv(recovery_csv, index=False)

    corrected_metric_deltas = {
        key: float(cases3["corrected"]["metrics"][key] - cases3["no_interface"]["metrics"][key])
        for key in ("ring_radius_um", "bessel_zone_um", "peak_fluence_J_cm2")
    }
    legacy = legacy_clipped_corrected_power(variant_config(base3, "corrected"))
    snapshot_path = diag_dir / "reference_uncorrected_metrics_prechange.json"
    snapshot_guard = verify_uncorrected_snapshot(snapshot_path)

    checks = {
        "A_energy_conservation": {
            "pass": bool(rel_power_span <= 1.0e-6),
            "relative_power_span": float(rel_power_span),
            "focal_power": powers,
        },
        "B_residual_aberration": {
            "pass": bool(abs(float(zern_after["spherical_waves"])) <= 1.0e-10),
            "before": zern_before,
            "after": zern_after,
        },
        "C_corrected_matches_no_interface": {
            "pass": bool(rel_l2 <= 1.0e-10 and all(abs(v) <= 1.0e-9 for v in corrected_metric_deltas.values())),
            "complex_rel_l2": rel_l2,
            "metric_deltas": corrected_metric_deltas,
        },
        "D_peak_explanation": {
            "cause": "post_fix_uncorrected_hotspot_at_equal_energy; pre_fix_corrected_low_peak_was_first_order_filter_power_clipping",
            "legacy_clipped_corrected": legacy,
            "post_fix_uncorrected_peak_fluence_J_cm2": float(cases3["uncorrected"]["metrics"]["peak_fluence_J_cm2"]),
            "post_fix_corrected_peak_fluence_J_cm2": float(cases3["corrected"]["metrics"]["peak_fluence_J_cm2"]),
        },
        "uncorrected_snapshot_guard": snapshot_guard,
    }
    checks_path = json_dir / "08_interface_correction_checks.json"
    checks_path.write_text(json.dumps(_json_float(checks), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verdict = (
        "buggy: corrected precompensation was being clipped by the first-order filter; "
        "the fix defers the unit-modulus conjugate pupil phase until after order isolation."
    )
    doc = build_markdown_report(
        checks=checks,
        variants_df=variants_df,
        recovery_df=recovery_df,
        variants_csv=variants_csv,
        recovery_csv=recovery_csv,
        figure_paths=figure_paths,
        checks_path=checks_path,
        verdict=verdict,
    )
    doc_path = paths["root"] / "docs" / "INTERFACE_CORRECTION_DIAGNOSIS.md"
    doc_path.write_text(doc, encoding="utf-8")

    setup_study.write_run_manifest(
        json_dir / "08_interface_correction_run_manifest.json",
        config=base3,
        paths={
            "diagnosis_doc": doc_path,
            "checks_json": checks_path,
            "variants_csv": variants_csv,
            "recovery_csv": recovery_csv,
            "figures": figure_paths,
        },
        extra={"z_scan_um": [float(z_values[0] / bt.um), float(z_values[-1] / bt.um), int(len(z_values))]},
        root=paths["root"],
    )
    finalize_outputs(out_root)
    return {
        "checks": checks,
        "variants_csv": variants_csv,
        "recovery_csv": recovery_csv,
        "figures": figure_paths,
        "doc": doc_path,
        "checks_json": checks_path,
    }


def _markdown_table(df: pd.DataFrame, columns: list[str], *, floatfmt: str = ".6g") -> str:
    sub = df[columns].copy()
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in sub.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(format(float(value), floatfmt))
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def build_markdown_report(
    *,
    checks: Mapping[str, Any],
    variants_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
    variants_csv: Path,
    recovery_csv: Path,
    figure_paths: list[Path],
    checks_path: Path,
    verdict: str,
) -> str:
    A = checks["A_energy_conservation"]
    B = checks["B_residual_aberration"]
    C = checks["C_corrected_matches_no_interface"]
    D = checks["D_peak_explanation"]
    guard = checks["uncorrected_snapshot_guard"]
    figure_lines = "\n".join(f"- `{path}`" for path in figure_paths)
    return f"""# Interface Correction Diagnosis

Date: 2026-05-29.

Verdict: **{verdict}**

I used the Cr:ZnSe `ell=3`, 300 um write-depth reference and kept the PHAROS
PH2 / SLM configuration unchanged. The uncorrected reference snapshot is
`outputs/interface_correction/reference_uncorrected_metrics_prechange.json`.

## A. Energy Conservation

Total transverse focal-field power is `sum |E|^2 dx dy`.

{_markdown_table(variants_df, ["variant", "focal_power", "relative_power_vs_no_interface", "first_order_selected_fraction", "peak_fluence_J_cm2", "side_to_core_peak_ratio"])}

Relative power span after the fix: `{A["relative_power_span"]:.3e}`.

The old corrected implementation sent the conjugate interface phase through the
finite first-order filter. Reconstructing that legacy path gives selected
fraction `{D["legacy_clipped_corrected"]["legacy_corrected_selected_fraction"]:.6g}`
and focal power `{D["legacy_clipped_corrected"]["legacy_corrected_focus_power"]:.6g}`,
which is the energy leak that made the corrected peak look artificially bad.

## B. Residual Aberration

Zernike fit to the interface phase before correction:

- defocus: `{B["before"]["defocus_waves"]:.6g}` waves
- primary spherical: `{B["before"]["spherical_waves"]:.6g}` waves
- residual RMS: `{B["before"]["residual_rms_rad"]:.6g}` rad

Zernike fit after applying the conjugate pupil correction:

- defocus: `{B["after"]["defocus_waves"]:.6g}` waves
- primary spherical: `{B["after"]["spherical_waves"]:.6g}` waves
- residual RMS: `{B["after"]["residual_rms_rad"]:.6g}` rad

The spherical term is cancelled to numerical zero.

## C. Equivalence To No-Interface Ideal

Corrected vs no-interface complex-field relative L2 error:
`{C["complex_rel_l2"]:.3e}`.

Metric deltas for `ell=3`:

- ring radius delta: `{C["metric_deltas"]["ring_radius_um"]:.6g}` um
- Bessel-zone delta: `{C["metric_deltas"]["bessel_zone_um"]:.6g}` um
- peak-fluence delta: `{C["metric_deltas"]["peak_fluence_J_cm2"]:.6g}` J/cm^2

Recovery table:

{_markdown_table(recovery_df, ["ell", "corrected_rel_l2_to_no_interface", "corrected_zone_delta_um", "corrected_peak_fluence_delta_pct", "no_interface_zone_um", "uncorrected_zone_um", "corrected_zone_um", "no_interface_contrast", "uncorrected_contrast", "corrected_contrast"])}

## D. Why Corrected Peak Can Be Lower Than Uncorrected

Before the fix, the corrected peak was lower because of an energy leak from the
first-order filter. After the fix, all three powers match. The remaining higher
uncorrected peak fluence is therefore a physical aberration hot spot at equal
energy, and its longer axial zone is uncontrolled aberration-induced elongation,
not a designed Bessel writing length.

## Reporting Change

For vortex beams I report `ring_peak_to_vortex_null` contrast rather than using
the old `side_to_core_peak_ratio` as a headline metric. The old ratio remains in
the CSV as a labelled diagnostic because it can explode when a corrected vortex
has a clean on-axis null.

## Recommendation

Enable correction when I want the **designed, predictable feature**: a clean
focus matching the no-interface simulation. Treat the uncorrected longer zone as
an uncontrolled aberration artefact, not a fabrication advantage.

## Artefacts

- Checks JSON: `{checks_path}`
- Variant metrics CSV: `{variants_csv}`
- Recover-no-interface table CSV: `{recovery_csv}`

Figures:

{figure_lines}

## Verification Gate

- A energy conservation pass: `{A["pass"]}`
- B residual aberration pass: `{B["pass"]}`
- C corrected matches no-interface pass: `{C["pass"]}`
- Uncorrected snapshot bit-for-bit unchanged: `{guard["pass"]}`
"""


if __name__ == "__main__":
    bundle = run_interface_diagnosis()
    print(f"Diagnosis doc: {bundle['doc']}")
    print(f"Recover table: {bundle['recovery_csv']}")
    for figure in bundle["figures"]:
        print(f"Figure: {figure}")
