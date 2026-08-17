from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vbb_study.digital_twin.nathan_mode2w_fix_sequential_master import (
    _bench_from_config,
    _source_config,
)
from vbb_study.digital_twin.nathan_mode2y_continuous_vs_averaged import (
    _after_axicon,
    _intensity_from_prepared,
    _prepare_projected_spectrum,
    _realistic_common_4f_field,
    build_mode2y_input_fields,
)
from vbb_study.digital_twin.phase2b_visual_cases import _scalar_seed
from vbb_study.digital_twin.propagation_audit import (
    central_roi_mask,
    compare_intensity_arrays,
    compare_intensity_fields,
    scalar_padded_reference,
    vector_padded_reference_from_projected_spectra,
)
from vbb_study.digital_twin.vortex_continuous_propagation import (
    adjacent_row_continuity_metrics,
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny


def _comparison_dict(value: Any) -> dict[str, float]:
    return {
        "intensity_correlation": float(value.intensity_correlation),
        "normalised_relative_l2": float(value.normalised_relative_l2),
        "peak_ratio_candidate_to_reference": float(value.peak_ratio_candidate_to_reference),
        "roi_power_ratio_candidate_to_reference": float(value.roi_power_ratio_candidate_to_reference),
    }


def _scalar_audit(grid_n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    continuity_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    z_checks = (20e-3, 60e-3, 120e-3, 200e-3)
    z_stack = np.arange(0.0, 200e-3 + 2e-3, 2e-3)

    for case_id, ell in (("G0", 0), ("B0", 0), ("V1", 1), ("V3", 3)):
        field0, grid, meta = _scalar_seed(case_id, ell, grid_n=int(grid_n))
        wavelength = float(meta["wavelength_m"])
        fixed = build_fixed_support_spectrum(
            field0,
            grid,
            wavelength_m=wavelength,
            z_max_m=max(z_stack),
            minimum_retained_spectral_power=0.99,
        )
        roi = central_roi_mask(grid, 1.5e-3 if case_id != "G0" else 2.5e-3)

        legacy_xz = np.empty((z_stack.size, int(grid["N"])), dtype=float)
        fixed_xz = np.empty_like(legacy_xz)
        legacy_yz = np.empty_like(legacy_xz)
        fixed_yz = np.empty_like(legacy_xz)
        mid = int(grid["N"]) // 2
        for iz, z in enumerate(z_stack):
            legacy = field0 if np.isclose(z, 0.0) else angular_spectrum_propagate_bl(
                field0,
                dict(grid),
                wavelength,
                float(z),
                n_medium=1.0,
                bandlimit=True,
                include_evanescent=True,
            )
            fixed_field = native_field_at_z(fixed, float(z))
            legacy_i = np.abs(legacy) ** 2
            fixed_i = np.abs(fixed_field) ** 2
            legacy_xz[iz] = legacy_i[mid, :]
            legacy_yz[iz] = legacy_i[:, mid]
            fixed_xz[iz] = fixed_i[mid, :]
            fixed_yz[iz] = fixed_i[:, mid]

        continuity_rows.append(
            {
                "family": "phase2b_scalar",
                "case_id": case_id,
                "grid_n": int(grid_n),
                "retained_fixed_support_fraction": float(fixed.retained_spectral_power_fraction),
                "legacy_xz_max_over_median": adjacent_row_continuity_metrics(legacy_xz)["adjacent_row_rms_change_max_over_median"],
                "legacy_yz_max_over_median": adjacent_row_continuity_metrics(legacy_yz)["adjacent_row_rms_change_max_over_median"],
                "fixed_xz_max_over_median": adjacent_row_continuity_metrics(fixed_xz)["adjacent_row_rms_change_max_over_median"],
                "fixed_yz_max_over_median": adjacent_row_continuity_metrics(fixed_yz)["adjacent_row_rms_change_max_over_median"],
            }
        )

        for z in z_checks:
            reference = scalar_padded_reference(
                field0,
                grid,
                wavelength_m=wavelength,
                z_m=float(z),
                pad_factor=2,
            )
            legacy = angular_spectrum_propagate_bl(
                field0,
                dict(grid),
                wavelength,
                float(z),
                n_medium=1.0,
                bandlimit=True,
                include_evanescent=True,
            )
            candidate = native_field_at_z(fixed, float(z))
            legacy_cmp = compare_intensity_fields(
                legacy,
                reference,
                roi_mask=roi,
                dx_m=float(grid["dx"]),
            )
            fixed_cmp = compare_intensity_fields(
                candidate,
                reference,
                roi_mask=roi,
                dx_m=float(grid["dx"]),
            )
            row = {
                "family": "phase2b_scalar",
                "case_id": case_id,
                "z_m": float(z),
                "grid_n": int(grid_n),
                "retained_fixed_support_fraction": float(fixed.retained_spectral_power_fraction),
                **{f"legacy_{k}": v for k, v in _comparison_dict(legacy_cmp).items()},
                **{f"fixed_{k}": v for k, v in _comparison_dict(fixed_cmp).items()},
            }
            rows.append(row)
            if fixed_cmp.intensity_correlation < 0.985 or fixed_cmp.normalised_relative_l2 > 0.20:
                failures.append(
                    f"Phase2B scalar {case_id} z={z*1e3:.0f} mm fixed-support vs padded reference failed: "
                    f"corr={fixed_cmp.intensity_correlation:.6f}, L2={fixed_cmp.normalised_relative_l2:.6f}"
                )

    return rows, continuity_rows, failures


def _vector_audit(grid_n: int) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    cfg = _source_config(grid_n=int(grid_n), z_planes=3, z_start_m=59e-3, z_end_m=61e-3)
    bench = _bench_from_config(cfg)
    data = bench["data"]
    inputs = build_mode2y_input_fields(data)
    amplitude = np.asarray(data["A"], dtype=float)
    roi = central_roi_mask(data["grid"], 1.5e-3)
    wavelength = float(data["config"].wavelength_m)

    for label, alpha in (
        ("continuous", inputs.continuous_alpha_rad),
        ("sector_averaged", inputs.averaged_alpha_rad),
    ):
        prefield, _ = _realistic_common_4f_field(
            amplitude,
            alpha,
            data,
            carrier_lpmm=6.25,
            iris_radius_frac=0.40,
        )
        after, _ = _after_axicon(prefield, data)
        prepared = _prepare_projected_spectrum(after)
        for z in (30e-3, 60e-3, 150e-3, 200e-3):
            candidate = _intensity_from_prepared(prepared, float(z))
            reference = vector_padded_reference_from_projected_spectra(
                prepared,
                data["grid"],
                wavelength_m=wavelength,
                z_m=float(z),
                n_medium=1.0,
                pad_factor=2,
            )
            cmp = compare_intensity_arrays(
                candidate,
                reference,
                roi_mask=roi,
                dx_m=float(data["grid"]["dx"]),
            )
            rows.append(
                {
                    "family": "phase2b_vector_hex",
                    "input_model": label,
                    "z_m": float(z),
                    "grid_n": int(grid_n),
                    **_comparison_dict(cmp),
                }
            )
            if cmp.intensity_correlation < 0.975 or cmp.normalised_relative_l2 > 0.25:
                failures.append(
                    f"Phase2B vector {label} z={z*1e3:.0f} mm vs padded reference failed: "
                    f"corr={cmp.intensity_correlation:.6f}, L2={cmp.normalised_relative_l2:.6f}"
                )
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the propagation paths behind Phase 2B report figures.")
    parser.add_argument("--scalar-grid-n", type=int, default=512)
    parser.add_argument("--vector-grid-n", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/propagation_audit/phase2b_propagation_audit.json"),
    )
    args = parser.parse_args()

    scalar_rows, continuity_rows, scalar_failures = _scalar_audit(int(args.scalar_grid_n))
    vector_rows, vector_failures = _vector_audit(int(args.vector_grid_n))
    failures = [*scalar_failures, *vector_failures]
    payload = {
        "outcome": "PHASE2B-PROPAGATION-AUDIT",
        "hard_pass": not failures,
        "scalar_candidate": "single max-z Matsushima support, fixed for complete z stack",
        "scalar_legacy": "independent per-plane Matsushima support used by frozen Phase2B scalar maps",
        "scalar_reference": "2x spatial zero-padding + unbandlimited ASM, central ROI",
        "vector_candidate": "projected three-component angular spectrum on native fixed grid",
        "vector_reference": "reconstructed projected vector field, 2x spatial padding, transverse re-projection on padded k-grid, unbandlimited propagation",
        "scalar_rows": scalar_rows,
        "scalar_continuity_rows": continuity_rows,
        "vector_rows": vector_rows,
        "hard_failures": failures,
        "interpretation_policy": (
            "A passing numerical audit does not validate upstream optical assumptions. It only establishes that the displayed propagation is not dominated by the tested sampling/window/support artefacts."
        ),
        "report_figures_authorised": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
