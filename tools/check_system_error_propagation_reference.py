from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vbb_study.digital_twin.propagation_audit import (
    central_roi_mask,
    compare_intensity_fields,
    scalar_padded_reference,
)
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry
from vbb_study.digital_twin.vortex_system_route import build_system_route
from vbb_study.equations.propagation import (
    angular_spectrum_propagate_bl,
    bandlimit_mask_matsushima,
)
from vbb_study.equations.fields import fft2c


EPS = np.finfo(float).tiny


def _nominal_value(values: tuple[Any, ...]) -> Any:
    finite = [v for v in values if np.isfinite(float(v))]
    if any(np.isclose(float(v), 1.0) for v in finite):
        return min(finite, key=lambda value: abs(float(value) - 1.0))
    nonfinite = [v for v in values if not np.isfinite(float(v))]
    if nonfinite:
        return nonfinite[0]
    return min(values, key=lambda value: abs(float(value)))


def _same_value(a: Any, b: Any) -> bool:
    aa, bb = float(a), float(b)
    if not np.isfinite(aa) or not np.isfinite(bb):
        return (not np.isfinite(aa)) and (not np.isfinite(bb))
    return bool(np.isclose(aa, bb, rtol=0.0, atol=1e-15))


def _audit_values(values: tuple[Any, ...]) -> tuple[Any, ...]:
    nominal = _nominal_value(values)
    selected: list[Any] = [values[0], nominal, values[-1]]
    out: list[Any] = []
    for value in selected:
        if not any(_same_value(value, existing) for existing in out):
            out.append(value)
    return tuple(out)


def _json_float(value: Any) -> float | str:
    v = float(value)
    return v if np.isfinite(v) else str(v)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare all executable beam/SLM/4F error propagation against a padded ASM reference."
    )
    parser.add_argument("--grid-n", type=int, default=256)
    parser.add_argument("--cases", nargs="+", default=["B0", "V1", "V3"])
    parser.add_argument("--z-mm", nargs="+", type=float, default=[20.0, 60.0, 140.0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/propagation_audit/system_error_all_families_reference.json"),
    )
    args = parser.parse_args()

    registry = system_sweep_registry()
    families = [name for name in registry if not name.startswith("axicon_")]
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for case_id in args.cases:
        for family in families:
            spec = registry[family]
            for value in _audit_values(tuple(spec["values"])):
                route = build_system_route(
                    case_id,
                    grid_n=int(args.grid_n),
                    config=spec["builder"](value),
                )
                grid = dict(route["grid"])
                wavelength = float(route["metadata"]["wavelength_m"])
                spectrum = fft2c(route["post_axicon"])
                spectrum_power = float(np.sum(np.abs(spectrum) ** 2))
                roi = central_roi_mask(grid, 1.5e-3)
                for z_mm in args.z_mm:
                    z = float(z_mm) * 1e-3
                    candidate = angular_spectrum_propagate_bl(
                        route["post_axicon"],
                        grid,
                        wavelength,
                        z,
                        n_medium=1.0,
                        bandlimit=True,
                        include_evanescent=True,
                    )
                    reference = scalar_padded_reference(
                        route["post_axicon"],
                        grid,
                        wavelength_m=wavelength,
                        z_m=z,
                        n_medium=1.0,
                        pad_factor=2,
                    )
                    comparison = compare_intensity_fields(
                        candidate,
                        reference,
                        roi_mask=roi,
                        dx_m=float(grid["dx"]),
                    )
                    support = bandlimit_mask_matsushima(
                        grid,
                        wavelength,
                        z,
                        n_medium=1.0,
                    )
                    retained = float(
                        np.sum(np.abs(spectrum[support]) ** 2) / max(spectrum_power, EPS)
                    )
                    row = {
                        "case_id": case_id,
                        "family": family,
                        "value": _json_float(value),
                        "units": str(spec["units"]),
                        "fidelity": str(spec["fidelity"]),
                        "grid_n": int(args.grid_n),
                        "z_m": z,
                        "distance_specific_support_retained_fraction": retained,
                        "intensity_correlation": float(comparison.intensity_correlation),
                        "normalised_relative_l2": float(comparison.normalised_relative_l2),
                        "peak_ratio_candidate_to_reference": float(
                            comparison.peak_ratio_candidate_to_reference
                        ),
                        "roi_power_ratio_candidate_to_reference": float(
                            comparison.roi_power_ratio_candidate_to_reference
                        ),
                    }
                    rows.append(row)
                    if retained < 0.985:
                        failures.append(
                            f"{case_id}/{family}/{value} z={z_mm:g} mm retains only {retained:.6f}"
                        )
                    if comparison.intensity_correlation < 0.985:
                        failures.append(
                            f"{case_id}/{family}/{value} z={z_mm:g} mm correlation {comparison.intensity_correlation:.6f}"
                        )
                    if comparison.normalised_relative_l2 > 0.12:
                        failures.append(
                            f"{case_id}/{family}/{value} z={z_mm:g} mm L2 {comparison.normalised_relative_l2:.6f}"
                        )

    payload = {
        "outcome": "SYSTEM-ERROR-PROPAGATION-PADDED-REFERENCE-AUDIT",
        "hard_pass": not failures,
        "grid_n": int(args.grid_n),
        "cases": list(args.cases),
        "families": families,
        "z_mm": list(args.z_mm),
        "sweep_sampling": "first + nominal + last value per family",
        "candidate": "distance-specific Matsushima BL-ASM on native 10 mm window",
        "reference": "2x spatial zero-padding + unbandlimited ASM, central 3 mm square ROI",
        "thresholds": {
            "minimum_support_retained_fraction": 0.985,
            "minimum_intensity_correlation": 0.985,
            "maximum_normalised_relative_l2": 0.12,
        },
        "rows": rows,
        "hard_failures": failures,
        "claim_boundary": (
            "This validates downstream sampled free-space propagation for the tested error spectra. "
            "It does not validate the physical fidelity or absolute magnitude of the upstream error model."
        ),
        "report_figures_authorised": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit("; ".join(failures[:12]))


if __name__ == "__main__":
    main()
