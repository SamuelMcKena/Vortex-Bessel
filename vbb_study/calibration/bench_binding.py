"""Bind measured calibration values onto the canonical hardware manifest.

The binder never fabricates a missing measurement.  It starts from the audited
canonical manifest and replaces a value only when the calibration bundle
contains a non-null value.  The source/provenance of every replacement is
preserved so downstream code can distinguish a physical bench prediction from a
sensitivity calculation that still contains assumptions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping

from vbb_study.calibration.schema import CalibrationBundle, source_at, value_at
from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest


CALIBRATED_SOURCES = {"measured", "manufacturer", "derived", "synthetic_measurement", "supplied"}


@dataclass(frozen=True)
class BenchBindingReport:
    manifest: Mapping[str, Any]
    replaced_parameters: tuple[str, ...]
    unresolved_parameters: tuple[str, ...]
    absolute_bench_ready: bool


# path in calibration bundle -> canonical hardware parameter.
# clear aperture is deliberately not mapped because the historical schema name
# did not specify radius versus diameter.  The explicit refractive axicon route
# requires a semantically named physical radius supplied separately.
_BINDINGS = {
    "laser.wavelength_m": "wavelength_m",
    "laser.pulse_energy_J": "input_pulse_energy_J",
    "laser.beam_radius_on_slm_m": "beam_radius_on_slm_m",
    "fourier_filter.focal_length_m": "fourf_focal_length_m",
    "fourier_filter.iris_radius_m": "fourier_iris_radius_m",
    "objective.numerical_aperture": "objective_NA",
    "objective.focal_length_m": "objective_focal_length_m",
    "objective.effective_pupil_radius_m": "objective_pupil_radius_m",
    "relay.magnification": "relay_magnification_to_sample",
    "axicon.base_angle_deg": "axicon_base_angle_deg",
    "axicon.refractive_index": "axicon_refractive_index",
    "transmissions.slm1": "slm_reflectivity",
    "transmissions.objective": "objective_transmission",
    "transmissions.interface": "sample_surface_transmission",
}


def _parameter_rows(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["parameter"]): row for row in manifest["parameters"]}


def bind_calibration_to_manifest(
    bundle: CalibrationBundle,
    *,
    base_manifest: Mapping[str, Any] | None = None,
) -> BenchBindingReport:
    """Return a provenance-preserving hardware manifest for one lab state."""

    manifest = deepcopy(dict(base_manifest or canonical_hardware_manifest()))
    rows = _parameter_rows(manifest)
    replaced: list[str] = []
    for path, parameter in _BINDINGS.items():
        value = value_at(bundle, path)
        if value in (None, ""):
            continue
        if parameter not in rows:
            raise KeyError(f"canonical manifest has no parameter {parameter!r}")
        source = source_at(bundle, path)
        row = rows[parameter]
        row["value"] = value
        row["provenance"] = source if source in {"measured", "manufacturer", "derived"} else "calibration_required"
        row["evidence"] = f"calibration bundle {bundle.calibration_id}: {path}"
        row["status"] = "active" if source in CALIBRATED_SOURCES else "calibration_required"
        row["notes"] = f"Runtime bench binding from {path}; source={source}."
        replaced.append(parameter)

    # The physical carrier is not an inferred fit parameter.  It is the bench
    # command contract: 20 pixels * 8 um = 160 um -> 6.25 lp/mm.
    carrier = rows["carrier_frequency_cpm"]
    carrier["value"] = 6250.0
    carrier["provenance"] = "derived"
    carrier["evidence"] = "docs/78 question 9 + 20-pixel bench blaze confirmation"
    carrier["status"] = "active"
    carrier["notes"] = "20 px period on the 8 um PLUTO panel; physical-bench canonical carrier."

    unresolved = tuple(
        str(row["parameter"])
        for row in manifest["parameters"]
        if str(row.get("status", "")) == "calibration_required"
    )
    manifest["phase"] = "PHASE 2G BENCH BINDING"
    manifest["calibration_id"] = bundle.calibration_id
    manifest["calibration_data_classification"] = bundle.data_classification
    manifest["calibration_required_parameters"] = list(unresolved)
    manifest["absolute_sample_plane_claim_ready"] = len(unresolved) == 0 and not bundle.is_synthetic
    manifest["absolute_energy_claim_ready"] = bool(
        not bundle.is_synthetic
        and all(value_at(bundle, f"transmissions.{name}") not in (None, "") for name in ("slm1", "slm2", "four_f", "objective", "interface"))
        and value_at(bundle, "laser.pulse_energy_J") not in (None, "")
    )
    return BenchBindingReport(
        manifest=manifest,
        replaced_parameters=tuple(replaced),
        unresolved_parameters=unresolved,
        absolute_bench_ready=bool(manifest["absolute_sample_plane_claim_ready"]),
    )


__all__ = ["BenchBindingReport", "bind_calibration_to_manifest"]
