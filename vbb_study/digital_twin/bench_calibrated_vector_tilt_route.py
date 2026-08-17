"""Calibration-gated segmented-vector bench route with physical axicon tilt.

This module deliberately wraps the already-validated Phase-2G serial dual-SLM
vector generator rather than duplicating SLM/HWP/QWP/4F physics.  Phase 2G is run
through the selected 20-pixel diffraction order with zero axicon tilt; its
post-4F spatial Ex/Ey field is then passed into the Phase-2H common-eikonal
physical two-surface axicon solver.

Absolute tilted-axicon use is refused unless the calibration bundle explicitly
contains the geometrical quantities that define the implemented physical surface:

- base angle with convention ``base_angle_from_flat_face``;
- clear radius;
- centre thickness;
- refractive index;
- verified flat-face-upstream orientation.

The current Phase-2H surface geometry is flat entrance -> conical exit.  A
cone-first laboratory orientation is *not* silently reinterpreted.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

from vbb_study.calibration.bench_binding import bind_calibration_to_manifest
from vbb_study.calibration.schema import source_at, value_at
from vbb_study.digital_twin.bench_calibrated_vector_route import (
    BenchCalibratedVectorInputs,
    build_calibrated_segmented_vector_route,
)
from vbb_study.digital_twin.nathan_vector_hexagon import NathanHexagonConfig
from vbb_study.digital_twin.vector_refractive_axicon_eikonal import (
    VectorRefractiveAxiconResult,
    build_tilted_vector_refractive_axicon_field,
)
from vbb_study.digital_twin.vortex_refractive_axicon import RefractiveAxiconGeometry
from vbb_study.digital_twin.phase2a_contracts import hardware_value


_UNCALIBRATED_SOURCES = {
    "",
    "missing",
    "unspecified",
    "assumed",
    "calibration_required",
    "template",
    "unknown",
}


def _calibrated_value(bundle, path: str) -> Any:
    value = value_at(bundle, path)
    source = str(source_at(bundle, path)).strip().lower()
    if value in (None, ""):
        raise ValueError(f"Phase 2H calibrated vector tilt requires {path}")
    if source in _UNCALIBRATED_SOURCES:
        raise ValueError(
            f"Phase 2H calibrated vector tilt requires calibrated provenance for {path}; "
            f"source={source!r}"
        )
    return value


def refractive_axicon_geometry_from_calibration(calibrated: BenchCalibratedVectorInputs) -> RefractiveAxiconGeometry:
    """Resolve the exact implemented plano-conical geometry from calibration."""

    bundle = calibrated.calibration_bundle
    convention = value_at(bundle, "axicon.angle_convention")
    if convention != "base_angle_from_flat_face":
        raise ValueError(
            "Phase 2H currently requires axicon.angle_convention='base_angle_from_flat_face'; "
            "apex/deviation/vendor conventions must be converted explicitly, never guessed"
        )
    orientation = value_at(bundle, "axicon.flat_face_upstream_verified")
    if orientation is not True:
        raise ValueError(
            "Phase 2H current surface solver requires axicon.flat_face_upstream_verified=True; "
            "cone-first orientation needs its own physical surface order"
        )

    base_angle_deg = float(_calibrated_value(bundle, "axicon.base_angle_deg"))
    clear_radius_m = float(_calibrated_value(bundle, "axicon.clear_radius_m"))
    centre_thickness_m = float(_calibrated_value(bundle, "axicon.centre_thickness_m"))
    refractive_index = float(_calibrated_value(bundle, "axicon.refractive_index"))

    binding = bind_calibration_to_manifest(bundle)
    external_index = float(hardware_value(binding.manifest, "axicon_external_medium_index"))
    geometry = RefractiveAxiconGeometry(
        base_angle_rad=math.radians(base_angle_deg),
        clear_radius_m=clear_radius_m,
        centre_thickness_m=centre_thickness_m,
        refractive_index=refractive_index,
        external_index=external_index,
    )
    geometry.validate()
    return geometry


def build_calibrated_segmented_vector_tilt_route(
    config: NathanHexagonConfig,
    *,
    calibrated: BenchCalibratedVectorInputs,
    grid_n: int | None = None,
    vector_axicon_output_n: int | None = None,
    vector_axicon_output_window_m: float | None = None,
    reference_gap_m: float = 0.25e-3,
    apex_exclusion_radius_m: float = 0.0,
) -> dict[str, Any]:
    """Build SLM1->HWP->SLM2->QWP->4F->tilted real axicon as one vector route."""

    geometry = refractive_axicon_geometry_from_calibration(calibrated)
    tx, ty = map(float, calibrated.axicon_tilt_rad)

    # Reuse the Phase-2G calibrated generator/4F path, but intentionally prevent
    # its normal-incidence axicon stage from seeing a non-zero tilt.  We consume
    # only its post-4F selected-order field and replace the downstream axicon
    # field with the Phase-2H physical result.
    upstream_inputs = replace(calibrated, axicon_tilt_rad=(0.0, 0.0))
    route = build_calibrated_segmented_vector_route(
        config,
        calibrated=upstream_inputs,
        grid_n=grid_n,
    )
    post_4f = route["post_4f_selected_order"]
    vector_result: VectorRefractiveAxiconResult = build_tilted_vector_refractive_axicon_field(
        post_4f,
        geometry=geometry,
        tilt_x_rad=tx,
        tilt_y_rad=ty,
        axicon_decentre_m=calibrated.axicon_decentre_m,
        reference_gap_m=float(reference_gap_m),
        output_n=vector_axicon_output_n,
        output_window_m=vector_axicon_output_window_m,
        apex_exclusion_radius_m=float(apex_exclusion_radius_m),
    )

    metadata = dict(route["metadata"])
    metadata.update(
        {
            "axicon_model": "phase2h_common_eikonal_two_surface_vector_refractive",
            "axicon_tilt_rad": [tx, ty],
            "axicon_surface_order": "flat_entrance_then_conical_exit",
            "axicon_angle_convention": "base_angle_from_flat_face",
            "axicon_geometry_calibration_sources": {
                path: source_at(calibrated.calibration_bundle, path)
                for path in (
                    "axicon.base_angle_deg",
                    "axicon.clear_radius_m",
                    "axicon.centre_thickness_m",
                    "axicon.refractive_index",
                )
            },
            "vector_tilt_solver_metadata": dict(vector_result.metadata),
            "vector_rigid_axicon_tilt_supported": True,
            "vector_rigid_axicon_tilt_claim_scope": (
                "calibrated flat-first macroscopic common-eikonal vector Snell/Fresnel model; "
                "subject to solver sampling/eikonal/flux gates"
            ),
        }
    )
    return {
        **route,
        "post_axicon": vector_result.field,
        "vector_refractive_axicon_result": vector_result,
        "metadata": metadata,
    }


__all__ = [
    "build_calibrated_segmented_vector_tilt_route",
    "refractive_axicon_geometry_from_calibration",
]
