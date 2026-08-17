"""Carrier-tracked scalar wave route for a rigidly tilted thin axicon.

This route exists to make the *scalar thin-element* oblique-axicon model usable
at angles for which the physical carrier cannot be sampled on the 10 mm FFT
window.  The upstream SLM/4F field is basebanded about its measured spectral
centre, transformed into the axicon frame with the absolute carrier retained as
metadata, multiplied by the axicon transmittance, then transformed back.

The calculation is benchmarkable against the independent two-interface Snell
ray reference in :mod:`vortex_axicon_oblique_reference`.  It is not labelled as
a full refractive-surface/vector solution; that higher-fidelity branch requires
explicit physical axicon angle convention, surface geometry and Fresnel data.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from vbb_study.digital_twin.vortex_axicon_oblique_reference import (
    oblique_refractive_axicon_rays,
)
from vbb_study.digital_twin.vortex_rotated_plane import spectral_centroid_cpm
from vbb_study.digital_twin.vortex_rotated_plane_baseband import (
    rotate_baseband_angular_spectrum,
)
from vbb_study.digital_twin.vortex_system_route import (
    SystemErrorConfig,
    build_system_route,
    physical_axicon_on_own_plane,
)


TWOPI = 2.0 * np.pi


def build_carrier_tracked_oblique_axicon_route(
    case_id: str,
    *,
    grid_n: int,
    config: SystemErrorConfig,
    window_m: float = 10.0e-3,
) -> dict[str, Any]:
    """Build the scalar oblique-axicon route without sampling the oblique carrier."""

    tx, ty = map(float, config.axicon.tilt_rad)
    zero_tilt_axicon = replace(config.axicon, tilt_rad=(0.0, 0.0))
    zero_tilt_config = replace(config, axicon=zero_tilt_axicon)
    base_route = build_system_route(
        case_id,
        grid_n=int(grid_n),
        config=zero_tilt_config,
        window_m=float(window_m),
    )
    if tx == 0.0 and ty == 0.0:
        metadata = dict(base_route["metadata"])
        metadata["axicon_tilt_status"] = "carrier_tracked_identity"
        metadata["oblique_scalar_model"] = "identity_zero_tilt"
        return {**base_route, "metadata": metadata}

    grid = dict(base_route["grid"])
    wavelength = float(base_route["metadata"]["wavelength_m"])
    selected = np.asarray(base_route["post_4f_selected_order"], dtype=np.complex128)
    fsx, fsy = spectral_centroid_cpm(selected, grid)
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    selected_envelope = selected * np.exp(-1j * TWOPI * (fsx * X + fsy * Y))

    local_envelope, to_local = rotate_baseband_angular_spectrum(
        selected_envelope,
        grid,
        wavelength_m=wavelength,
        tilt_x_rad=tx,
        tilt_y_rad=ty,
        source_spectral_center_cpm=(fsx, fsy),
        inverse=False,
    )
    local_center = tuple(map(float, to_local["destination_spectral_center_cpm"]))

    ax_meta0 = base_route["metadata"]["axicon"]
    axicon_t, axicon_meta = physical_axicon_on_own_plane(
        grid,
        wavelength_m=wavelength,
        base_angle_rad=float(ax_meta0["base_angle_rad"]),
        refractive_index=float(ax_meta0["refractive_index"]),
        external_index=float(ax_meta0["external_index"]),
        error=config.axicon,
    )
    post_local_envelope = np.asarray(local_envelope) * np.asarray(axicon_t)

    lab_envelope, to_lab = rotate_baseband_angular_spectrum(
        post_local_envelope,
        grid,
        wavelength_m=wavelength,
        tilt_x_rad=tx,
        tilt_y_rad=ty,
        source_spectral_center_cpm=local_center,
        inverse=True,
    )
    lab_center = tuple(map(float, to_lab["destination_spectral_center_cpm"]))
    post_axicon = np.asarray(lab_envelope) * np.exp(
        1j * TWOPI * (lab_center[0] * X + lab_center[1] * Y)
    )

    ray = oblique_refractive_axicon_rays(
        base_angle_rad=float(axicon_meta["base_angle_rad"]),
        refractive_index=float(axicon_meta["refractive_index"]),
        external_index=float(axicon_meta["external_index"]),
        tilt_x_rad=tx,
        tilt_y_rad=ty,
    )
    metadata = dict(base_route["metadata"])
    metadata.update(
        {
            "axicon": axicon_meta,
            "axicon_rigid_tilt_rad": (tx, ty),
            "axicon_tilt_status": "carrier_tracked_scalar_thin_axicon",
            "oblique_scalar_model": (
                "tilted_plane_baseband_envelope_times_axisymmetric_axicon_transmittance"
            ),
            "lab_to_tilted": to_local,
            "tilted_to_lab": to_lab,
            "input_lab_spectral_center_cpm": [float(fsx), float(fsy)],
            "local_analytic_carrier_cpm": list(local_center),
            "output_lab_spectral_center_cpm": list(lab_center),
            "independent_snell_ray_reference": {
                "cone_radius_mean_direction_sine": float(ray.cone_radius_mean),
                "cone_radius_anisotropy_fraction": float(
                    ray.cone_radius_anisotropy_fraction
                ),
                "second_harmonic_fraction": float(ray.second_harmonic_fraction),
                "mean_outgoing_lab": ray.mean_outgoing_lab.tolist(),
            },
            "report_policy": (
                "scalar oblique-wave sensitivity only; full two-surface refractive/vector "
                "claim remains blocked"
            ),
        }
    )
    return {
        **base_route,
        "field_on_axicon_plane": np.asarray(local_envelope, dtype=np.complex128),
        "post_axicon_local": np.asarray(post_local_envelope, dtype=np.complex128),
        "post_axicon": np.asarray(post_axicon, dtype=np.complex128),
        "metadata": metadata,
    }


__all__ = ["build_carrier_tracked_oblique_axicon_route"]
