"""Pure configuration builders and inverse-design helpers."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import scipy.special as sp

from vbb_study import vbb_planes
from vbb_study.config import (
    BeamDesign,
    BeamTarget,
    EPS,
    GridConfig,
    LaserConfig,
    MaterialConfig,
    OpticalMappingMode,
    SimulationPreset,
    TwinConfig,
    um,
)
from vbb_study.equations.objective_pupil import objective_map_from_design_inputs


J0_FIRST_ZERO = float(sp.jn_zeros(0, 1)[0])


def get_preset(name: str = "fast") -> SimulationPreset:
    """Return a deterministic grid preset."""

    key = str(name).lower().strip()
    if key == "paper":
        return SimulationPreset(
            name="paper",
            grid=GridConfig(
                N=2048,
                device_downsample=1,
                axial_range_m=360.0 * um,
                axial_points=181,
                axial_target_factor=3.0,
                crop_pixels=512,
                coarse_scan_points=41,
                ideal_N=1024,
                ideal_dx_m=0.18 * um,
                label="paper",
            ),
        )
    if key in {"publication", "study"}:
        return SimulationPreset(
            name="publication",
            grid=GridConfig(
                N=1024,
                device_downsample=2,
                axial_range_m=300.0 * um,
                axial_points=91,
                axial_target_factor=2.6,
                crop_pixels=256,
                coarse_scan_points=43,
                ideal_N=768,
                ideal_dx_m=0.20 * um,
                label="publication",
            ),
        )
    if key == "balanced":
        return SimulationPreset(
            name="balanced",
            grid=GridConfig(
                N=1024,
                device_downsample=2,
                axial_range_m=240.0 * um,
                axial_points=81,
                axial_target_factor=2.2,
                crop_pixels=288,
                coarse_scan_points=31,
                ideal_N=768,
                ideal_dx_m=0.22 * um,
                label="balanced",
            ),
        )
    return SimulationPreset(name="fast", grid=GridConfig())


def default_config(preset: str = "fast") -> TwinConfig:
    """Build the default Cr:ZnSe PHAROS+SLM twin configuration."""

    return TwinConfig(grid=get_preset(preset).grid)


def axial_scan_values(config: TwinConfig, design: BeamDesign, *, z_anchor_m: float = 0.0) -> np.ndarray:
    """Return the forward z samples used for Bessel-region measurements.

    The scan is deliberately longer than the requested design length so a
    measured half-maximum interval cannot be truncated by the numerical
    boundary and mistaken for a physical Bessel-zone length.
    """

    span = float(config.grid.axial_range_m)
    target = float(design.target_bessel_length_m)
    target_factor = max(1.0, float(getattr(config.grid, "axial_target_factor", 1.8)))
    z_max = max(2.0 * span, float(z_anchor_m) + span, target_factor * target)
    return np.linspace(0.0, z_max, int(config.grid.axial_points))


def compute_design_from_targets(
    laser: LaserConfig,
    target: BeamTarget,
    material: MaterialConfig,
    beam_radius_on_slm_m: Optional[float] = None,
    *,
    mapping_mode: OpticalMappingMode = "target_matched_inverse_design",
    fixed_objective_map: Optional[vbb_planes.ObjectiveMap] = None,
) -> BeamDesign:
    """Inverse-design SLM cone strength from a declared target scale and length.

    ``target_core_diameter_m`` is the equivalent charge-zero first-null
    diameter.  The exact first positive zero of ``J_0`` is used; the old
    repository shorthand ``2.405`` has been removed so the reference and its
    inverse are mathematically identical.

    The requested Bessel length is interpreted as the standard geometrical
    overlap/reference length ``z_ref = w0*k/k_r``.  It is not silently equated
    with a numerical intensity FWHM.  Under ``fixed_physical_optics`` the
    hardware mapping fixes ``w0`` and therefore the predicted reference length;
    under ``target_matched_inverse_design`` the target is used to infer the
    required waist/mapping and is a feasibility design rather than a measured
    bench prediction.
    """

    D = max(float(target.target_core_diameter_m), EPS)
    L = max(float(target.target_bessel_length_m), EPS)
    k_medium = laser.k0 * float(material.refractive_index)

    # D = 2*j_0,1/k_r exactly for the equivalent ell=0 first-null diameter.
    kr_sample = 2.0 * J0_FIRST_ZERO / D
    required_w0_sample = L * kr_sample / max(k_medium, EPS)
    mode = str(mapping_mode).lower().strip()
    if mode == "target_matched_inverse_design":
        objective_map = objective_map_from_design_inputs(
            laser,
            target,
            material,
            beam_radius_on_slm_m=beam_radius_on_slm_m,
        )
        w0_sample = required_w0_sample
    elif mode == "fixed_physical_optics":
        if fixed_objective_map is None:
            raise ValueError("fixed_physical_optics requires an explicit fixed_objective_map")
        objective_map = fixed_objective_map
        w_slm = float(
            laser.beam_radius_on_slm_m
            if beam_radius_on_slm_m is None
            else beam_radius_on_slm_m
        )
        w0_sample = objective_map.pre_to_sample_m(w_slm)
    else:
        raise ValueError(f"Unsupported optical mapping mode: {mapping_mode!r}")

    M = float(objective_map.demag)
    kr_slm = objective_map.sample_to_pre_spatial_frequency_m_inv(kr_sample)
    predicted_bessel_length = float(w0_sample * k_medium / max(kr_sample, EPS))

    # Digital conical phase-screen angle.  k0 rather than k_medium appears
    # because this is an optical-path phase gradient at the modulator plane.
    denom = laser.k0 * (float(target.n_axicon) - float(target.hologram_medium_n))
    gamma = math.atan(kr_slm / max(denom, EPS))
    ell_abs = abs(int(target.ell))
    first_zero_r = float(J0_FIRST_ZERO / kr_sample)
    first_zero_d = float(2.0 * first_zero_r)

    # This remains the infinite-Bessel J'_ell reference.  A finite BG ring can
    # be shifted by Gaussian apodization and is measured separately in the
    # Phase 2K reference module/output metrics.
    ring_r = 0.0 if ell_abs == 0 else float(sp.jnp_zeros(ell_abs, 1)[0] / kr_sample)

    return BeamDesign(
        ell=int(target.ell),
        target_core_diameter_m=D,
        target_scale_definition="equivalent_l0_first_zero_diameter",
        target_equivalent_l0_core_diameter_m=float(D),
        target_bessel_length_m=L,
        n_axicon=float(target.n_axicon),
        hologram_medium_n=float(target.hologram_medium_n),
        sample_medium_n=float(material.refractive_index),
        kr_sample_m_inv=float(kr_sample),
        kr_slm_m_inv=float(kr_slm),
        gamma_slm_rad=float(gamma),
        gamma_slm_deg=float(math.degrees(gamma)),
        magnification_to_sample=float(M),
        mapping_mode=mode,
        objective_map_source=str(objective_map.source),
        objective_map_demag=float(objective_map.demag),
        w0_sample_m=float(w0_sample),
        predicted_bessel_length_m=predicted_bessel_length,
        equivalent_l0_core_radius_m=first_zero_r,
        equivalent_l0_core_diameter_m=first_zero_d,
        equivalent_l0_first_zero_radius_m=first_zero_r,
        equivalent_l0_first_zero_diameter_m=first_zero_d,
        vortex_main_ring_radius_m=float(ring_r),
        vortex_main_ring_diameter_m=float(2.0 * ring_r),
        signum_pi_flip=bool(target.signum_pi_flip),
    )


def fixed_objective_map_from_config(config: TwinConfig) -> vbb_planes.ObjectiveMap:
    """Return only hardware-configured mapping, never a target-derived fallback."""

    if config.relay.magnification_to_sample is not None:
        return vbb_planes.ObjectiveMap(
            demag=float(config.relay.magnification_to_sample),
            n_sample=float(config.material.refractive_index),
            source="RelayConfig.magnification_to_sample",
            mapping_mode="fixed_physical_optics",
        )
    relay_f = float(getattr(config.relay, "effective_relay_f_m", 0.0))
    if relay_f > EPS:
        return vbb_planes.ObjectiveMap(
            demag=float(config.objective.f_eff_m) / relay_f,
            n_sample=float(config.material.refractive_index),
            source="objective_f_eff_over_effective_relay_f",
            mapping_mode="fixed_physical_optics",
        )
    raise ValueError(
        "fixed_physical_optics requires RelayConfig.magnification_to_sample "
        "or a positive RelayConfig.effective_relay_f_m"
    )


def compute_design_from_config(config: TwinConfig) -> BeamDesign:
    """Build a design under the configuration's explicit mapping contract."""

    mode = str(config.mapping_mode).lower().strip()
    fixed_map = fixed_objective_map_from_config(config) if mode == "fixed_physical_optics" else None
    return compute_design_from_targets(
        config.laser,
        config.target,
        config.material,
        mapping_mode=mode,
        fixed_objective_map=fixed_map,
    )


def objective_map_from_config(config: TwinConfig, design: Optional[BeamDesign] = None):
    """Return the map selected by the explicit optical mapping mode."""

    mode = str(config.mapping_mode).lower().strip()
    if mode == "fixed_physical_optics":
        return fixed_objective_map_from_config(config)
    if mode != "target_matched_inverse_design":
        raise ValueError(f"Unsupported optical mapping mode: {config.mapping_mode!r}")
    design = design or compute_design_from_config(config)
    return vbb_planes.ObjectiveMap(
        demag=float(design.objective_map_demag),
        n_sample=float(config.material.refractive_index),
        source=str(design.objective_map_source),
        mapping_mode="target_matched_inverse_design",
    )


__all__ = [
    "J0_FIRST_ZERO",
    "axial_scan_values",
    "compute_design_from_targets",
    "compute_design_from_config",
    "default_config",
    "fixed_objective_map_from_config",
    "get_preset",
    "objective_map_from_config",
]
