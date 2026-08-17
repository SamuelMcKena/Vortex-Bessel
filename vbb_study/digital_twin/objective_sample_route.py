"""Calibrated objective-entrance-pupil -> sample vector field route.

This is a reusable wrapper around the independently validated Phase 2C
Richards-Wolf/Debye and spectral vector Fresnel solvers. It does not guess the
relay mapping into the objective: the caller must provide the field actually
mapped to the objective entrance-pupil coordinates.

Both a scalar pupil plus global Jones vector and a genuinely spatially varying
``Ex(x,y), Ey(x,y)`` pupil are supported. The latter is required for radial,
azimuthal and segmented vector beams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vbb_study.equations.vector_debye import DebyeConfig, VectorFieldPlane, debye_focus_plane
from vbb_study.equations.vector_fresnel_interface import (
    FresnelInterfaceConfig,
    FresnelInterfaceResult,
    transmit_vector_field_planar_interface,
)
from vbb_study.vector_field import VectorField, propagate_vector_asm


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class ObjectiveSampleConfig:
    wavelength_m: float
    numerical_aperture: float
    objective_focal_length_m: float
    objective_pupil_radius_m: float
    incident_medium_index: float = 1.0
    sample_refractive_index: complex = 1.45
    sample_depth_m: float = 0.0
    apodisation: str = "sqrt_cosine"
    fft_pad_factor: int = 2

    def validate(self) -> None:
        if self.wavelength_m <= 0.0 or self.objective_focal_length_m <= 0.0 or self.objective_pupil_radius_m <= 0.0:
            raise ValueError("wavelength, focal length and pupil radius must be positive")
        if not 0.0 < self.numerical_aperture <= self.incident_medium_index:
            raise ValueError("numerical_aperture must lie in (0,n_incident]")
        sample_n = complex(self.sample_refractive_index)
        if sample_n.real <= 0.0 or sample_n.imag < 0.0:
            raise ValueError("sample_refractive_index must use passive Im(n)>=0 convention")
        if self.sample_depth_m < 0.0:
            raise ValueError("sample_depth_m cannot be negative")
        if int(self.fft_pad_factor) < 1:
            raise ValueError("fft_pad_factor must be >=1")


@dataclass(frozen=True)
class ObjectiveSampleResult:
    focal_plane_air: VectorFieldPlane
    interface_transmission: FresnelInterfaceResult
    field_in_sample: VectorField
    pupil_power_fraction: float
    metadata: Mapping[str, Any]


def _uniform_square_grid(grid: Mapping[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    if shape[0] != shape[1]:
        raise ValueError("Debye wrapper currently requires a square pupil field")
    x = np.asarray(grid["x"], dtype=float)
    if x.ndim != 1 or x.size != shape[1] or np.any(np.diff(x) <= 0.0):
        raise ValueError("grid x-axis is incompatible with pupil field")
    y = np.asarray(grid.get("y", x), dtype=float)
    if y.shape != x.shape or not np.allclose(y, x, rtol=1e-9, atol=1e-15):
        raise ValueError("current Debye wrapper requires identical square x/y axes")
    dx = float(np.median(np.diff(x)))
    if not np.allclose(np.diff(x), dx, rtol=1e-9, atol=1e-15):
        raise ValueError("pupil grid must be uniform")
    return x, dx


def _validate_material_depth(config: ObjectiveSampleConfig) -> complex:
    config.validate()
    sample_n = complex(config.sample_refractive_index)
    if config.sample_depth_m > 0.0 and abs(sample_n.imag) > 1.0e-14:
        raise ValueError(
            "current vector ASM material propagation requires a real lossless refractive index; "
            "absorbing-media depth propagation needs a separately validated complex-k propagator"
        )
    return sample_n


def focus_vector_pupil_into_sample(
    ex_pupil_field: np.ndarray,
    ey_pupil_field: np.ndarray,
    pupil_grid: Mapping[str, Any],
    *,
    config: ObjectiveSampleConfig,
) -> ObjectiveSampleResult:
    """Focus a spatially varying transverse Jones pupil into the sample.

    This is the authoritative entrance-pupil API for cylindrical and segmented
    vector beams. ``Ex`` and ``Ey`` retain their local amplitude/phase structure
    across the pupil; no global polarization approximation is introduced.

    The vector Fresnel interface supports a passive complex refractive index.
    The repository's validated in-material vector ASM currently assumes a real
    lossless propagation index, so absorbing-media propagation to non-zero depth
    is explicitly blocked rather than silently dropping Im(n).
    """

    sample_n = _validate_material_depth(config)
    ex = np.asarray(ex_pupil_field, dtype=np.complex128)
    ey = np.asarray(ey_pupil_field, dtype=np.complex128)
    if ex.ndim != 2 or ey.shape != ex.shape:
        raise ValueError("ex_pupil_field and ey_pupil_field must be same-shape 2-D arrays")
    x, dx = _uniform_square_grid(pupil_grid, ex.shape)
    X, Y = np.meshgrid(x, x, indexing="xy")
    pupil = np.hypot(X, Y) <= float(config.objective_pupil_radius_m)
    total_in = float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2))
    ex = np.where(pupil, ex, 0.0j)
    ey = np.where(pupil, ey, 0.0j)
    pupil_power = float(np.sum(np.abs(ex) ** 2 + np.abs(ey) ** 2))
    pupil_fraction = pupil_power / max(total_in, EPS)

    n_pad = ex.shape[1] * int(config.fft_pad_factor)
    native_frequency = np.fft.fftshift(np.fft.fftfreq(n_pad, d=dx))
    output_axis = (
        float(config.wavelength_m)
        * float(config.objective_pupil_radius_m)
        / float(config.numerical_aperture)
        * native_frequency
    )
    debye_cfg = DebyeConfig(
        wavelength_m=float(config.wavelength_m),
        refractive_index=float(config.incident_medium_index),
        numerical_aperture=float(config.numerical_aperture),
        focal_length_m=float(config.objective_focal_length_m),
        pupil_radius_m=float(config.objective_pupil_radius_m),
        apodisation=config.apodisation,
        propagation_direction="+z",
        backend="cartesian_fft",
        quadrature_order_r=48,
        quadrature_order_phi=144,
        max_output_points=max(1024, int(output_axis.size)),
        fft_pad_factor=int(config.fft_pad_factor),
    )
    focal = debye_focus_plane(
        ex,
        ey,
        x,
        x,
        output_axis,
        output_axis,
        0.0,
        debye_cfg,
    )
    output_dx = float(np.median(np.diff(output_axis)))
    interface = transmit_vector_field_planar_interface(
        focal.Ex,
        focal.Ey,
        focal.Ez,
        output_dx,
        output_dx,
        FresnelInterfaceConfig(
            wavelength_m=float(config.wavelength_m),
            n_incident=complex(config.incident_medium_index),
            n_transmitted=sample_n,
            include_evanescent=False,
        ),
    )
    output_grid = {
        "N": int(output_axis.size),
        "dx": output_dx,
        "dy": output_dx,
        "x": np.asarray(output_axis, dtype=float),
        "y": np.asarray(output_axis, dtype=float),
    }
    material_input = VectorField(
        ex=interface.Ex,
        ey=interface.Ey,
        ez=interface.Ez,
        grid=output_grid,
        wavelength_m=float(config.wavelength_m),
        medium_index=float(sample_n.real),
        metadata={"interface_model": "spectral_vector_fresnel"},
    )
    material = (
        propagate_vector_asm(material_input, float(config.sample_depth_m))
        if config.sample_depth_m > 0.0
        else material_input
    )
    return ObjectiveSampleResult(
        focal_plane_air=focal,
        interface_transmission=interface,
        field_in_sample=material,
        pupil_power_fraction=float(pupil_fraction),
        metadata={
            "solver_chain": "vector_Debye_Richards_Wolf -> spectral_s/p_Fresnel -> vector_ASM_in_sample",
            "input_plane": "calibrated_objective_entrance_pupil",
            "input_polarization_model": "spatially_varying_Ex_Ey",
            "objective_NA": float(config.numerical_aperture),
            "objective_focal_length_m": float(config.objective_focal_length_m),
            "objective_pupil_radius_m": float(config.objective_pupil_radius_m),
            "sample_refractive_index": [float(sample_n.real), float(sample_n.imag)],
            "sample_depth_m": float(config.sample_depth_m),
            "in_material_propagation_index": float(sample_n.real),
            "absorbing_depth_propagation_supported": False,
            "pupil_power_fraction": float(pupil_fraction),
            "interface_transmitted_power_fraction": float(interface.diagnostics["transmitted_power_fraction"]),
            "interface_R_plus_T": float(interface.diagnostics["lossless_R_plus_T"]),
            "calibration_requirement": "relay/objective pupil mapping must be measured before calling the supplied pupil field a bench prediction",
        },
    )


def focus_pupil_field_into_sample(
    scalar_pupil_field: np.ndarray,
    pupil_grid: Mapping[str, Any],
    *,
    config: ObjectiveSampleConfig,
    input_jones_xy: tuple[complex, complex] = (1.0 + 0.0j, 0.0 + 0.0j),
) -> ObjectiveSampleResult:
    """Focus a scalar pupil carrying one global Jones polarization.

    This compatibility wrapper constructs ``Ex/Ey`` and then calls
    :func:`focus_vector_pupil_into_sample`; the vector solver itself therefore
    has only one authoritative objective/interface implementation.
    """

    scalar = np.asarray(scalar_pupil_field, dtype=np.complex128)
    if scalar.ndim != 2:
        raise ValueError("scalar_pupil_field must be 2-D")
    jones = np.asarray(input_jones_xy, dtype=np.complex128)
    norm = float(np.linalg.norm(jones))
    if norm <= EPS:
        raise ValueError("input_jones_xy cannot be zero")
    jones = jones / norm
    result = focus_vector_pupil_into_sample(
        scalar * jones[0],
        scalar * jones[1],
        pupil_grid,
        config=config,
    )
    metadata = dict(result.metadata)
    metadata["input_polarization_model"] = "global_Jones_times_scalar_pupil"
    metadata["input_jones_xy"] = [
        [float(jones[0].real), float(jones[0].imag)],
        [float(jones[1].real), float(jones[1].imag)],
    ]
    return ObjectiveSampleResult(
        focal_plane_air=result.focal_plane_air,
        interface_transmission=result.interface_transmission,
        field_in_sample=result.field_in_sample,
        pupil_power_fraction=result.pupil_power_fraction,
        metadata=metadata,
    )


__all__ = [
    "ObjectiveSampleConfig",
    "ObjectiveSampleResult",
    "focus_pupil_field_into_sample",
    "focus_vector_pupil_into_sample",
]
