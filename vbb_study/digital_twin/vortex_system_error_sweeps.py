"""Predeclared screening sweeps for the physical vortex/Bessel system route.

Values here are sensitivity values, not measurements of the lab.  Every output
carries that provenance.  Measured/LUT/map-driven families remain blocked until
real data are supplied.

The registry deliberately includes x/y counterparts where the carrier or
hardware breaks rotational symmetry.  A few transformations that are physically
degenerate are not promoted to independent sweeps (for example translating an
ideal infinite carrier only adds a global phase).
"""

from __future__ import annotations

import math
from typing import Any, Callable

from vbb_study.digital_twin.vortex_beam_slm_errors import GaussianBeamError, SLMError
from vbb_study.digital_twin.vortex_explicit_4f import FourFError, LensError
from vbb_study.digital_twin.vortex_system_route import AxiconError, SystemErrorConfig


SweepBuilder = Callable[[float], SystemErrorConfig]


def _beam(**kwargs: Any) -> SystemErrorConfig:
    return SystemErrorConfig(beam=GaussianBeamError(**kwargs))


def _slm1(**kwargs: Any) -> SystemErrorConfig:
    return SystemErrorConfig(slm1=SLMError(**kwargs))


def _slm2(**kwargs: Any) -> SystemErrorConfig:
    return SystemErrorConfig(slm2=SLMError(**kwargs))


def _fourf(error: FourFError) -> SystemErrorConfig:
    return SystemErrorConfig(fourf=error)


def _axicon(**kwargs: Any) -> SystemErrorConfig:
    return SystemErrorConfig(axicon=AxiconError(**kwargs))


def system_sweep_registry() -> dict[str, dict[str, Any]]:
    curvature_values = (-4.0, -2.0, math.inf, 2.0, 4.0)
    lateral_values = (-500e-6, -250e-6, 0.0, 250e-6, 500e-6)
    small_lateral_values = (-200e-6, -100e-6, 0.0, 100e-6, 200e-6)
    lens_tilt_values = tuple(math.radians(v) for v in (-0.5, -0.25, 0.0, 0.25, 0.5))
    return {
        "beam_lateral_decentre_x": {
            "values": lateral_values,
            "units": "m",
            "builder": lambda v: _beam(decentre_m=(float(v), 0.0)),
            "fidelity": "physical_input_plane",
        },
        "beam_lateral_decentre_y": {
            "values": lateral_values,
            "units": "m",
            "builder": lambda v: _beam(decentre_m=(0.0, float(v))),
            "fidelity": "physical_input_plane",
        },
        "beam_radius_scale": {
            "values": (0.7, 0.85, 1.0, 1.15, 1.3),
            "units": "ratio",
            "builder": lambda v: _beam(radius_x_scale=float(v), radius_y_scale=float(v)),
            "fidelity": "physical_input_plane",
        },
        "beam_ellipticity": {
            "values": (0.7, 0.85, 1.0, 1.15, 1.3),
            "units": "wx/wy with area approximately held",
            "builder": lambda v: _beam(
                radius_x_scale=math.sqrt(float(v)),
                radius_y_scale=1.0 / math.sqrt(float(v)),
            ),
            "fidelity": "physical_input_plane",
        },
        "beam_curvature_common": {
            "values": curvature_values,
            "units": "m radius of curvature in x and y",
            "builder": lambda v: _beam(
                curvature_radius_x_m=float(v),
                curvature_radius_y_m=float(v),
            ),
            "fidelity": "paraxial_Gaussian_wavefront",
        },
        "beam_curvature_x": {
            "values": curvature_values,
            "units": "m radius of curvature in x only",
            "builder": lambda v: _beam(curvature_radius_x_m=float(v)),
            "fidelity": "paraxial_Gaussian_wavefront",
        },
        "beam_curvature_y": {
            "values": curvature_values,
            "units": "m radius of curvature in y only",
            "builder": lambda v: _beam(curvature_radius_y_m=float(v)),
            "fidelity": "paraxial_Gaussian_wavefront",
        },
        "slm1_hologram_offset_x": {
            "values": small_lateral_values,
            "units": "m",
            "builder": lambda v: _slm1(pattern_offset_m=(float(v), 0.0)),
            "fidelity": "physical_pattern_registration",
        },
        "slm1_hologram_offset_y": {
            "values": small_lateral_values,
            "units": "m",
            "builder": lambda v: _slm1(pattern_offset_m=(0.0, float(v))),
            "fidelity": "physical_pattern_registration",
        },
        "slm2_carrier_rotation": {
            "values": tuple(math.radians(v) for v in (-0.4, -0.2, 0.0, 0.2, 0.4)),
            "units": "rad",
            "builder": lambda v: _slm2(pattern_rotation_rad=float(v)),
            "fidelity": "physical_pattern_registration",
        },
        "slm2_carrier_scale_x": {
            "values": (0.98, 0.99, 1.0, 1.01, 1.02),
            "units": "coordinate scale ratio; effective carrier frequency changes inversely",
            "builder": lambda v: _slm2(pattern_scale_x=float(v)),
            "fidelity": "physical_commanded_carrier_frequency_error",
        },
        "slm_phase_stroke": {
            "values": (0.85, 0.925, 1.0, 1.075, 1.15),
            "units": "ratio applied to both panels",
            "builder": lambda v: SystemErrorConfig(
                slm1=SLMError(phase_stroke_scale=float(v)),
                slm2=SLMError(phase_stroke_scale=float(v)),
            ),
            "fidelity": "sensitivity_only_until_LUT_measured",
        },
        "slm1_phase_stroke": {
            "values": (0.85, 0.925, 1.0, 1.075, 1.15),
            "units": "ratio",
            "builder": lambda v: _slm1(phase_stroke_scale=float(v)),
            "fidelity": "sensitivity_only_until_LUT_measured",
        },
        "slm2_phase_stroke": {
            "values": (0.85, 0.925, 1.0, 1.075, 1.15),
            "units": "ratio",
            "builder": lambda v: _slm2(phase_stroke_scale=float(v)),
            "fidelity": "sensitivity_only_until_LUT_measured",
        },
        "slm_fringing_sigma_x": {
            "values": (0.0, 0.25, 0.5, 0.75, 1.0),
            "units": "pixel sigma applied to both panels in x",
            "builder": lambda v: SystemErrorConfig(
                slm1=SLMError(fringing_sigma_x_px=float(v)),
                slm2=SLMError(fringing_sigma_x_px=float(v)),
            ),
            "fidelity": "calibration_required_convolution_surrogate",
        },
        "slm_fringing_sigma_y": {
            "values": (0.0, 0.25, 0.5, 0.75, 1.0),
            "units": "pixel sigma applied to both panels in y",
            "builder": lambda v: SystemErrorConfig(
                slm1=SLMError(fringing_sigma_y_px=float(v)),
                slm2=SLMError(fringing_sigma_y_px=float(v)),
            ),
            "fidelity": "calibration_required_convolution_surrogate",
        },
        "slm1_fringing_sigma_x": {
            "values": (0.0, 0.25, 0.5, 0.75, 1.0),
            "units": "pixel sigma",
            "builder": lambda v: _slm1(fringing_sigma_x_px=float(v)),
            "fidelity": "calibration_required_convolution_surrogate",
        },
        "slm2_fringing_sigma_x": {
            "values": (0.0, 0.25, 0.5, 0.75, 1.0),
            "units": "pixel sigma",
            "builder": lambda v: _slm2(fringing_sigma_x_px=float(v)),
            "fidelity": "calibration_required_convolution_surrogate",
        },
        "fourf_iris_offset_x": {
            "values": (-0.6e-3, -0.3e-3, 0.0, 0.3e-3, 0.6e-3),
            "units": "m relative to nominal +1 order",
            "builder": lambda v: _fourf(FourFError(iris_offset_m=(float(v), 0.0))),
            "fidelity": "physical_parallel_plane_4f",
        },
        "fourf_iris_offset_y": {
            "values": (-0.6e-3, -0.3e-3, 0.0, 0.3e-3, 0.6e-3),
            "units": "m relative to nominal +1 order",
            "builder": lambda v: _fourf(FourFError(iris_offset_m=(0.0, float(v)))),
            "fidelity": "physical_parallel_plane_4f",
        },
        "fourf_iris_radius_scale": {
            "values": (0.7, 0.85, 1.0, 1.15, 1.3),
            "units": "ratio",
            "builder": lambda v: _fourf(FourFError(iris_radius_scale=float(v))),
            "fidelity": "physical_parallel_plane_4f",
        },
        "fourf_lens1_despace": {
            "values": (-10e-3, -5e-3, 0.0, 5e-3, 10e-3),
            "units": "m",
            "builder": lambda v: _fourf(FourFError(lens1_axial_shift_m=float(v))),
            "fidelity": "physical_parallel_plane_4f",
        },
        "fourf_lens2_despace": {
            "values": (-10e-3, -5e-3, 0.0, 5e-3, 10e-3),
            "units": "m",
            "builder": lambda v: _fourf(FourFError(lens2_axial_shift_m=float(v))),
            "fidelity": "physical_parallel_plane_4f",
        },
        "fourf_lens1_focal_scale": {
            "values": (0.98, 0.99, 1.0, 1.01, 1.02),
            "units": "focal-length ratio",
            "builder": lambda v: _fourf(FourFError(lens1=LensError(focal_length_scale=float(v)))),
            "fidelity": "paraxial_thin_lens_focal_error",
        },
        "fourf_lens2_focal_scale": {
            "values": (0.98, 0.99, 1.0, 1.01, 1.02),
            "units": "focal-length ratio",
            "builder": lambda v: _fourf(FourFError(lens2=LensError(focal_length_scale=float(v)))),
            "fidelity": "paraxial_thin_lens_focal_error",
        },
        "fourf_lens1_decentre_x": {
            "values": lateral_values,
            "units": "m",
            "builder": lambda v: _fourf(FourFError(lens1=LensError(decentre_m=(float(v), 0.0)))),
            "fidelity": "paraxial_thin_lens_physical_decentre",
        },
        "fourf_lens1_decentre_y": {
            "values": lateral_values,
            "units": "m",
            "builder": lambda v: _fourf(FourFError(lens1=LensError(decentre_m=(0.0, float(v))))),
            "fidelity": "paraxial_thin_lens_physical_decentre",
        },
        "fourf_lens2_decentre_x": {
            "values": lateral_values,
            "units": "m",
            "builder": lambda v: _fourf(FourFError(lens2=LensError(decentre_m=(float(v), 0.0)))),
            "fidelity": "paraxial_thin_lens_physical_decentre",
        },
        "fourf_lens2_decentre_y": {
            "values": lateral_values,
            "units": "m",
            "builder": lambda v: _fourf(FourFError(lens2=LensError(decentre_m=(0.0, float(v))))),
            "fidelity": "paraxial_thin_lens_physical_decentre",
        },
        "fourf_lens1_tilt_x": {
            "values": lens_tilt_values,
            "units": "rad",
            "builder": lambda v: _fourf(FourFError(lens1=LensError(tilt_rad=(float(v), 0.0)))),
            "fidelity": "scalar_paraxial_rotated_thin_lens_plane",
        },
        "fourf_lens1_tilt_y": {
            "values": lens_tilt_values,
            "units": "rad",
            "builder": lambda v: _fourf(FourFError(lens1=LensError(tilt_rad=(0.0, float(v))))),
            "fidelity": "scalar_paraxial_rotated_thin_lens_plane",
        },
        "fourf_lens2_tilt_x": {
            "values": lens_tilt_values,
            "units": "rad",
            "builder": lambda v: _fourf(FourFError(lens2=LensError(tilt_rad=(float(v), 0.0)))),
            "fidelity": "scalar_paraxial_rotated_thin_lens_plane",
        },
        "fourf_lens2_tilt_y": {
            "values": lens_tilt_values,
            "units": "rad",
            "builder": lambda v: _fourf(FourFError(lens2=LensError(tilt_rad=(0.0, float(v))))),
            "fidelity": "scalar_paraxial_rotated_thin_lens_plane",
        },
        "axicon_lateral_decentre_x": {
            "values": lateral_values,
            "units": "m",
            "builder": lambda v: _axicon(decentre_m=(float(v), 0.0)),
            "fidelity": "physical_axicon_coordinate_translation",
        },
        "axicon_lateral_decentre_y": {
            "values": lateral_values,
            "units": "m",
            "builder": lambda v: _axicon(decentre_m=(0.0, float(v))),
            "fidelity": "physical_axicon_coordinate_translation",
        },
        "axicon_rigid_tilt_x": {
            "values": lens_tilt_values,
            "units": "rad",
            "builder": lambda v: _axicon(tilt_rad=(float(v), 0.0)),
            "fidelity": "scalar_rotated_angular_spectrum; full refractive-vector claims blocked",
        },
        "axicon_rigid_tilt_y": {
            "values": lens_tilt_values,
            "units": "rad",
            "builder": lambda v: _axicon(tilt_rad=(0.0, float(v))),
            "fidelity": "scalar_rotated_angular_spectrum; full refractive-vector claims blocked",
        },
        "axicon_round_tip": {
            "values": (0.0, 2e-6, 5e-6, 10e-6, 20e-6),
            "units": "m hyperboloidal parameter",
            "builder": lambda v: _axicon(
                tip_model="sharp" if float(v) == 0.0 else "hyperboloidal_round",
                rounding_parameter_m=float(v),
            ),
            "fidelity": "shallow_angle_tip_defect_on_exact_Snell_cone",
        },
        "axicon_flat_tip": {
            "values": (0.0, 10e-6, 25e-6, 50e-6, 100e-6),
            "units": "m flat radius",
            "builder": lambda v: _axicon(
                tip_model="sharp" if float(v) == 0.0 else "flat_blunt",
                flat_tip_radius_m=float(v),
            ),
            "fidelity": "shallow_angle_tip_defect_on_exact_Snell_cone",
        },
        "axicon_base_angle_scale": {
            "values": (0.9, 0.95, 1.0, 1.05, 1.1),
            "units": "ratio",
            "builder": lambda v: _axicon(base_angle_scale=float(v)),
            "fidelity": "exact_normal_incidence_Snell_cone",
        },
        "axicon_index_scale": {
            "values": (0.99, 0.995, 1.0, 1.005, 1.01),
            "units": "ratio",
            "builder": lambda v: _axicon(refractive_index_scale=float(v)),
            "fidelity": "exact_normal_incidence_Snell_cone; dispersion calibration required",
        },
    }


def blocked_or_data_driven_families() -> dict[str, dict[str, Any]]:
    return {
        "slm_phase_LUT": {
            "status": "data_required",
            "required": "measured grey->phase LUT for each HOLOEYE panel at 1029 nm and bench incidence/polarisation",
        },
        "slm_static_SPNU": {
            "status": "data_required",
            "required": "measured static phase/wavefront map per panel",
        },
        "slm_fringing_absolute": {
            "status": "calibration_required",
            "required": "fit directional crosstalk kernel or subpixel Jones model to measured diffraction efficiencies",
        },
        "slm_incidence_and_polarisation_response": {
            "status": "calibration_required",
            "required": "panel-specific phase response versus incidence angle and input polarisation; do not infer from a generic LCOS model",
        },
        "lens_OPD_maps": {
            "status": "data_required",
            "required": "measured/manufacturer lens wavefront maps or declared generic sensitivity coefficients",
        },
        "lens_clear_apertures": {
            "status": "measurement_required",
            "required": "actual clear radii and mechanical stops of both 4F lenses",
        },
        "thick_lens_large_tilt": {
            "status": "full_surface_refraction_required",
            "required": "surface-by-surface thick-lens/vector refraction for absolute large-angle prediction",
        },
        "axicon_clear_aperture": {
            "status": "measurement_required",
            "required": "actual clear aperture radius",
        },
        "axicon_surface_error": {
            "status": "data_required",
            "required": "surface profilometry/interferometric height map",
        },
        "high_angle_refractive_axicon_surface": {
            "status": "full_surface_refraction_required",
            "required": "explicit manufacturer angle convention and vector Snell/Fresnel two-surface model",
        },
        "objective_and_sample_errors": {
            "status": "separate_vector_branch",
            "required": "Phase 2C vector Debye/interface route; do not fold into source-scale Bessel propagation",
        },
        "camera_errors": {
            "status": "measurement_branch",
            "required": "pixel scale, saturation, background, exposure and sensor response",
        },
        "pulse_energy_and_wavelength_jitter": {
            "status": "statistical_calibration_branch",
            "required": "measured distributions/time series rather than arbitrary single perturbations",
        },
    }
