"""Stage 8C.3R physical component-plane lab-realism pipeline.

This module builds a physically honest path that carries a *complex* optical
field through the modelled optical planes and applies each perturbation at the
plane where it physically lives, **before** propagation:

    InputFieldState -> SLMFieldState -> PupilPlaneState -> SampleEntranceState
                    -> PropagatedFieldStack (genuine angular-spectrum propagation)

It reuses the locked scalar-engine primitives (no locked equation is modified):

    * ``vbb_study.equations.fields.make_xy_grid``      (centred square grid)
    * ``vbb_study.equations.propagation.make_bl_asm_propagator``
                                                        (band-limited ASM)
    * ``vbb_study.equations.holography.quantize_phase_rad``

Perturbations therefore produce *emergent* downstream behaviour: an aperture
clips the field and the output shows diffraction rings (never a hard disc edge);
a beam tilt is a pupil phase ramp and the walk-off grows with z; a pupil clip
genuinely lowers transmitted energy.  Phase-only operations conserve power;
passive apertures lose power and that loss is carried into the fluence scaling
WITHOUT any per-plane re-normalisation back to the pre-clip energy.

Model status: optical / fluence diagnostic only.  ``final_export_allowed=False``.
No material response / absorbed energy / dose / plasma / index change is modelled
or claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.holography import quantize_phase_rad
from vbb_study.equations.propagation import make_bl_asm_propagator

from vbb_study.digital_twin.component_plane_states import (
    ComponentPlaneState,
    PropagatedFieldStack,
    field_power,
)

TWOPI = 2.0 * np.pi
_UM = 1e-6

# Controls that this engine genuinely cannot represent at a physical plane
# (no explicit 4F / relay imaging plane exists in the direct-propagation model).
# They are honestly retained as warnings rather than faked.
WARNING_ONLY_CONTROLS: tuple[str, ...] = (
    "enable_first_order_filter",
    "enable_first_order_filter_decentre",
    "enable_first_order_filter_clipping",
    "enable_unwanted_order_leakage",
    "enable_relay_magnification_error",
    "enable_relay_decentre",
    "enable_relay_tilt",
    "enable_relay_aperture",
    "enable_slm_rotation",
    "enable_mask_rotation",
    "enable_physical_axicon_misalignment_angle",
    "physical_axicon_angle_error_deg",
    "enable_axicon_apex_defect",
    "enable_pointing_jitter",
    "enable_stage_position_jitter",
    "enable_focus_drift",
)

_WARNING_REASON = {
    "enable_first_order_filter": "no explicit 4F Fourier plane in the direct-propagation engine",
    "enable_first_order_filter_decentre": "no explicit 4F Fourier plane in the direct-propagation engine",
    "enable_first_order_filter_clipping": "no explicit 4F Fourier plane in the direct-propagation engine",
    "enable_unwanted_order_leakage": "shifted diffraction orders need an explicit Fourier plane",
    "enable_relay_magnification_error": "no relay imaging plane is modelled",
    "enable_relay_decentre": "no relay imaging plane is modelled",
    "enable_relay_tilt": "no relay imaging plane is modelled",
    "enable_relay_aperture": "no relay imaging plane is modelled",
    "enable_slm_rotation": "mask resampling/rotation not implemented",
    "enable_mask_rotation": "mask resampling/rotation not implemented",
    "physical_axicon_angle_error_deg": "cone-angle retune needs a route-level k_r change",
    "enable_axicon_apex_defect": "sub-resolution apex defect not modelled",
    "enable_pointing_jitter": "statistical ensemble not modelled",
    "enable_stage_position_jitter": "statistical ensemble not modelled",
    "enable_focus_drift": "statistical ensemble not modelled",
}


@dataclass(frozen=True)
class ComponentPlaneConfig:
    """Geometry and beam parameters for the physical component-plane pipeline."""

    wavelength_nm: float = 1030.0
    n_medium: float = 1.0
    ell: int = 3
    beam_waist_um: float = 26.0
    kr_rad_per_um: float = 1.05
    grid_N: int = 256
    dx_um: float = 0.5
    z_min_um: float = 0.0
    z_max_um: float = 225.0
    n_z: int = 46
    input_pulse_energy_uJ: float = 95.76
    pupil_norm_radius_um: float = 30.0
    bandlimit: bool = True

    @classmethod
    def fast(cls, **overrides: Any) -> "ComponentPlaneConfig":
        """A smaller grid/stack for tests and quick previews."""
        base = dict(
            grid_N=160,
            dx_um=0.6,
            n_z=24,
            beam_waist_um=22.0,
            pupil_norm_radius_um=26.0,
        )
        base.update(overrides)
        return cls(**base)

    @property
    def wavelength_m(self) -> float:
        return float(self.wavelength_nm) * 1e-9

    @property
    def dx_m(self) -> float:
        return float(self.dx_um) * _UM

    @property
    def k_medium_rad_per_m(self) -> float:
        return TWOPI * float(self.n_medium) / self.wavelength_m


@dataclass(frozen=True)
class ComponentPlaneRun:
    """Full pipeline output: per-plane states + propagated stack + metadata."""

    config: ComponentPlaneConfig
    input_state: ComponentPlaneState
    slm_state: ComponentPlaneState
    pupil_state: ComponentPlaneState
    sample_entrance_state: ComponentPlaneState
    propagated_stack: PropagatedFieldStack
    warnings: tuple[str, ...]
    applied_components: tuple[str, ...]
    predicted_steering: Mapping[str, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    final_export_allowed: bool = False
    model_status: str = "optical_prediction"

    @property
    def reference_plane_state(self) -> ComponentPlaneState:
        """Stage 8C.3R.1 alias: the free-space reference plane (n=1.0), in air.

        This is the intended sample-entrance reference plane.  No material model is
        active; it is not an in-material plane.
        """
        return self.sample_entrance_state


# ---------------------------------------------------------------------------
# Control access helpers
# ---------------------------------------------------------------------------


def _flag(c: Mapping[str, Any], name: str) -> bool:
    return bool(c.get(name, False))


def _f(c: Mapping[str, Any], name: str, default: float = 0.0) -> float:
    v = c.get(name, default)
    return float(default if v is None else v)


# ---------------------------------------------------------------------------
# Plane builders
# ---------------------------------------------------------------------------


def _build_input_field(
    grid: dict[str, Any],
    config: ComponentPlaneConfig,
    c: Mapping[str, Any],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Complex entrance field with input-plane perturbations on the field itself."""
    X = np.asarray(grid["X"], dtype=float)  # metres
    Y = np.asarray(grid["Y"], dtype=float)
    applied: list[str] = []
    meta: dict[str, Any] = {}

    # Beam centre (decentre is an input-plane amplitude shift).
    x0 = y0 = 0.0
    if _flag(c, "enable_beam_decentre"):
        x0 = _f(c, "beam_decentre_x_um") * _UM
        y0 = _f(c, "beam_decentre_y_um") * _UM
        applied.append("input_beam_decentre")

    # Beam waist / ellipticity / rotation.
    if _flag(c, "enable_beam_ellipticity"):
        wx = max(_f(c, "beam_radius_x_um", config.beam_waist_um), 1e-3) * _UM
        wy = max(_f(c, "beam_radius_y_um", config.beam_waist_um), 1e-3) * _UM
        rot = np.deg2rad(_f(c, "beam_rotation_deg"))
        applied.append("input_beam_ellipticity")
    else:
        wx = wy = float(config.beam_waist_um) * _UM
        rot = 0.0

    Xc = X - x0
    Yc = Y - y0
    Xr = Xc * np.cos(rot) + Yc * np.sin(rot)
    Yr = -Xc * np.sin(rot) + Yc * np.cos(rot)
    amp = np.exp(-((Xr / wx) ** 2 + (Yr / wy) ** 2))
    E = amp.astype(complex)

    # Beam tilt = pupil/input phase ramp E -> E exp[i(kx x + ky y)].
    if _flag(c, "enable_beam_tilt"):
        k = config.k_medium_rad_per_m
        kx = k * np.sin(_f(c, "beam_tilt_x_mrad") * 1e-3)  # rad/m, +x direction
        ky = k * np.sin(_f(c, "beam_tilt_y_mrad") * 1e-3)
        E = E * np.exp(1j * (kx * X + ky * Y))
        applied.append("input_beam_tilt_phase_ramp")
        meta["beam_tilt_phase_ramp"] = "E(x,y) exp[i(kx0 x + ky0 y)]"
        meta["tilt_kx_rad_per_m"] = float(kx)
        meta["tilt_ky_rad_per_m"] = float(ky)

    return E, applied, meta


def _circular_mask(grid: dict[str, Any], radius_um: float, cx_um: float, cy_um: float) -> np.ndarray:
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    r = np.hypot(X - cx_um * _UM, Y - cy_um * _UM)
    return (r <= radius_um * _UM).astype(float)


def _rect_mask(grid: dict[str, Any], w_um: float, h_um: float, cx_um: float, cy_um: float) -> np.ndarray:
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    inside = (np.abs(X - cx_um * _UM) <= 0.5 * w_um * _UM) & (
        np.abs(Y - cy_um * _UM) <= 0.5 * h_um * _UM
    )
    return inside.astype(float)


def _phase_mask(
    grid: dict[str, Any],
    config: ComponentPlaneConfig,
    c: Mapping[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Vortex + axicon phase with INDEPENDENT centres, quantisation and noise."""
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    applied: list[str] = []

    # Common phase-mask centre offset, plus independent vortex/axicon offsets.
    cx = cy = 0.0
    if _flag(c, "enable_slm_phase_centre_offset"):
        cx += _f(c, "slm_phase_centre_offset_x_um")
        cy += _f(c, "slm_phase_centre_offset_y_um")
        applied.append("slm_common_phase_centre_offset")
    xv, yv = cx, cy
    xa, ya = cx, cy
    if _flag(c, "enable_vortex_centre_offset"):
        xv += _f(c, "vortex_centre_offset_x_um")
        yv += _f(c, "vortex_centre_offset_y_um")
        applied.append("vortex_centre_offset")
    if _flag(c, "enable_axicon_centre_offset"):
        xa += _f(c, "axicon_centre_offset_x_um")
        ya += _f(c, "axicon_centre_offset_y_um")
        applied.append("axicon_centre_offset")

    kr = float(config.kr_rad_per_um) / _UM  # rad/m
    phi_vortex = float(config.ell) * np.arctan2(Y - yv * _UM, X - xv * _UM)
    phi_axicon = -kr * np.hypot(X - xa * _UM, Y - ya * _UM)
    phi = phi_vortex + phi_axicon

    if _flag(c, "enable_slm_phase_quantisation"):
        levels = max(2, int(_f(c, "slm_phase_levels", 256)))
        bits = max(1, int(round(np.log2(levels))))
        phi = quantize_phase_rad(phi, bits)
        applied.append(f"slm_phase_quantisation_{1 << bits}lvl")

    if _flag(c, "enable_slm_phase_noise") and _f(c, "slm_phase_noise_rms_rad") > 0:
        rng = np.random.default_rng(int(_f(c, "slm_phase_noise_seed", 11)))
        phi = phi + rng.normal(0.0, _f(c, "slm_phase_noise_rms_rad"), size=phi.shape)
        applied.append("slm_phase_noise")

    return phi, applied


def _device_amplitude_mask(
    grid: dict[str, Any],
    c: Mapping[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """SLM device amplitude effects (dead pixels / fill factor / active area)."""
    X = np.asarray(grid["X"], dtype=float)
    mask = np.ones_like(X, dtype=float)
    applied: list[str] = []

    if _flag(c, "enable_slm_fill_factor"):
        ff = float(np.clip(_f(c, "slm_fill_factor", 1.0), 0.0, 1.0))
        mask = mask * np.sqrt(ff)
        applied.append("slm_fill_factor")

    if _flag(c, "enable_slm_dead_pixels") and _f(c, "dead_pixel_fraction") > 0:
        rng = np.random.default_rng(int(_f(c, "dead_pixel_seed", 7)))
        dead = rng.random(X.shape) < float(np.clip(_f(c, "dead_pixel_fraction"), 0.0, 1.0))
        mask = mask * (~dead).astype(float)
        applied.append("slm_dead_pixels")

    if _flag(c, "enable_slm_active_area"):
        mask = mask * _rect_mask(
            grid,
            _f(c, "slm_active_width_um", 50.0),
            _f(c, "slm_active_height_um", 50.0),
            _f(c, "slm_active_area_decentre_x_um"),
            _f(c, "slm_active_area_decentre_y_um"),
        )
        applied.append("slm_active_area_clip")

    return mask, applied


def _zernike_phase(
    grid: dict[str, Any],
    config: ComponentPlaneConfig,
    c: Mapping[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Low-order Zernike pupil phase in waves; phase = 2*pi*sum(c_j Z_j)."""
    X = np.asarray(grid["X"], dtype=float) / (config.pupil_norm_radius_um * _UM)
    Y = np.asarray(grid["Y"], dtype=float) / (config.pupil_norm_radius_um * _UM)
    rho2 = X**2 + Y**2
    defocus = _f(c, "zernike_defocus_waves")
    astig0 = _f(c, "zernike_astig_0_waves")
    astig45 = _f(c, "zernike_astig_45_waves")
    coma_x = _f(c, "zernike_coma_x_waves")
    coma_y = _f(c, "zernike_coma_y_waves")
    spherical = _f(c, "zernike_spherical_waves")
    waves = (
        defocus * (2.0 * rho2 - 1.0)
        + astig0 * (X**2 - Y**2)
        + astig45 * (2.0 * X * Y)
        + coma_x * (3.0 * rho2 - 2.0) * X
        + coma_y * (3.0 * rho2 - 2.0) * Y
        + spherical * (6.0 * rho2**2 - 6.0 * rho2 + 1.0)
    )
    applied = ["zernike_pupil_phase"]
    return TWOPI * waves, applied


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_component_plane_pipeline(
    controls: Mapping[str, Any] | None = None,
    *,
    config: ComponentPlaneConfig | None = None,
) -> ComponentPlaneRun:
    """Run the physical component-plane pipeline and return all plane states."""
    c = dict(controls or {})
    config = config or ComponentPlaneConfig()
    grid = make_xy_grid(int(config.grid_N), config.dx_m)
    x_um = np.asarray(grid["x"], dtype=float) * 1e6
    y_um = x_um.copy()
    dx_um = float(config.dx_um)
    E_in_nominal = float(config.input_pulse_energy_uJ)

    warnings: list[str] = []
    applied_all: list[str] = []

    # --- Input field state -------------------------------------------------
    E, applied_in, meta_in = _build_input_field(grid, config, c)
    p_input = field_power(E, dx_um, dx_um)  # reference power AFTER amplitude build
    p_input = max(p_input, 1e-30)

    if _flag(c, "enable_input_aperture"):
        E = E * _circular_mask(
            grid,
            _f(c, "input_aperture_radius_um", 24.0),
            _f(c, "input_aperture_decentre_x_um"),
            _f(c, "input_aperture_decentre_y_um"),
        )
        applied_in.append("input_aperture_clip")
    p_after_input = field_power(E, dx_um, dx_um)
    applied_all += applied_in

    input_state = ComponentPlaneState(
        plane_name="input_complex_field",
        field=E.copy(),
        x_um=x_um,
        y_um=y_um,
        dx_um=dx_um,
        dy_um=dx_um,
        pulse_energy_before_uJ=E_in_nominal,
        pulse_energy_after_uJ=E_in_nominal * (p_after_input / p_input),
        transmitted_fraction=p_after_input / p_input,
        applied_components=tuple(applied_in),
        metadata=meta_in,
    )

    # --- SLM field state ---------------------------------------------------
    phi, applied_phase = _phase_mask(grid, config, c)
    dev_mask, applied_dev = _device_amplitude_mask(grid, c)
    E_carrier = E * dev_mask  # unmodulated zero-order carrier (device-limited)
    E_modulated = E * dev_mask * np.exp(1j * phi)

    leak = 0.0
    applied_leak: list[str] = []
    if _flag(c, "enable_zero_order_leakage"):
        leak = float(np.clip(_f(c, "zero_order_leakage_fraction"), 0.0, 0.95))
        applied_leak.append("zero_order_carrier_leakage")
    E_slm = np.sqrt(1.0 - leak) * E_modulated + np.sqrt(leak) * E_carrier

    p_after_slm = field_power(E_slm, dx_um, dx_um)
    applied_slm = applied_phase + applied_dev + applied_leak
    applied_all += applied_slm

    slm_state = ComponentPlaneState(
        plane_name="SLM_field",
        field=E_slm.copy(),
        x_um=x_um,
        y_um=y_um,
        dx_um=dx_um,
        dy_um=dx_um,
        pulse_energy_before_uJ=E_in_nominal * (p_after_input / p_input),
        pulse_energy_after_uJ=E_in_nominal * (p_after_slm / p_input),
        transmitted_fraction=p_after_slm / max(p_after_input, 1e-30),
        applied_components=tuple(applied_slm),
        metadata={"zero_order_leakage_fraction": leak},
    )
    E = E_slm

    # --- Pupil plane state -------------------------------------------------
    applied_pupil: list[str] = []
    if _flag(c, "enable_pupil_clipping"):
        E = E * _circular_mask(
            grid,
            _f(c, "pupil_radius_um", 20.0),
            _f(c, "pupil_decentre_x_um"),
            _f(c, "pupil_decentre_y_um"),
        )
        applied_pupil.append("pupil_clip")
    if _flag(c, "enable_zernike_aberrations"):
        zphi, applied_z = _zernike_phase(grid, config, c)
        E = E * np.exp(1j * zphi)
        applied_pupil += applied_z
    p_after_pupil = field_power(E, dx_um, dx_um)
    applied_all += applied_pupil

    pupil_state = ComponentPlaneState(
        plane_name="objective_pupil_plane",
        field=E.copy(),
        x_um=x_um,
        y_um=y_um,
        dx_um=dx_um,
        dy_um=dx_um,
        pulse_energy_before_uJ=E_in_nominal * (p_after_slm / p_input),
        pulse_energy_after_uJ=E_in_nominal * (p_after_pupil / p_input),
        transmitted_fraction=p_after_pupil / max(p_after_slm, 1e-30),
        applied_components=tuple(applied_pupil),
    )

    transmitted_fraction = float(np.clip(p_after_pupil / p_input, 0.0, 1.0))
    sample_pulse_energy = E_in_nominal * transmitted_fraction

    sample_entrance_state = ComponentPlaneState(
        plane_name="free_space_reference_plane",
        field=E.copy(),
        x_um=x_um,
        y_um=y_um,
        dx_um=dx_um,
        dy_um=dx_um,
        pulse_energy_before_uJ=E_in_nominal,
        pulse_energy_after_uJ=sample_pulse_energy,
        transmitted_fraction=transmitted_fraction,
        applied_components=("free_space_reference_entrance",),
        metadata={
            "reference_plane": "intended sample-entrance reference plane, n=1.0",
            "no_material_model": True,
        },
    )

    # --- Propagation -------------------------------------------------------
    prop = make_bl_asm_propagator(
        E,
        grid,
        config.wavelength_m,
        n_medium=float(config.n_medium),
        bandlimit=bool(config.bandlimit),
    )
    z_um = np.linspace(float(config.z_min_um), float(config.z_max_um), int(config.n_z))
    intensity = np.empty((z_um.size, x_um.size, x_um.size), dtype=float)
    for i, zi in enumerate(z_um):
        U = prop(float(zi) * _UM)
        intensity[i] = np.abs(U) ** 2

    # Warning-only controls (no physical plane in this engine).
    for name in WARNING_ONLY_CONTROLS:
        if _flag(c, name):
            warnings.append(
                f"{name} is warning-only in the component-plane engine: {_WARNING_REASON.get(name, 'not modelled')}."
            )

    stack = PropagatedFieldStack(
        intensity_zyx=intensity,
        x_um=x_um,
        y_um=y_um,
        z_um=z_um,
        input_pulse_energy_uJ=E_in_nominal,
        sample_pulse_energy_uJ=sample_pulse_energy,
        transmitted_fraction=transmitted_fraction,
        plane_states=(input_state, slm_state, pupil_state, sample_entrance_state),
        warnings=tuple(warnings),
        metadata={
            "stage": "stage8c3r_component_plane",
            "ell": int(config.ell),
            "kr_rad_per_um": float(config.kr_rad_per_um),
            "n_medium": float(config.n_medium),
            "applied_components": list(applied_all),
        },
    )

    # Predicted steering from the input tilt ramp, consistent with the ASM.
    predicted = {"predicted_steering_x_mrad": 0.0, "predicted_steering_y_mrad": 0.0,
                 "predicted_shift_x_um_at_zmax": 0.0, "predicted_shift_y_um_at_zmax": 0.0}
    if _flag(c, "enable_beam_tilt"):
        k = config.k_medium_rad_per_m
        kx = float(meta_in.get("tilt_kx_rad_per_m", 0.0))
        ky = float(meta_in.get("tilt_ky_rad_per_m", 0.0))
        kz = np.sqrt(max(k**2 - kx**2 - ky**2, 1e-12))
        predicted["predicted_steering_x_mrad"] = float(np.arctan2(kx, kz) * 1000.0)
        predicted["predicted_steering_y_mrad"] = float(np.arctan2(ky, kz) * 1000.0)
        predicted["predicted_shift_x_um_at_zmax"] = float((kx / kz) * config.z_max_um)
        predicted["predicted_shift_y_um_at_zmax"] = float((ky / kz) * config.z_max_um)

    return ComponentPlaneRun(
        config=config,
        input_state=input_state,
        slm_state=slm_state,
        pupil_state=pupil_state,
        sample_entrance_state=sample_entrance_state,
        propagated_stack=stack,
        warnings=tuple(warnings),
        applied_components=tuple(applied_all),
        predicted_steering=predicted,
        metadata={"controls_applied": [k for k in c if _flag(c, k)]},
    )
