"""Short-Bessel inverse design for a programmable axicon + physical axicon route.

This module answers a deliberately narrow design question:

    Given a desired J0 Bessel core FWHM and a desired finite Bessel-Gauss
    overlap length, what radial wavevector, beam radius, objective NA and SLM
    radial phase slope are required?

It also supports a practical lab workflow in which the physical-axicon-only
Bessel core is first measured.  The measured core is converted to a
sample-equivalent radial wavevector and the SLM is asked to provide only the
signed residual radial wavevector needed to reach the target.

Important claim boundary
------------------------
The algebra here is an inverse-design / feasibility model.  It does *not* make
a calibrated prediction for the current laboratory 4F + physical-axicon train.
The real 4F coordinate mapping, physical axicon cone parameter, SLM calibration,
filter aperture and downstream objective must still be measured/bound before a
bench prediction is claimed.

The phase convention used for the programmable radial term is

    phi_slm(r) = -k_r,slm * r

so positive ``k_r,slm`` adds conical phase in the same sign convention as the
repository's scalar axicon field.  A negative value represents partial
cancellation of the physical-axicon contribution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Iterable

import numpy as np
from scipy import optimize, special

from vbb_study.config import LaserConfig, ObjectiveConfig, RelayConfig, SLMConfig


EPS = 1.0e-30
TWOPI = 2.0 * math.pi
UM = 1.0e-6
MM = 1.0e-3

J0_FIRST_ZERO = float(special.jn_zeros(0, 1)[0])
J0_HALF_INTENSITY_ARGUMENT = float(
    optimize.brentq(
        lambda x: float(special.j0(x) ** 2 - 0.5),
        0.0,
        J0_FIRST_ZERO,
    )
)


def _positive(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}")
    return out


def kr_from_j0_fwhm_diameter_m(fwhm_diameter_m: float) -> float:
    """Return ``k_r`` from the full-width-at-half-maximum of ``J0^2(k_r r)``.

    If ``x_1/2`` solves ``J0(x_1/2)^2 = 1/2``, then

    ``D_FWHM = 2*x_1/2/k_r``.
    """

    diameter = _positive(fwhm_diameter_m, "fwhm_diameter_m")
    return 2.0 * J0_HALF_INTENSITY_ARGUMENT / diameter


def j0_fwhm_diameter_from_kr_m(kr_m_inv: float) -> float:
    """Return the J0 intensity FWHM diameter corresponding to ``k_r``."""

    kr = _positive(abs(float(kr_m_inv)), "abs(kr_m_inv)")
    return 2.0 * J0_HALF_INTENSITY_ARGUMENT / kr


def first_zero_diameter_from_kr_m(kr_m_inv: float) -> float:
    """Return the equivalent ell=0 first-zero diameter used by the repository."""

    kr = _positive(abs(float(kr_m_inv)), "abs(kr_m_inv)")
    return 2.0 * J0_FIRST_ZERO / kr


def kr_from_first_zero_diameter_m(first_zero_diameter_m: float) -> float:
    """Return ``k_r`` from the repository's equivalent J0 first-zero diameter."""

    diameter = _positive(first_zero_diameter_m, "first_zero_diameter_m")
    return 2.0 * J0_FIRST_ZERO / diameter


def required_sample_radius_for_length_m(
    bessel_length_m: float,
    *,
    k_medium_m_inv: float,
    kr_m_inv: float,
) -> float:
    """Return the Gaussian 1/e field radius for ``z_ref = w0*k/k_r``."""

    length = _positive(bessel_length_m, "bessel_length_m")
    k_medium = _positive(k_medium_m_inv, "k_medium_m_inv")
    kr = _positive(abs(float(kr_m_inv)), "abs(kr_m_inv)")
    return length * kr / k_medium


def geometric_bessel_length_m(
    sample_beam_radius_m: float,
    *,
    k_medium_m_inv: float,
    kr_m_inv: float,
) -> float:
    """Return the finite Bessel-Gauss overlap reference ``w0*k/k_r``."""

    radius = _positive(sample_beam_radius_m, "sample_beam_radius_m")
    k_medium = _positive(k_medium_m_inv, "k_medium_m_inv")
    kr = _positive(abs(float(kr_m_inv)), "abs(kr_m_inv)")
    return radius * k_medium / kr


def programmable_axicon_phase_rad(R_m: Any, signed_kr_m_inv: float) -> np.ndarray:
    """Return a signed programmable conical phase ``-k_r*r`` in radians."""

    return -float(signed_kr_m_inv) * np.asarray(R_m, dtype=float)


def wrap_phase_rad(phase_rad: Any) -> np.ndarray:
    """Wrap phase to ``[0, 2*pi)``."""

    return np.mod(np.asarray(phase_rad, dtype=float), TWOPI)


def phase_to_uint8(phase_rad: Any) -> np.ndarray:
    """Convert wrapped phase to an 8-bit 0..255 command map."""

    wrapped = wrap_phase_rad(phase_rad)
    levels = np.floor(wrapped / TWOPI * 256.0 + 0.5).astype(np.int64) % 256
    return levels.astype(np.uint8)


@dataclass(frozen=True)
class ShortBesselDesignInput:
    """Inputs for a target-driven short-Bessel feasibility calculation."""

    target_fwhm_um: float = 5.0
    target_length_um: float = 500.0
    wavelength_nm: float = LaserConfig().wavelength_m * 1.0e9
    sample_index: float = 1.45

    # The repository's fixed-optics default is objective f_eff / effective relay f.
    relay_demag: float = ObjectiveConfig().f_eff_m / RelayConfig().effective_relay_f_m
    objective_na: float = ObjectiveConfig().NA

    slm_pixel_pitch_um: float = SLMConfig().pixel_pitch_m / UM
    slm_resolution_x: int = SLMConfig().resolution_x
    slm_resolution_y: int = SLMConfig().resolution_y
    slm_safe_radius_fraction: float = 0.90
    minimum_pixels_per_phase_wrap: float = 4.0

    current_pre_beam_radius_mm: float = LaserConfig().beam_radius_on_slm_m / MM

    # Optional lab measurement.  When provided, the code infers the physical
    # axicon's sample-equivalent k_r from its measured J0 core FWHM and solves
    # only for the signed SLM residual required to reach the target.
    measured_physical_fwhm_um: float | None = None

    def validated(self) -> "ShortBesselDesignInput":
        _positive(self.target_fwhm_um, "target_fwhm_um")
        _positive(self.target_length_um, "target_length_um")
        _positive(self.wavelength_nm, "wavelength_nm")
        _positive(self.sample_index, "sample_index")
        _positive(self.relay_demag, "relay_demag")
        _positive(self.objective_na, "objective_na")
        _positive(self.slm_pixel_pitch_um, "slm_pixel_pitch_um")
        _positive(float(self.slm_resolution_x), "slm_resolution_x")
        _positive(float(self.slm_resolution_y), "slm_resolution_y")
        safe = float(self.slm_safe_radius_fraction)
        if not 0.0 < safe <= 1.0:
            raise ValueError("slm_safe_radius_fraction must lie in (0, 1].")
        _positive(self.minimum_pixels_per_phase_wrap, "minimum_pixels_per_phase_wrap")
        _positive(self.current_pre_beam_radius_mm, "current_pre_beam_radius_mm")
        if self.measured_physical_fwhm_um is not None:
            _positive(self.measured_physical_fwhm_um, "measured_physical_fwhm_um")
        return self


@dataclass(frozen=True)
class ShortBesselDesignResult:
    """Dimensionally explicit output of :func:`design_short_bessel`."""

    inputs: ShortBesselDesignInput
    target_kr_sample_m_inv: float
    target_first_zero_diameter_um: float
    target_fwhm_um: float
    target_length_um: float
    target_cone_angle_in_sample_deg: float
    target_air_equivalent_cone_angle_deg: float
    required_objective_na: float
    objective_na_available: float
    na_feasible: bool

    required_sample_beam_radius_um: float
    required_pre_beam_radius_mm: float
    current_pre_beam_radius_mm: float
    predicted_length_with_current_beam_um: float
    required_beam_radius_scale_factor: float

    relay_demag: float
    target_kr_pre_m_inv: float
    target_phase_wrap_period_um: float
    target_pixels_per_phase_wrap: float

    slm_safe_radius_mm: float
    slm_aperture_margin_mm: float
    aperture_feasible: bool
    target_phase_sampling_feasible: bool

    physical_kr_sample_m_inv: float | None
    residual_slm_kr_sample_m_inv: float | None
    residual_slm_kr_pre_m_inv: float | None
    residual_phase_wrap_period_um: float | None
    residual_pixels_per_phase_wrap: float | None
    residual_phase_sign: str | None
    residual_phase_sampling_feasible: bool | None

    hard_feasible: bool
    claim_boundary: str

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["inputs"] = asdict(self.inputs)
        return data

    @property
    def command_kr_pre_m_inv(self) -> float:
        """Return the SLM k_r command to use for a phase-mask preview.

        If a physical-axicon-only FWHM was measured, use the signed residual.
        Otherwise return the full target pre-plane k_r and label it as a
        target-only design, not a combined-system command.
        """

        if self.residual_slm_kr_pre_m_inv is not None:
            return float(self.residual_slm_kr_pre_m_inv)
        return float(self.target_kr_pre_m_inv)

    @property
    def command_mode(self) -> str:
        return "physical_measured_residual" if self.residual_slm_kr_pre_m_inv is not None else "target_only_no_physical_calibration"


def _period_and_sampling(signed_kr_m_inv: float, pixel_pitch_um: float) -> tuple[float, float]:
    kr_abs = abs(float(signed_kr_m_inv))
    if kr_abs <= EPS:
        return float("inf"), float("inf")
    period_m = TWOPI / kr_abs
    period_um = period_m / UM
    return period_um, period_um / float(pixel_pitch_um)


def design_short_bessel(inputs: ShortBesselDesignInput | None = None) -> ShortBesselDesignResult:
    """Design a finite J0 Bessel target and audit first-order feasibility."""

    cfg = (inputs or ShortBesselDesignInput()).validated()

    wavelength_m = float(cfg.wavelength_nm) * 1.0e-9
    k0 = TWOPI / wavelength_m
    k_medium = k0 * float(cfg.sample_index)

    target_fwhm_m = float(cfg.target_fwhm_um) * UM
    target_length_m = float(cfg.target_length_um) * UM
    kr_sample = kr_from_j0_fwhm_diameter_m(target_fwhm_m)
    first_zero_um = first_zero_diameter_from_kr_m(kr_sample) / UM

    sample_ratio = kr_sample / k_medium
    air_ratio = kr_sample / k0
    if sample_ratio >= 1.0:
        raise ValueError("target k_r exceeds the sample-medium propagating-wave limit")
    if air_ratio >= 1.0:
        air_angle = float("nan")
    else:
        air_angle = math.degrees(math.asin(air_ratio))

    sample_angle = math.degrees(math.asin(sample_ratio))
    required_na = kr_sample / k0
    na_feasible = required_na <= float(cfg.objective_na) + 1.0e-12

    required_sample_radius_m = required_sample_radius_for_length_m(
        target_length_m,
        k_medium_m_inv=k_medium,
        kr_m_inv=kr_sample,
    )
    required_pre_radius_m = required_sample_radius_m / float(cfg.relay_demag)

    current_pre_radius_m = float(cfg.current_pre_beam_radius_mm) * MM
    current_sample_radius_m = current_pre_radius_m * float(cfg.relay_demag)
    predicted_current_length_m = geometric_bessel_length_m(
        current_sample_radius_m,
        k_medium_m_inv=k_medium,
        kr_m_inv=kr_sample,
    )
    radius_scale = required_pre_radius_m / current_pre_radius_m

    target_kr_pre = kr_sample * float(cfg.relay_demag)
    target_period_um, target_pixels_per_wrap = _period_and_sampling(
        target_kr_pre,
        cfg.slm_pixel_pitch_um,
    )
    target_sampling_feasible = target_pixels_per_wrap >= float(cfg.minimum_pixels_per_phase_wrap)

    active_half_height_mm = 0.5 * float(cfg.slm_resolution_y) * float(cfg.slm_pixel_pitch_um) * 1.0e-3
    safe_radius_mm = active_half_height_mm * float(cfg.slm_safe_radius_fraction)
    required_pre_radius_mm = required_pre_radius_m / MM
    margin_mm = safe_radius_mm - required_pre_radius_mm
    aperture_feasible = margin_mm >= 0.0

    physical_kr: float | None = None
    residual_sample: float | None = None
    residual_pre: float | None = None
    residual_period_um: float | None = None
    residual_pixels: float | None = None
    residual_sign: str | None = None
    residual_sampling: bool | None = None

    if cfg.measured_physical_fwhm_um is not None:
        physical_kr = kr_from_j0_fwhm_diameter_m(float(cfg.measured_physical_fwhm_um) * UM)
        residual_sample = kr_sample - physical_kr
        residual_pre = residual_sample * float(cfg.relay_demag)
        residual_period_um, residual_pixels = _period_and_sampling(residual_pre, cfg.slm_pixel_pitch_um)
        if residual_sample > 1.0e-12:
            residual_sign = "add_radial_wavevector"
        elif residual_sample < -1.0e-12:
            residual_sign = "subtract_radial_wavevector"
        else:
            residual_sign = "no_radial_correction_needed"
        residual_sampling = residual_pixels >= float(cfg.minimum_pixels_per_phase_wrap)

    sampling_gate = target_sampling_feasible if residual_sampling is None else residual_sampling
    hard_feasible = bool(na_feasible and aperture_feasible and sampling_gate)

    return ShortBesselDesignResult(
        inputs=cfg,
        target_kr_sample_m_inv=float(kr_sample),
        target_first_zero_diameter_um=float(first_zero_um),
        target_fwhm_um=float(cfg.target_fwhm_um),
        target_length_um=float(cfg.target_length_um),
        target_cone_angle_in_sample_deg=float(sample_angle),
        target_air_equivalent_cone_angle_deg=float(air_angle),
        required_objective_na=float(required_na),
        objective_na_available=float(cfg.objective_na),
        na_feasible=bool(na_feasible),
        required_sample_beam_radius_um=float(required_sample_radius_m / UM),
        required_pre_beam_radius_mm=float(required_pre_radius_mm),
        current_pre_beam_radius_mm=float(cfg.current_pre_beam_radius_mm),
        predicted_length_with_current_beam_um=float(predicted_current_length_m / UM),
        required_beam_radius_scale_factor=float(radius_scale),
        relay_demag=float(cfg.relay_demag),
        target_kr_pre_m_inv=float(target_kr_pre),
        target_phase_wrap_period_um=float(target_period_um),
        target_pixels_per_phase_wrap=float(target_pixels_per_wrap),
        slm_safe_radius_mm=float(safe_radius_mm),
        slm_aperture_margin_mm=float(margin_mm),
        aperture_feasible=bool(aperture_feasible),
        target_phase_sampling_feasible=bool(target_sampling_feasible),
        physical_kr_sample_m_inv=None if physical_kr is None else float(physical_kr),
        residual_slm_kr_sample_m_inv=None if residual_sample is None else float(residual_sample),
        residual_slm_kr_pre_m_inv=None if residual_pre is None else float(residual_pre),
        residual_phase_wrap_period_um=None if residual_period_um is None else float(residual_period_um),
        residual_pixels_per_phase_wrap=None if residual_pixels is None else float(residual_pixels),
        residual_phase_sign=residual_sign,
        residual_phase_sampling_feasible=residual_sampling,
        hard_feasible=hard_feasible,
        claim_boundary=(
            "inverse-design feasibility only; target-matched scalar J0/Bessel-Gauss algebra; "
            "not a calibrated 4F + physical-axicon laboratory prediction"
        ),
    )


def slm_phase_preview(
    result: ShortBesselDesignResult,
    *,
    use_full_resolution: bool = True,
) -> dict[str, Any]:
    """Render the radial phase command implied by a design result.

    When no physical-only FWHM measurement was supplied, this is the *full target
    SLM phase* and not the residual command for a combined physical-axicon setup.
    """

    cfg = result.inputs
    nx = int(cfg.slm_resolution_x if use_full_resolution else min(cfg.slm_resolution_x, 640))
    ny = int(cfg.slm_resolution_y if use_full_resolution else min(cfg.slm_resolution_y, 360))
    pitch_m = float(cfg.slm_pixel_pitch_um) * UM
    x = (np.arange(nx, dtype=float) - 0.5 * (nx - 1)) * pitch_m
    y = (np.arange(ny, dtype=float) - 0.5 * (ny - 1)) * pitch_m
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    phase_continuous = programmable_axicon_phase_rad(R, result.command_kr_pre_m_inv)
    phase_wrapped = wrap_phase_rad(phase_continuous)
    return {
        "x_m": x,
        "y_m": y,
        "phase_continuous_rad": phase_continuous,
        "phase_wrapped_rad": phase_wrapped,
        "gray_uint8": phase_to_uint8(phase_wrapped),
        "signed_kr_pre_m_inv": float(result.command_kr_pre_m_inv),
        "mode": result.command_mode,
        "claim_boundary": result.claim_boundary,
    }


def design_sweep(
    fwhm_values_um: Iterable[float],
    length_values_um: Iterable[float],
    *,
    base: ShortBesselDesignInput | None = None,
) -> list[dict[str, Any]]:
    """Return a rectangular target sweep as JSON/CSV-friendly rows."""

    base_cfg = base or ShortBesselDesignInput()
    rows: list[dict[str, Any]] = []
    for fwhm_um in fwhm_values_um:
        for length_um in length_values_um:
            result = design_short_bessel(
                replace(
                    base_cfg,
                    target_fwhm_um=float(fwhm_um),
                    target_length_um=float(length_um),
                )
            )
            rows.append(
                {
                    "target_fwhm_um": result.target_fwhm_um,
                    "target_length_um": result.target_length_um,
                    "target_first_zero_diameter_um": result.target_first_zero_diameter_um,
                    "target_kr_sample_m_inv": result.target_kr_sample_m_inv,
                    "required_objective_na": result.required_objective_na,
                    "required_sample_beam_radius_um": result.required_sample_beam_radius_um,
                    "required_pre_beam_radius_mm": result.required_pre_beam_radius_mm,
                    "required_beam_radius_scale_factor": result.required_beam_radius_scale_factor,
                    "target_kr_pre_m_inv": result.target_kr_pre_m_inv,
                    "target_phase_wrap_period_um": result.target_phase_wrap_period_um,
                    "target_pixels_per_phase_wrap": result.target_pixels_per_phase_wrap,
                    "slm_aperture_margin_mm": result.slm_aperture_margin_mm,
                    "na_feasible": result.na_feasible,
                    "aperture_feasible": result.aperture_feasible,
                    "phase_sampling_feasible": (
                        result.target_phase_sampling_feasible
                        if result.residual_phase_sampling_feasible is None
                        else result.residual_phase_sampling_feasible
                    ),
                    "hard_feasible": result.hard_feasible,
                }
            )
    return rows


__all__ = [
    "J0_FIRST_ZERO",
    "J0_HALF_INTENSITY_ARGUMENT",
    "ShortBesselDesignInput",
    "ShortBesselDesignResult",
    "design_short_bessel",
    "design_sweep",
    "first_zero_diameter_from_kr_m",
    "geometric_bessel_length_m",
    "j0_fwhm_diameter_from_kr_m",
    "kr_from_first_zero_diameter_m",
    "kr_from_j0_fwhm_diameter_m",
    "phase_to_uint8",
    "programmable_axicon_phase_rad",
    "required_sample_radius_for_length_m",
    "slm_phase_preview",
    "wrap_phase_rad",
]
