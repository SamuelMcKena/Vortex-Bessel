"""Steep-axicon propagation for the Virtual Lab, following the bench digital twin.

A refractive axicon writes a conical phase whose radial period is
``2*pi / k_perp``.  For the bench optic that period is ~13 µm, far finer than a
working SLM-relay grid over a window wide enough to hold the beam, so a single
grid either clips the beam or aliases the cone.

This module ports the route that already works in
``Publication_Study/notebooks/experimental/axicon_aberration_correction/
real_bmg_digital_twin_correction.py`` (``propagate_route`` ->
``vbb_study.digital_twin.vortex_system_route.build_multirate_system_route``):

1. run the SLM relay on the working grid;
2. hand the (band-limited) field to a finer grid over the *same physical
   window* by Fourier zero-padding;
3. apply the axicon on the fine grid;
4. freeze one band-limited angular spectrum and evaluate each camera plane
   from it, so moving the camera costs a single inverse FFT.

The axicon is specified, as in that reference, by its measured transverse
wavenumber ``k_perp``.  The manufacturer's "20°" label is not a model base
angle: the measured k_perp implies a ~9.9° internal base angle in the
repository's exact refractive convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from vbb_study.digital_twin.vortex_error_reference_models import exact_refractive_axicon_kr_m_inv

TWOPI = 2.0 * math.pi

# Robust median of per-plane azimuthal radial fits across the 18-plane q=20
# BeamGage z-scan, from the bench digital-twin correction run.
MEASURED_BENCH_AXICON_K_PERP_M_INV = 482741.41419101483
MEASURED_BENCH_AXICON_SOURCE = (
    "measured: Publication_Study/notebooks/experimental/axicon_aberration_correction/outputs/"
    "digital_twin_correction/measured_k_perp_calibration.json (robust median radial fit, "
    "18 BeamGage q=20 z-scan planes)"
)

# Four field samples per cone period is two per intensity fringe (|E|² of the
# Bessel field repeats every pi/k_perp).  Below that the *intensity* aliases on
# the model grid itself and no detector model downstream can be trusted.  The
# reference fit ran at 2.67 (fitting field-derived quantities) and its
# production pass at 4.0.
DEFAULT_MIN_SAMPLES_PER_PERIOD = 4.0

# Gentec Beamage-4M: 2048 x 2048 pixels at 5.5 µm.
BEAMAGE_PIXEL_UM = 5.5
BEAMAGE_PIXELS = 2048


def fourier_resample_fixed_window(field: np.ndarray, output_n: int) -> np.ndarray:
    """Band-limit resample a square field without changing its physical window.

    Verbatim port of ``vbb_study.digital_twin.vortex_system_route.
    fourier_resample_fixed_window`` from Publication_Study, which this Virtual
    Lab snapshot of ``vbb_study`` predates.  Fourier zero-padding is the right
    handoff because the selected-order relay output is already band limited.
    """

    source = np.asarray(field, dtype=np.complex128)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("field must be a square 2D array")
    input_n = int(source.shape[0])
    output_n = int(output_n)
    if output_n < input_n:
        raise ValueError("output_n must be at least the input grid size")
    if output_n == input_n:
        return source.copy()
    spectrum = np.fft.fftshift(np.fft.fft2(source))
    # make_xy_grid uses cell-centred coordinates, so refining N shifts the first
    # coordinate by a fraction of an input pixel; correct that before padding.
    delta_samples = 0.5 * (float(input_n) / output_n - 1.0)
    frequency_cycles_per_sample = np.fft.fftshift(np.fft.fftfreq(input_n, d=1.0))
    fy, fx = np.meshgrid(frequency_cycles_per_sample, frequency_cycles_per_sample, indexing="ij")
    spectrum *= np.exp(1j * TWOPI * delta_samples * (fx + fy))
    padded = np.zeros((output_n, output_n), dtype=np.complex128)
    start = (output_n - input_n) // 2
    padded[start:start + input_n, start:start + input_n] = spectrum
    return np.fft.ifft2(np.fft.ifftshift(padded)) * (float(output_n) / input_n) ** 2


def k_perp_from_base_angle(
    base_angle_deg: float, *, wavelength_m: float, refractive_index: float, external_index: float,
) -> float:
    return float(
        exact_refractive_axicon_kr_m_inv(
            wavelength_m=float(wavelength_m),
            base_angle_rad=math.radians(float(base_angle_deg)),
            refractive_index=float(refractive_index),
            external_index=float(external_index),
        )
    )


def base_angle_deg_from_k_perp(
    k_perp_m_inv: float, *, wavelength_m: float, refractive_index: float, external_index: float,
) -> float:
    """Invert the exact refractive cone relation, as ``effective_axicon_from_kp`` does."""

    target = abs(float(k_perp_m_inv))

    def residual(deg: float) -> float:
        return k_perp_from_base_angle(
            deg, wavelength_m=wavelength_m, refractive_index=refractive_index, external_index=external_index,
        ) - target

    low, high = 0.05, 35.0
    try:
        f_low, f_high = residual(low), residual(high)
    except ValueError as exc:
        raise ValueError(f"k_perp inversion left the refractive branch: {exc}") from exc
    if f_low > 0.0 or f_high < 0.0:
        raise ValueError(
            f"k_perp={target:.4g} m^-1 is outside the range a {low:g}-{high:g}° refractive axicon can produce."
        )
    for _ in range(80):
        mid = 0.5 * (low + high)
        if residual(mid) < 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


@dataclass(frozen=True)
class AxiconPropagationPlan:
    valid: bool
    message: str
    k_perp_m_inv: float = float("nan")
    base_angle_deg: float = float("nan")
    cone_half_angle_deg: float = float("nan")
    radial_period_um: float = float("nan")
    relay_grid_n: int = 0
    propagation_grid_n: int = 0
    integration_factor: int = 1
    samples_per_period: float = float("nan")
    core_radius_um: float = float("nan")
    bessel_zone_mm: float = float("nan")

    def ring_radius_mm_at(self, z_mm: float) -> float:
        """Geometric annulus radius once the camera is past the Bessel zone."""
        return abs(float(z_mm)) * math.tan(math.radians(self.cone_half_angle_deg))

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def plan_axicon_propagation(
    geometry: Any,
    *,
    wavelength_m: float,
    relay_grid_n: int,
    beam_radius_mm: float,
    detector_n: int | None = None,
) -> AxiconPropagationPlan:
    """Decide how a given axicon must be sampled on a given working grid.

    The GUI and the engine both call this, so what the operator is told is
    always exactly what will be computed.
    """

    relay_grid_n = int(relay_grid_n)
    window_mm = float(geometry.simulation_window_mm)
    n_ax = float(geometry.axicon_refractive_index)
    n_ext = float(geometry.axicon_external_index)
    try:
        if geometry.axicon_k_perp_m_inv is not None:
            k_perp = abs(float(geometry.axicon_k_perp_m_inv))
            base_deg = base_angle_deg_from_k_perp(
                k_perp, wavelength_m=wavelength_m, refractive_index=n_ax, external_index=n_ext,
            )
        else:
            base_deg = float(geometry.axicon_model_base_angle_deg)
            k_perp = abs(k_perp_from_base_angle(
                base_deg, wavelength_m=wavelength_m, refractive_index=n_ax, external_index=n_ext,
            ))
    except ValueError as exc:
        return AxiconPropagationPlan(False, f"This axicon cannot be modelled: {exc}")
    if not k_perp > 0.0:
        return AxiconPropagationPlan(False, "The axicon transverse wavenumber must be positive.")

    k = TWOPI * n_ext / float(wavelength_m)
    cone_half_angle_deg = math.degrees(math.asin(min(1.0, k_perp / k)))
    period_um = TWOPI / k_perp * 1e6
    relay_dx_um = window_mm * 1000.0 / relay_grid_n
    minimum = float(getattr(geometry, "axicon_min_samples_per_period", DEFAULT_MIN_SAMPLES_PER_PERIOD))
    required_n = math.ceil(minimum * window_mm * 1000.0 / period_um)
    # The fine grid must be an exact multiple of the detector so each camera
    # pixel integrates whole model samples, and no coarser than the relay.
    detector_n = int(detector_n) if detector_n else relay_grid_n
    factor = max(1, math.ceil(max(required_n, relay_grid_n) / detector_n))
    fine_n = detector_n * factor
    samples = period_um / (window_mm * 1000.0 / fine_n)
    tan_beta = math.tan(math.radians(cone_half_angle_deg))
    common = dict(
        k_perp_m_inv=k_perp,
        base_angle_deg=base_deg,
        cone_half_angle_deg=cone_half_angle_deg,
        radial_period_um=period_um,
        relay_grid_n=relay_grid_n,
        core_radius_um=2.404825557695773 / k_perp * 1e6,
        bessel_zone_mm=float(beam_radius_mm) / tan_beta if tan_beta > 0 else float("inf"),
    )
    maximum = int(getattr(geometry, "max_propagation_grid_n", 4096))
    if fine_n > maximum:
        return AxiconPropagationPlan(
            False,
            f"k⊥ = {k_perp:.3g} m⁻¹ repeats every {period_um:.1f} µm and would need a "
            f"{fine_n}×{fine_n} axicon grid over the {window_mm:g} mm window (limit {maximum}). "
            "That is steeper than the bench optic; use the measured bench axicon or a smaller angle.",
            propagation_grid_n=fine_n,
            integration_factor=factor,
            samples_per_period=samples,
            **common,
        )
    return AxiconPropagationPlan(
        True,
        f"Relay {relay_grid_n}² ({relay_dx_um:.2f} µm) → axicon {fine_n}² ({period_um / samples:.2f} µm, "
        f"{samples:.1f} samples per {period_um:.1f} µm cone period) → camera {detector_n}² "
        f"({window_mm * 1000.0 / detector_n:.2f} µm pixels).",
        propagation_grid_n=fine_n,
        integration_factor=factor,
        samples_per_period=samples,
        **common,
    )


# ---------------------------------------------------------------------------
# Lean resolved-intensity stage
#
# The reference helpers build every 4096² coordinate/phase map in float64,
# which needs well over a gigabyte for one axicon step.  For the sharp-tip
# axicon the Virtual Lab uses, the transmission is exactly exp(-i k_perp r);
# the functions below compute the same fixed-window Fourier handoff,
# Matsushima-bandlimited angular spectrum and propagation in single precision
# with a multithreaded FFT.  ``tests`` checks them against the reference path.
# ---------------------------------------------------------------------------

import os

from scipy import fft as _sfft

_WORKERS = max(1, (os.cpu_count() or 2) - 1)


def _centred_axis(n: int, dx: float) -> np.ndarray:
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2 + 0.5) * float(dx)


def fourier_resample_fixed_window_fast(field: np.ndarray, output_n: int) -> np.ndarray:
    """Single-precision, multithreaded equivalent of ``fourier_resample_fixed_window``."""

    source = np.asarray(field)
    input_n = int(source.shape[0])
    output_n = int(output_n)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("field must be a square 2D array")
    if output_n < input_n:
        raise ValueError("output_n must be at least the input grid size")
    if output_n == input_n:
        return source.astype(np.complex64, copy=True)
    spectrum = _sfft.fftshift(_sfft.fft2(source.astype(np.complex128), workers=_WORKERS))
    delta_samples = 0.5 * (float(input_n) / output_n - 1.0)
    ramp = np.exp(1j * TWOPI * delta_samples * np.fft.fftshift(np.fft.fftfreq(input_n, d=1.0)))
    spectrum *= ramp[None, :]
    spectrum *= ramp[:, None]
    padded = np.zeros((output_n, output_n), dtype=np.complex64)
    start = (output_n - input_n) // 2
    padded[start:start + input_n, start:start + input_n] = spectrum
    del spectrum
    out = _sfft.ifft2(_sfft.ifftshift(padded), workers=_WORKERS)
    out *= np.float32((float(output_n) / input_n) ** 2)
    return out.astype(np.complex64, copy=False)


@dataclass
class ResolvedAxiconSpectrum:
    spectrum: np.ndarray  # complex64, centred, band-limited
    kz: np.ndarray  # float32 propagating longitudinal wavenumber (rad/m)
    n: int
    dx_m: float
    retained_spectral_power_fraction: float


def build_resolved_axicon_spectrum(
    field_on_axicon: np.ndarray,
    *,
    window_m: float,
    fine_n: int,
    wavelength_m: float,
    k_perp_m_inv: float,
    decentre_m: tuple[float, float],
    z_support_m: float,
    minimum_retained_spectral_power: float = 0.98,
) -> ResolvedAxiconSpectrum:
    """Fourier handoff, sharp refractive cone, and one frozen band-limited spectrum."""

    if int(fine_n) % 2:
        raise ValueError("the resolved axicon grid must have an even number of samples")
    fine = fourier_resample_fixed_window_fast(field_on_axicon, fine_n)
    dx = float(window_m) / int(fine_n)
    axis = _centred_axis(fine_n, dx)
    x = (axis - float(decentre_m[0])).astype(np.float32)
    y = (axis - float(decentre_m[1])).astype(np.float32)
    phase = np.hypot(x[None, :], y[:, None])
    phase *= np.float32(-abs(float(k_perp_m_inv)))
    fine *= np.exp(1j * phase).astype(np.complex64, copy=False)
    del phase
    # For an even grid the centring shifts of fftshift(fft2(ifftshift(.))) and
    # fftshift(ifft2(ifftshift(.))) cancel exactly, so the spectrum is kept in
    # plain FFT order: each camera plane is one inverse FFT with no roll copies.
    spectrum = _sfft.fft2(fine, workers=_WORKERS, overwrite_x=True).astype(np.complex64, copy=False)
    del fine

    freq = np.fft.fftfreq(int(fine_n), d=dx)
    du = 1.0 / (int(fine_n) * dx)
    u_lim = 1.0 / (float(wavelength_m) * math.sqrt((2.0 * du * abs(float(z_support_m))) ** 2 + 1.0))
    keep = (np.abs(freq) <= u_lim)
    power = spectrum.real ** 2 + spectrum.imag ** 2
    total = float(power.sum(dtype=np.float64))
    retained = float(power[np.ix_(keep, keep)].sum(dtype=np.float64))
    del power
    fraction = retained / max(total, 1e-300)
    if fraction < float(minimum_retained_spectral_power):
        raise RuntimeError(
            "fixed z-sweep angular-spectrum support removes appreciable source spectrum: "
            f"retained={fraction:.8f} < required={minimum_retained_spectral_power:.8f}; increase window/sampling"
        )
    spectrum[~keep, :] = 0
    spectrum[:, ~keep] = 0
    fsq = (freq ** 2).astype(np.float32)
    arg = np.float32(TWOPI ** 2) * (np.float32(1.0 / float(wavelength_m) ** 2) - fsq[None, :] - fsq[:, None])
    np.maximum(arg, 0.0, out=arg)
    kz = np.sqrt(arg, out=arg)
    return ResolvedAxiconSpectrum(spectrum, kz, int(fine_n), dx, fraction)


def resolved_field_at_z(propagator: ResolvedAxiconSpectrum, z_m: float) -> np.ndarray:
    phase = propagator.kz * np.float32(z_m)
    transfer = np.empty(phase.shape, dtype=np.complex64)
    np.cos(phase, out=transfer.real)
    np.sin(phase, out=transfer.imag)
    del phase
    transfer *= propagator.spectrum
    return _sfft.ifft2(transfer, workers=_WORKERS, overwrite_x=True)


def detector_intensity(field: np.ndarray, factor: int) -> np.ndarray:
    """|E|² area-integrated over square detector pixels of ``factor`` samples."""

    intensity = field.real.astype(np.float32) ** 2 + field.imag.astype(np.float32) ** 2
    factor = int(factor)
    if factor > 1:
        n = intensity.shape[0] // factor
        intensity = intensity.reshape(n, factor, n, factor).mean(axis=(1, 3))
    return intensity
