"""Scalar propagation checks for the short-Bessel inverse design.

This module deliberately sits one level below the full laboratory digital twin.
It takes the *sample-equivalent* conical field implied by ``short_bessel`` and
propagates it in a homogeneous medium with the repository's band-limited
angular-spectrum propagator.

It is useful for answering a focused question before the real 4F + physical
axicon calibration is available:

    Does the radial wavevector inferred from a target core, together with the
    beam radius inferred from a target finite overlap length, actually produce
    the expected micron-scale transverse feature and a longer axial Bessel zone
    under scalar propagation?

It does NOT claim that an SLM command already produces this field on the bench.
The SLM, 4F order-selection optics, physical axicon and objective are collapsed
into the equivalent conical field at z=0.  Those components must be propagated
explicitly before this becomes a calibrated bench prediction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from vbb_study.equations.fields import gaussian_amplitude, make_xy_grid
from vbb_study.equations.propagation import (
    discrete_power,
    make_bl_asm_propagator,
    samples_per_radial_period,
)
from vbb_study.short_bessel import ShortBesselDesignResult, geometric_bessel_length_m


UM = 1.0e-6
MM = 1.0e-3
EPS = 1.0e-30


@dataclass(frozen=True)
class ShortBesselPropagationConfig:
    """Numerical settings for the equivalent sample-plane propagation check."""

    grid_n: int = 513
    dx_um: float = 0.25
    z_min_um: float = 0.0
    z_max_um: float = 700.0
    z_points: int = 141
    bandlimit: bool = True
    include_evanescent: bool = True

    def validated(self) -> "ShortBesselPropagationConfig":
        if int(self.grid_n) < 129:
            raise ValueError("grid_n must be >= 129 for the default short-Bessel check")
        if float(self.dx_um) <= 0.0:
            raise ValueError("dx_um must be positive")
        if int(self.z_points) < 9:
            raise ValueError("z_points must be >= 9")
        if float(self.z_max_um) <= float(self.z_min_um):
            raise ValueError("z_max_um must exceed z_min_um")
        return self


@dataclass(frozen=True)
class PropagationMetrics:
    case_id: str
    sample_beam_radius_um: float
    geometric_reference_length_um: float
    peak_z_um: float
    transverse_fwhm_at_peak_um: float
    on_axis_halfmax_zone_start_um: float
    on_axis_halfmax_zone_end_um: float
    on_axis_halfmax_zone_length_um: float
    samples_per_radial_period: float
    input_power_arb: float
    min_power_ratio: float
    max_power_ratio: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PropagationCase:
    case_id: str
    x_um: np.ndarray
    z_um: np.ndarray
    input_field: np.ndarray
    xz_intensity: np.ndarray
    on_axis_intensity: np.ndarray
    power_ratio: np.ndarray
    peak_xy_intensity: np.ndarray
    peak_field: np.ndarray
    snapshots_xy_intensity: dict[float, np.ndarray]
    metrics: PropagationMetrics


@dataclass
class ShortBesselPropagationComparison:
    design: ShortBesselDesignResult
    config: ShortBesselPropagationConfig
    target: PropagationCase
    current: PropagationCase
    claim_boundary: str = (
        "Equivalent sample-plane scalar propagation only. The SLM, 4F order selection, "
        "physical axicon, objective and sample interface are not explicitly propagated."
    )

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "design": self.design.as_dict(),
            "propagation_config": asdict(self.config),
            "target": self.target.metrics.as_dict(),
            "current": self.current.metrics.as_dict(),
            "claim_boundary": self.claim_boundary,
        }


def equivalent_conical_field(
    design: ShortBesselDesignResult,
    *,
    sample_beam_radius_m: float,
    grid: dict[str, Any],
) -> np.ndarray:
    """Build ``Gaussian(r; w) * exp(-i*k_r*r)`` at the equivalent axicon plane."""

    radius = float(sample_beam_radius_m)
    if radius <= 0.0:
        raise ValueError("sample_beam_radius_m must be positive")
    amp = gaussian_amplitude(np.asarray(grid["R"], dtype=float), radius)
    phase = np.exp(-1j * float(design.target_kr_sample_m_inv) * np.asarray(grid["R"], dtype=float))
    return np.asarray(amp * phase, dtype=complex)


def _crossing(x0: float, y0: float, x1: float, y1: float, level: float) -> float:
    if abs(float(y1) - float(y0)) <= EPS:
        return 0.5 * (float(x0) + float(x1))
    t = (float(level) - float(y0)) / (float(y1) - float(y0))
    return float(x0) + float(np.clip(t, 0.0, 1.0)) * (float(x1) - float(x0))


def _fwhm_width(x_um: np.ndarray, intensity: np.ndarray) -> float:
    x = np.asarray(x_um, dtype=float)
    y = np.maximum(np.asarray(intensity, dtype=float), 0.0)
    if y.size != x.size or not np.any(np.isfinite(y)) or float(np.nanmax(y)) <= 0.0:
        return float("nan")
    y = y / float(np.nanmax(y))
    c = int(np.nanargmax(y))
    left = c
    while left > 0 and y[left] >= 0.5:
        left -= 1
    right = c
    while right < y.size - 1 and y[right] >= 0.5:
        right += 1
    if left == c or right == c:
        return float("nan")
    xl = _crossing(x[left], y[left], x[left + 1], y[left + 1], 0.5)
    xr = _crossing(x[right - 1], y[right - 1], x[right], y[right], 0.5)
    return float(xr - xl)


def _halfmax_zone(z_um: np.ndarray, on_axis: np.ndarray) -> tuple[float, float, float]:
    z = np.asarray(z_um, dtype=float)
    y = np.maximum(np.asarray(on_axis, dtype=float), 0.0)
    if float(np.max(y)) <= 0.0:
        return float("nan"), float("nan"), float("nan")
    y = y / float(np.max(y))
    peak = int(np.argmax(y))
    left = peak
    while left > 0 and y[left] >= 0.5:
        left -= 1
    right = peak
    while right < y.size - 1 and y[right] >= 0.5:
        right += 1

    if left == peak:
        z0 = float(z[peak])
    elif y[left] < 0.5:
        z0 = _crossing(z[left], y[left], z[left + 1], y[left + 1], 0.5)
    else:
        z0 = float(z[left])

    if right == peak:
        z1 = float(z[peak])
    elif y[right] < 0.5:
        z1 = _crossing(z[right - 1], y[right - 1], z[right], y[right], 0.5)
    else:
        z1 = float(z[right])
    return float(z0), float(z1), float(max(z1 - z0, 0.0))


def propagate_equivalent_case(
    design: ShortBesselDesignResult,
    *,
    case_id: str,
    sample_beam_radius_m: float,
    config: ShortBesselPropagationConfig | None = None,
    snapshot_z_um: Iterable[float] = (0.0, 100.0, 200.0, 300.0, 400.0, 500.0),
) -> PropagationCase:
    """Propagate one equivalent conical field through the sample medium."""

    cfg = (config or ShortBesselPropagationConfig()).validated()
    dx_m = float(cfg.dx_um) * UM
    grid = make_xy_grid(int(cfg.grid_n), dx_m)
    x_um = np.asarray(grid["x"], dtype=float) / UM
    z_um = np.linspace(float(cfg.z_min_um), float(cfg.z_max_um), int(cfg.z_points))
    z_m = z_um * UM

    U0 = equivalent_conical_field(design, sample_beam_radius_m=sample_beam_radius_m, grid=grid)
    wavelength_m = float(design.inputs.wavelength_nm) * 1.0e-9
    n_medium = float(design.inputs.sample_index)
    prop = make_bl_asm_propagator(
        U0,
        grid,
        wavelength_m,
        n_medium=n_medium,
        bandlimit=bool(cfg.bandlimit),
        include_evanescent=bool(cfg.include_evanescent),
    )

    centre = int(np.argmin(np.abs(x_um)))
    xz = np.empty((z_um.size, x_um.size), dtype=np.float32)
    on_axis = np.empty(z_um.size, dtype=float)
    power = np.empty(z_um.size, dtype=float)
    fields_at_requested: dict[int, np.ndarray] = {}
    requested_indices = {
        int(np.argmin(np.abs(z_um - float(value)))): float(z_um[int(np.argmin(np.abs(z_um - float(value))))])
        for value in snapshot_z_um
    }

    peak_value = -np.inf
    peak_field = None
    peak_index = 0
    input_power = discrete_power(U0, dx_m)

    for i, zz in enumerate(z_m):
        U = np.asarray(prop(float(zz)), dtype=complex)
        I = np.abs(U) ** 2
        xz[i] = np.asarray(I[centre, :], dtype=np.float32)
        on_axis[i] = float(I[centre, centre])
        power[i] = discrete_power(U, dx_m) / max(input_power, EPS)
        if i in requested_indices:
            fields_at_requested[i] = np.asarray(I, dtype=np.float32)
        if on_axis[i] > peak_value:
            peak_value = float(on_axis[i])
            peak_index = int(i)
            peak_field = U.copy()

    assert peak_field is not None
    peak_xy = np.asarray(np.abs(peak_field) ** 2, dtype=np.float32)
    fwhm = _fwhm_width(x_um, peak_xy[centre, :])
    z0, z1, zone = _halfmax_zone(z_um, on_axis)

    k0 = 2.0 * np.pi / wavelength_m
    k_medium = k0 * n_medium
    geometric = geometric_bessel_length_m(
        float(sample_beam_radius_m),
        k_medium_m_inv=k_medium,
        kr_m_inv=float(design.target_kr_sample_m_inv),
    )

    metrics = PropagationMetrics(
        case_id=str(case_id),
        sample_beam_radius_um=float(sample_beam_radius_m / UM),
        geometric_reference_length_um=float(geometric / UM),
        peak_z_um=float(z_um[peak_index]),
        transverse_fwhm_at_peak_um=float(fwhm),
        on_axis_halfmax_zone_start_um=float(z0),
        on_axis_halfmax_zone_end_um=float(z1),
        on_axis_halfmax_zone_length_um=float(zone),
        samples_per_radial_period=float(samples_per_radial_period(dx_m, design.target_kr_sample_m_inv)),
        input_power_arb=float(input_power),
        min_power_ratio=float(np.min(power)),
        max_power_ratio=float(np.max(power)),
    )
    snapshots = {requested_indices[i]: fields_at_requested[i] for i in sorted(fields_at_requested)}

    return PropagationCase(
        case_id=str(case_id),
        x_um=x_um,
        z_um=z_um,
        input_field=U0,
        xz_intensity=xz,
        on_axis_intensity=on_axis,
        power_ratio=power,
        peak_xy_intensity=peak_xy,
        peak_field=peak_field,
        snapshots_xy_intensity=snapshots,
        metrics=metrics,
    )


def run_short_bessel_propagation(
    design: ShortBesselDesignResult,
    *,
    config: ShortBesselPropagationConfig | None = None,
    snapshot_z_um: Iterable[float] = (0.0, 100.0, 200.0, 300.0, 400.0, 500.0),
) -> ShortBesselPropagationComparison:
    """Compare the target beam radius with the current repository beam radius."""

    cfg = (config or ShortBesselPropagationConfig()).validated()
    target_radius_m = float(design.required_sample_beam_radius_um) * UM
    current_radius_m = (
        float(design.current_pre_beam_radius_mm) * MM * float(design.relay_demag)
    )

    target = propagate_equivalent_case(
        design,
        case_id="target_radius",
        sample_beam_radius_m=target_radius_m,
        config=cfg,
        snapshot_z_um=snapshot_z_um,
    )
    current = propagate_equivalent_case(
        design,
        case_id="current_radius",
        sample_beam_radius_m=current_radius_m,
        config=cfg,
        snapshot_z_um=snapshot_z_um,
    )
    return ShortBesselPropagationComparison(design=design, config=cfg, target=target, current=current)


__all__ = [
    "PropagationCase",
    "PropagationMetrics",
    "ShortBesselPropagationComparison",
    "ShortBesselPropagationConfig",
    "equivalent_conical_field",
    "propagate_equivalent_case",
    "run_short_bessel_propagation",
]
