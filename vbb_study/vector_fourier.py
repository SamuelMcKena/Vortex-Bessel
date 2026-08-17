"""Carrier diagnostics and Fourier-plane signal-order iris helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from vbb_study.config import EPS
from vbb_study.equations.fields import fft2c, ifft2c
from vbb_study.slm_model import field_power

TWOPI = 2.0 * np.pi


def _grid_arrays(grid: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    if "FX" not in grid or "FY" not in grid:
        raise ValueError("grid must contain centred FX/FY frequency arrays.")
    FX = np.asarray(grid["FX"], dtype=float)
    FY = np.asarray(grid["FY"], dtype=float)
    if "X" in grid and "Y" in grid:
        X = np.asarray(grid["X"], dtype=float)
        Y = np.asarray(grid["Y"], dtype=float)
    else:
        x = np.asarray(grid["x"], dtype=float)
        y = np.asarray(grid.get("y", x), dtype=float)
        X, Y = np.meshgrid(x, y, indexing="xy")
    dx = float(grid["dx"])
    dy = float(grid.get("dy", dx))
    return X, Y, FX, FY, dx, dy


def _as_components(components: Iterable[np.ndarray] | np.ndarray) -> tuple[np.ndarray, ...]:
    if isinstance(components, np.ndarray):
        return (np.asarray(components, dtype=np.complex128),)
    out = tuple(np.asarray(component, dtype=np.complex128) for component in components)
    if not out:
        raise ValueError("At least one field component is required.")
    shape = out[0].shape
    if any(component.shape != shape for component in out):
        raise ValueError("All field components must share one shape.")
    return out


@dataclass(frozen=True)
class FourierPeak:
    """Peak location in the centred angular spectrum."""

    row: int
    col: int
    fx_cpm: float
    fy_cpm: float
    value: float
    diagnostic: str = "literal_peak"


@dataclass(frozen=True)
class CarrierCollinearity:
    """Peak separation report for two circular components."""

    plus_peak: FourierPeak
    minus_peak: FourierPeak
    separation_cpm: float
    separation_pixels: float
    frequency_pixel_cpm: float


@dataclass(frozen=True)
class IrisLedger:
    """Energy accounting for one Fourier-plane iris split."""

    incident_power: float
    signal_power: float
    blocked_power: float
    relative_error: float

    def as_dict(self) -> dict[str, float]:
        return {
            "incident_power": self.incident_power,
            "signal_power": self.signal_power,
            "blocked_power": self.blocked_power,
            "relative_error": self.relative_error,
        }


@dataclass(frozen=True)
class FourierIrisResult:
    """Fields and bookkeeping after a hard circular Fourier-plane iris."""

    signal: tuple[np.ndarray, ...]
    blocked: tuple[np.ndarray, ...]
    mask: np.ndarray
    ledger: IrisLedger
    residual_tilt_rad: tuple[float, float]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def locate_fourier_peak(field: np.ndarray, grid: Mapping[str, Any]) -> FourierPeak:
    """Locate the maximum spectral intensity of one component."""

    _, _, FX, FY, _, _ = _grid_arrays(grid)
    spectrum = np.abs(fft2c(np.asarray(field, dtype=complex))) ** 2
    row, col = np.unravel_index(int(np.nanargmax(spectrum)), spectrum.shape)
    return FourierPeak(
        row=int(row),
        col=int(col),
        fx_cpm=float(FX[row, col]),
        fy_cpm=float(FY[row, col]),
        value=float(spectrum[row, col]),
        diagnostic="literal_peak",
    )


def locate_fourier_centroid_peak(field: np.ndarray, grid: Mapping[str, Any]) -> FourierPeak:
    """Locate the carrier/order centre by spectral power centroid.

    A vortex-bearing order has an annular spectrum, so the brightest pixel is
    not a stable proxy for the carrier centre.  This returns the centroid as a
    ``FourierPeak``-shaped record so collinearity tests compare order centres.
    """

    _, _, FX, FY, _, _ = _grid_arrays(grid)
    spectrum = np.abs(fft2c(np.asarray(field, dtype=complex))) ** 2
    total = float(np.sum(spectrum))
    if total <= EPS:
        return locate_fourier_peak(field, grid)
    fx = float(np.sum(FX * spectrum) / total)
    fy = float(np.sum(FY * spectrum) / total)
    row = int(np.unravel_index(int(np.nanargmax(spectrum)), spectrum.shape)[0])
    col = int(np.unravel_index(int(np.nanargmax(spectrum)), spectrum.shape)[1])
    return FourierPeak(
        row=row,
        col=col,
        fx_cpm=fx,
        fy_cpm=fy,
        value=float(np.nanmax(spectrum)),
        diagnostic="spectral_centroid",
    )


def carrier_collinearity_report(
    plus_component: np.ndarray,
    minus_component: np.ndarray,
    grid: Mapping[str, Any],
) -> CarrierCollinearity:
    """Return Fourier-peak separation for two circular components."""

    _, _, FX, _, _, _ = _grid_arrays(grid)
    fx_line = np.asarray(FX[0, :], dtype=float)
    dfx = float(np.median(np.diff(fx_line))) if fx_line.size > 1 else 1.0
    plus = locate_fourier_centroid_peak(plus_component, grid)
    minus = locate_fourier_centroid_peak(minus_component, grid)
    sep = float(np.hypot(plus.fx_cpm - minus.fx_cpm, plus.fy_cpm - minus.fy_cpm))
    return CarrierCollinearity(
        plus_peak=plus,
        minus_peak=minus,
        separation_cpm=sep,
        separation_pixels=float(sep / max(abs(dfx), EPS)),
        frequency_pixel_cpm=abs(dfx),
    )


def fourier_iris_mask(
    grid: Mapping[str, Any],
    *,
    signal_fx_cpm: float,
    signal_fy_cpm: float = 0.0,
    iris_radius_frac: float,
) -> np.ndarray:
    """Return a hard circular iris centred on the selected diffraction order."""

    if not (0.0 < float(iris_radius_frac) <= 1.0):
        raise ValueError("iris_radius_frac must be in (0, 1].")
    _, _, FX, FY, _, _ = _grid_arrays(grid)
    separation = float(np.hypot(signal_fx_cpm, signal_fy_cpm))
    if separation <= EPS:
        raise ValueError("signal order must be separated from zero order.")
    radius = float(iris_radius_frac) * separation
    return (FX - float(signal_fx_cpm)) ** 2 + (FY - float(signal_fy_cpm)) ** 2 <= radius * radius


def remove_carrier(field: np.ndarray, grid: Mapping[str, Any], fx_cpm: float, fy_cpm: float = 0.0) -> np.ndarray:
    """Analytically remove a residual carrier tilt from a spatial field."""

    X, Y, _, _, _, _ = _grid_arrays(grid)
    return np.asarray(field, dtype=complex) * np.exp(-1j * TWOPI * (float(fx_cpm) * X + float(fy_cpm) * Y))


def spectral_centroid_tilt_rad(
    components: Sequence[np.ndarray],
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    medium_index: float = 1.0,
) -> tuple[float, float]:
    """Return spectral-centroid tilt angles for one or more components."""

    _, _, FX, FY, _, _ = _grid_arrays(grid)
    weight = np.zeros_like(FX, dtype=float)
    for component in components:
        weight += np.abs(fft2c(np.asarray(component, dtype=complex))) ** 2
    total = float(np.sum(weight))
    if total <= EPS:
        return 0.0, 0.0
    fx_bar = float(np.sum(FX * weight) / total)
    fy_bar = float(np.sum(FY * weight) / total)
    scale = float(wavelength_m) / max(float(medium_index), EPS)
    return float(np.arcsin(np.clip(scale * fx_bar, -1.0, 1.0))), float(
        np.arcsin(np.clip(scale * fy_bar, -1.0, 1.0))
    )


def apply_fourier_iris(
    components: Iterable[np.ndarray] | np.ndarray,
    grid: Mapping[str, Any],
    *,
    signal_fx_cpm: float,
    signal_fy_cpm: float = 0.0,
    iris_radius_frac: float,
    wavelength_m: float,
    medium_index: float = 1.0,
    remove_signal_carrier: bool = True,
    tilt_tolerance_rad: float = 1e-6,
) -> FourierIrisResult:
    """Split fields into signal and blocked orders with a hard Fourier iris."""

    comps = _as_components(components)
    mask = fourier_iris_mask(
        grid,
        signal_fx_cpm=signal_fx_cpm,
        signal_fy_cpm=signal_fy_cpm,
        iris_radius_frac=iris_radius_frac,
    )
    signal: list[np.ndarray] = []
    blocked: list[np.ndarray] = []
    for component in comps:
        spectrum = fft2c(component)
        sig = ifft2c(spectrum * mask)
        blk = ifft2c(spectrum * (~mask))
        if remove_signal_carrier:
            sig = remove_carrier(sig, grid, signal_fx_cpm, signal_fy_cpm)
        signal.append(np.asarray(sig, dtype=np.complex128))
        blocked.append(np.asarray(blk, dtype=np.complex128))

    p_in = float(sum(field_power(component, grid) for component in comps))
    p_signal = float(sum(field_power(component, grid) for component in signal))
    p_blocked = float(sum(field_power(component, grid) for component in blocked))
    rel = abs((p_signal + p_blocked) - p_in) / max(p_in, EPS)
    ledger = IrisLedger(
        incident_power=p_in,
        signal_power=p_signal,
        blocked_power=p_blocked,
        relative_error=float(rel),
    )
    assert ledger.relative_error < 1e-12

    tilt = spectral_centroid_tilt_rad(
        tuple(signal),
        grid,
        wavelength_m=wavelength_m,
        medium_index=medium_index,
    )
    residual = float(np.hypot(*tilt))
    assert residual < float(tilt_tolerance_rad)

    return FourierIrisResult(
        signal=tuple(signal),
        blocked=tuple(blocked),
        mask=mask,
        ledger=ledger,
        residual_tilt_rad=tilt,
        metadata={
            "signal_fx_cpm": float(signal_fx_cpm),
            "signal_fy_cpm": float(signal_fy_cpm),
            "iris_radius_frac": float(iris_radius_frac),
            "remove_signal_carrier": bool(remove_signal_carrier),
            "tilt_tolerance_rad": float(tilt_tolerance_rad),
        },
    )


def format_iris_ledger(rows: Mapping[str, float]) -> str:
    """Format a compact fraction ledger table for notebooks."""

    return "\n".join(f"{key:28s} {float(value):.12g}" for key, value in rows.items())


__all__ = [
    "CarrierCollinearity",
    "FourierIrisResult",
    "FourierPeak",
    "IrisLedger",
    "apply_fourier_iris",
    "carrier_collinearity_report",
    "format_iris_ledger",
    "fourier_iris_mask",
    "locate_fourier_centroid_peak",
    "locate_fourier_peak",
    "remove_carrier",
    "spectral_centroid_tilt_rad",
]
