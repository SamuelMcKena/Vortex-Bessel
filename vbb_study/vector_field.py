"""Three-component vector fields and angular-spectrum propagation.

Convention
----------
The repository scalar angular-spectrum kernel uses the physics convention
``exp(+i k z - i omega t)`` and a forward propagation factor
``exp(+i kz z)``.  This module matches that sign convention.

The upstream vector-arm optics are paraxial and are represented by transverse
Jones fields.  ``propagate_vector_asm`` applies the spectral transversality
projector once at vector-propagation entry; this is where ``Ez`` is generated
from transverse components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from vbb_study.config import EPS
from vbb_study.equations.fields import fft2c, ifft2c
from vbb_study.equations.vector_jones import (
    linear_to_circular,
    stokes_from_linear_components,
)

TWOPI = 2.0 * np.pi


def _as_complex_2d(values: Any, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array.")
    return arr


def _coerce_grid(grid: Mapping[str, Any], shape: tuple[int, int]) -> dict[str, Any]:
    out = dict(grid)
    ny, nx = int(shape[0]), int(shape[1])
    if "dx" not in out:
        if "x" not in out:
            raise ValueError("grid must provide either 'dx' or 'x'.")
        x = np.asarray(out["x"], dtype=float)
        if x.size < 2:
            raise ValueError("grid['x'] must contain at least two samples.")
        out["dx"] = float(np.median(np.diff(x)))
    dx = float(out["dx"])
    if "x" not in out:
        out["x"] = (np.arange(nx) - nx / 2 + 0.5) * dx
    if "y" not in out:
        out["y"] = (np.arange(ny) - ny / 2 + 0.5) * dx
    if "FX" not in out or "FY" not in out:
        fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
        dy = float(out.get("dy", dx))
        fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
        out["FX"], out["FY"] = np.meshgrid(fx, fy, indexing="xy")
    out.setdefault("N", nx if nx == ny else None)
    return out


@dataclass(frozen=True)
class VectorField:
    """A sampled complex vector field in the fixed lab x/y/z basis."""

    ex: np.ndarray
    ey: np.ndarray
    ez: np.ndarray | None = None
    grid: Mapping[str, Any] = field(default_factory=dict)
    wavelength_m: float = 1029e-9
    medium_index: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ex = _as_complex_2d(self.ex, "ex")
        ey = _as_complex_2d(self.ey, "ey")
        if ey.shape != ex.shape:
            raise ValueError("ex and ey must have the same shape.")
        ez = np.zeros_like(ex, dtype=np.complex128) if self.ez is None else _as_complex_2d(self.ez, "ez")
        if ez.shape != ex.shape:
            raise ValueError("ez must have the same shape as ex and ey.")
        if float(self.wavelength_m) <= 0.0:
            raise ValueError("wavelength_m must be positive.")
        if float(self.medium_index) <= 0.0:
            raise ValueError("medium_index must be positive.")
        object.__setattr__(self, "ex", ex)
        object.__setattr__(self, "ey", ey)
        object.__setattr__(self, "ez", ez)
        object.__setattr__(self, "grid", _coerce_grid(self.grid, ex.shape))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def Ex(self) -> np.ndarray:
        """Compatibility alias for existing SurfaceField-style consumers."""

        return self.ex

    @property
    def Ey(self) -> np.ndarray:
        """Compatibility alias for existing SurfaceField-style consumers."""

        return self.ey

    @property
    def Ez(self) -> np.ndarray:
        """Compatibility alias for existing SurfaceField-style consumers."""

        return self.ez

    @property
    def intensity(self) -> np.ndarray:
        """Return ``|Ex|^2 + |Ey|^2 + |Ez|^2``."""

        return np.abs(self.ex) ** 2 + np.abs(self.ey) ** 2 + np.abs(self.ez) ** 2

    @property
    def power(self) -> float:
        """Return the discrete transverse power integral."""

        dx = float(self.grid["dx"])
        dy = float(self.grid.get("dy", dx))
        return float(np.sum(self.intensity) * dx * dy)

    def stokes(self) -> dict[str, np.ndarray]:
        """Return transverse Stokes fields from ``Ex`` and ``Ey`` only."""

        return stokes_from_linear_components(self.ex, self.ey)

    def circular_components(self) -> tuple[np.ndarray, np.ndarray]:
        """Return circular-basis components ``(E_plus, E_minus)``.

        The basis is ``e_plus = (x + i y)/sqrt(2)`` and
        ``e_minus = (x - i y)/sqrt(2)``.  With this convention
        ``E_plus = (Ex - i Ey)/sqrt(2)`` and
        ``E_minus = (Ex + i Ey)/sqrt(2)``.
        """

        return linear_to_circular(self.ex, self.ey)

    def replace(
        self,
        *,
        ex: np.ndarray | None = None,
        ey: np.ndarray | None = None,
        ez: np.ndarray | None = None,
        grid: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "VectorField":
        """Return a copy with selected field arrays or metadata replaced."""

        return VectorField(
            ex=self.ex if ex is None else ex,
            ey=self.ey if ey is None else ey,
            ez=self.ez if ez is None else ez,
            grid=self.grid if grid is None else grid,
            wavelength_m=self.wavelength_m,
            medium_index=self.medium_index,
            metadata=self.metadata if metadata is None else metadata,
        )


def _spectral_axes(field: VectorField) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    grid = field.grid
    k = TWOPI * float(field.medium_index) / float(field.wavelength_m)
    kx = TWOPI * np.asarray(grid["FX"], dtype=float)
    ky = TWOPI * np.asarray(grid["FY"], dtype=float)
    if kx.shape != field.ex.shape or ky.shape != field.ex.shape:
        raise ValueError("grid frequency arrays must match the field shape.")
    kz = np.sqrt((k * k - kx * kx - ky * ky) + 0j)
    kz = np.where(np.imag(kz) < 0.0, -kz, kz)
    return kx, ky, kz, k


def _project_transverse_spectrum(
    ax: np.ndarray,
    ay: np.ndarray,
    az: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx = kx / max(float(k), EPS)
    sy = ky / max(float(k), EPS)
    sz = kz / max(float(k), EPS)
    dot = sx * ax + sy * ay + sz * az
    return ax - sx * dot, ay - sy * dot, az - sz * dot


def propagate_vector_asm(field: VectorField, z_m: float) -> VectorField:
    """Propagate a vector field by centred-FFT angular spectrum.

    The spectral projector ``P = I - s s^T`` is applied once before propagation
    and written componentwise so no per-pixel 3x3 matrix is built.  Evanescent
    components use the same ``exp(i kz z)`` transfer factor as the formula in
    the Stage 7 contract; for positive ``z`` their positive imaginary ``kz``
    decays.
    """

    kx, ky, kz, k = _spectral_axes(field)
    ax = fft2c(field.ex)
    ay = fft2c(field.ey)
    az = fft2c(field.ez)
    ax_p, ay_p, az_p = _project_transverse_spectrum(ax, ay, az, kx, ky, kz, k)
    transfer = np.exp(1j * kz * float(z_m))
    out = VectorField(
        ex=ifft2c(ax_p * transfer),
        ey=ifft2c(ay_p * transfer),
        ez=ifft2c(az_p * transfer),
        grid=field.grid,
        wavelength_m=field.wavelength_m,
        medium_index=field.medium_index,
        metadata={**dict(field.metadata), "vector_asm_z_m": float(z_m)},
    )
    return out


def spectral_transversality_residual(field: VectorField) -> float:
    """Return max normalized ``|k dot E|`` in the angular spectrum."""

    kx, ky, kz, k = _spectral_axes(field)
    ax = fft2c(field.ex)
    ay = fft2c(field.ey)
    az = fft2c(field.ez)
    residual = np.abs(kx * ax + ky * ay + kz * az)
    denom = max(float(k), EPS) * max(
        float(np.nanmax(np.sqrt(np.abs(ax) ** 2 + np.abs(ay) ** 2 + np.abs(az) ** 2))),
        EPS,
    )
    return float(np.nanmax(residual) / denom)


__all__ = ["VectorField", "propagate_vector_asm", "spectral_transversality_residual"]
