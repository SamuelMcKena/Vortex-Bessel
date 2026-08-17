"""Calibrated source-plane -> objective entrance-pupil mapping.

The structured-beam source model and the vector Debye objective model live at
different physical planes.  This bridge makes that separation explicit.

1. Propagate the post-axicon complex field over the measured free-space distance
   with the audited band-limited angular-spectrum propagator.
2. Apply an optional calibrated affine imaging map (magnification, rotation,
   translation) onto a declared objective-pupil grid.
3. Apply the measured/effective pupil aperture and report power capture.

The affine field amplitude includes the square-root Jacobian required for power
conservation in an ideal lossless coordinate magnification.  This is an ideal
relay mapping, not a replacement for explicit lens propagation when lens
prescriptions and separations are available.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import angular_spectrum_propagate_bl, discrete_power


EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class ObjectivePupilMappingConfig:
    free_space_distance_m: float
    output_window_m: float
    output_n: int
    pupil_radius_m: float
    magnification_x: float = 1.0
    magnification_y: float = 1.0
    rotation_rad: float = 0.0
    offset_x_m: float = 0.0
    offset_y_m: float = 0.0
    bandlimit: bool = True

    def validate(self) -> None:
        if not np.isfinite(self.free_space_distance_m):
            raise ValueError("free_space_distance_m must be finite")
        if self.output_window_m <= 0.0 or int(self.output_n) < 32:
            raise ValueError("output_window_m must be positive and output_n >=32")
        if self.pupil_radius_m <= 0.0:
            raise ValueError("pupil_radius_m must be positive")
        if self.magnification_x == 0.0 or self.magnification_y == 0.0:
            raise ValueError("relay magnifications cannot be zero")
        if not all(
            np.isfinite(v)
            for v in (
                self.magnification_x,
                self.magnification_y,
                self.rotation_rad,
                self.offset_x_m,
                self.offset_y_m,
            )
        ):
            raise ValueError("objective-pupil affine parameters must be finite")


@dataclass(frozen=True)
class ObjectivePupilMappingResult:
    field: np.ndarray
    grid: Mapping[str, Any]
    field_before_pupil: np.ndarray
    pupil_mask: np.ndarray
    metadata: Mapping[str, float | str | bool]


def _axis_from_grid(grid: Mapping[str, Any], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.asarray(grid["x"], dtype=float)
    y = np.asarray(grid.get("y", x), dtype=float)
    if shape != (y.size, x.size):
        raise ValueError("field shape does not match source grid")
    if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
        raise ValueError("source axes must increase")
    dx = float(np.median(np.diff(x)))
    dy = float(np.median(np.diff(y)))
    if not np.allclose(np.diff(x), dx, rtol=1e-9, atol=1e-15) or not np.allclose(
        np.diff(y), dy, rtol=1e-9, atol=1e-15
    ):
        raise ValueError("source grid must be uniform")
    return x, y, dx, dy


def _complex_interpolator(y: np.ndarray, x: np.ndarray, field: np.ndarray) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    return (
        RegularGridInterpolator((y, x), np.real(field), bounds_error=False, fill_value=0.0),
        RegularGridInterpolator((y, x), np.imag(field), bounds_error=False, fill_value=0.0),
    )


def map_post_axicon_to_objective_pupil(
    field: np.ndarray,
    source_grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    config: ObjectivePupilMappingConfig,
) -> ObjectivePupilMappingResult:
    """Propagate and map a complex source field to an objective entrance pupil."""

    config.validate()
    source = np.asarray(field, dtype=np.complex128)
    x, y, dx, dy = _axis_from_grid(source_grid, source.shape)
    if not np.isclose(dx, dy, rtol=1e-9, atol=1e-15):
        raise ValueError("current BL-ASM bridge requires square source sampling")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive")

    if abs(float(config.free_space_distance_m)) > 0.0:
        propagated = angular_spectrum_propagate_bl(
            source,
            dict(source_grid),
            float(wavelength_m),
            float(config.free_space_distance_m),
            n_medium=1.0,
            bandlimit=bool(config.bandlimit),
            include_evanescent=True,
        )
    else:
        propagated = source.copy()

    target_grid = make_xy_grid(int(config.output_n), float(config.output_window_m) / int(config.output_n))
    Xt = np.asarray(target_grid["X"], dtype=float)
    Yt = np.asarray(target_grid["Y"], dtype=float)

    # Forward coordinate convention:
    #   r_target = R(theta) diag(Mx,My) r_source + offset.
    # Invert it to sample the source field at each target coordinate.
    X0 = Xt - float(config.offset_x_m)
    Y0 = Yt - float(config.offset_y_m)
    c = math.cos(float(config.rotation_rad))
    s = math.sin(float(config.rotation_rad))
    xr = c * X0 + s * Y0
    yr = -s * X0 + c * Y0
    Xs = xr / float(config.magnification_x)
    Ys = yr / float(config.magnification_y)

    re_interp, im_interp = _complex_interpolator(y, x, propagated)
    query = np.column_stack([Ys.ravel(), Xs.ravel()])
    sampled = (
        re_interp(query).reshape(Xt.shape)
        + 1j * im_interp(query).reshape(Xt.shape)
    )
    jacobian_area = abs(float(config.magnification_x) * float(config.magnification_y))
    sampled = sampled / math.sqrt(max(jacobian_area, EPS))
    pupil = np.hypot(Xt, Yt) <= float(config.pupil_radius_m)
    output = np.where(pupil, sampled, 0.0j)

    source_power = float(np.sum(np.abs(source) ** 2) * dx * dy)
    propagated_power = float(np.sum(np.abs(propagated) ** 2) * dx * dy)
    target_dx = float(target_grid["dx"])
    mapped_power = discrete_power(sampled, target_dx)
    pupil_power = discrete_power(output, target_dx)
    return ObjectivePupilMappingResult(
        field=np.asarray(output, dtype=np.complex128),
        grid=target_grid,
        field_before_pupil=np.asarray(sampled, dtype=np.complex128),
        pupil_mask=np.asarray(pupil, dtype=bool),
        metadata={
            "model": "measured_free_space_BL_ASM_plus_calibrated_ideal_affine_relay",
            "free_space_distance_m": float(config.free_space_distance_m),
            "magnification_x": float(config.magnification_x),
            "magnification_y": float(config.magnification_y),
            "rotation_rad": float(config.rotation_rad),
            "offset_x_m": float(config.offset_x_m),
            "offset_y_m": float(config.offset_y_m),
            "source_power_au": source_power,
            "propagated_power_au": propagated_power,
            "propagation_power_ratio": propagated_power / max(source_power, EPS),
            "mapped_power_au": mapped_power,
            "mapped_to_propagated_power_ratio": mapped_power / max(propagated_power, EPS),
            "pupil_power_au": pupil_power,
            "pupil_capture_fraction_of_mapped": pupil_power / max(mapped_power, EPS),
            "affine_mapping_jacobian_area": jacobian_area,
            "absolute_bench_requirement": "free-space distance and relay affine mapping must be measured; explicit lenses supersede ideal affine relay when prescriptions are known",
        },
    )


__all__ = [
    "ObjectivePupilMappingConfig",
    "ObjectivePupilMappingResult",
    "map_post_axicon_to_objective_pupil",
]
