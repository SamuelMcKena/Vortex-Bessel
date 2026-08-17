"""Explicit 4F relay for physical error studies.

Unlike the historical collapsed ``FFT -> circular mask -> IFFT`` contract, this
module propagates through two physical thin-lens planes and a physical
Fourier-plane iris.  It supports lens/iris despace, lens decentre, lens rigid
plane tilt, finite aperture and measured/prescribed lens OPD maps.

The fixed Fourier iris is centred on the nominal selected +1 diffraction order,
not on the optical axis.  ``FourFError.iris_offset_m`` is an error relative to
that fixed nominal centre.

Rigid lens tilt is handled by rotating the angular spectrum onto the actual lens
plane, applying the thin-lens phase/aperture in local coordinates, and rotating
back.  This is a scalar/paraxial thin-lens model; strongly tilted thick lenses
still require surface-by-surface vector refraction for absolute prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vbb_study.digital_twin.vortex_rotated_plane import lab_to_tilted_plane, tilted_to_lab_plane
from vbb_study.equations.propagation import angular_spectrum_propagate_bl


EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class LensError:
    focal_length_scale: float = 1.0
    decentre_m: tuple[float, float] = (0.0, 0.0)
    tilt_rad: tuple[float, float] = (0.0, 0.0)
    clear_radius_m: float | None = None

    def validate(self) -> None:
        if self.focal_length_scale <= 0.0:
            raise ValueError("lens focal_length_scale must be positive")
        if self.clear_radius_m is not None and self.clear_radius_m <= 0.0:
            raise ValueError("lens clear radius must be positive")
        if np.hypot(*map(float, self.tilt_rad)) >= np.deg2rad(10.0):
            raise ValueError("scalar/paraxial tilted-lens model is not authorised at >=10 deg")


@dataclass(frozen=True)
class FourFError:
    """Errors of a nominal symmetric 4F relay with fixed object/iris/output planes."""

    lens1: LensError = LensError()
    lens2: LensError = LensError()
    lens1_axial_shift_m: float = 0.0
    lens2_axial_shift_m: float = 0.0
    iris_offset_m: tuple[float, float] = (0.0, 0.0)
    iris_radius_scale: float = 1.0

    def validate(self, nominal_focal_length_m: float) -> None:
        self.lens1.validate()
        self.lens2.validate()
        if self.iris_radius_scale <= 0.0:
            raise ValueError("iris_radius_scale must be positive")
        f = float(nominal_focal_length_m)
        distances = (
            f + float(self.lens1_axial_shift_m),
            f - float(self.lens1_axial_shift_m),
            f + float(self.lens2_axial_shift_m),
            f - float(self.lens2_axial_shift_m),
        )
        if min(distances) <= 0.0:
            raise ValueError("4F lens shift produces a non-positive propagation distance")


def field_power(field: np.ndarray, grid: Mapping[str, Any]) -> float:
    dx = float(grid["dx"])
    return float(np.sum(np.abs(np.asarray(field, dtype=np.complex128)) ** 2) * dx * dx)


def thin_lens_transmission(
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    focal_length_m: float,
    decentre_m: tuple[float, float] = (0.0, 0.0),
    clear_radius_m: float | None = None,
    opd_map_m: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Ideal paraxial thin lens in its own local plane."""

    X = np.asarray(grid["X"], dtype=float) - float(decentre_m[0])
    Y = np.asarray(grid["Y"], dtype=float) - float(decentre_m[1])
    k0 = TWOPI / float(wavelength_m)
    phase = -k0 * (X * X + Y * Y) / (2.0 * float(focal_length_m))
    transmission = np.exp(1j * phase)
    if clear_radius_m is not None:
        transmission *= (X * X + Y * Y <= float(clear_radius_m) ** 2)
    if opd_map_m is not None:
        opd = np.asarray(opd_map_m, dtype=float)
        if opd.shape != transmission.shape:
            raise ValueError("lens OPD map shape does not match simulation grid")
        transmission *= np.exp(1j * k0 * opd)
        opd_status = "measured_or_user_supplied"
    else:
        opd_status = "none"
    return np.asarray(transmission, dtype=np.complex128), {
        "focal_length_m": float(focal_length_m),
        "decentre_m": tuple(map(float, decentre_m)),
        "clear_radius_m": None if clear_radius_m is None else float(clear_radius_m),
        "opd_map_status": opd_status,
    }


def _apply_lens_plane(
    field_lab: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    focal_length_m: float,
    error: LensError,
    opd_map_m: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    tx, ty = map(float, error.tilt_rad)
    if tx != 0.0 or ty != 0.0:
        local_field, to_meta = lab_to_tilted_plane(
            field_lab,
            grid,
            wavelength_m=wavelength_m,
            tilt_x_rad=tx,
            tilt_y_rad=ty,
        )
    else:
        local_field = np.asarray(field_lab, dtype=np.complex128)
        to_meta = {"fidelity": "identity_parallel_plane", "spectral_clipped_fraction": 0.0}

    lens_t, lens_meta = thin_lens_transmission(
        grid,
        wavelength_m=wavelength_m,
        focal_length_m=focal_length_m,
        decentre_m=error.decentre_m,
        clear_radius_m=error.clear_radius_m,
        opd_map_m=opd_map_m,
    )
    local_after = local_field * lens_t

    if tx != 0.0 or ty != 0.0:
        returned, from_meta = tilted_to_lab_plane(
            local_after,
            grid,
            wavelength_m=wavelength_m,
            tilt_x_rad=tx,
            tilt_y_rad=ty,
        )
        tilt_status = "scalar_rotated_angular_spectrum_thin_lens"
    else:
        returned = local_after
        from_meta = {"fidelity": "identity_parallel_plane", "spectral_clipped_fraction": 0.0}
        tilt_status = "none"

    return np.asarray(returned, dtype=np.complex128), {
        **lens_meta,
        "tilt_rad": (tx, ty),
        "tilt_status": tilt_status,
        "lab_to_lens_plane": to_meta,
        "lens_plane_to_lab": from_meta,
        "tilt_fidelity": (
            "scalar_paraxial_rotated_plane"
            if tilt_status != "none"
            else "parallel_plane"
        ),
    }


def physical_iris(
    grid: Mapping[str, Any],
    *,
    radius_m: float,
    centre_m: tuple[float, float],
) -> np.ndarray:
    X = np.asarray(grid["X"], dtype=float) - float(centre_m[0])
    Y = np.asarray(grid["Y"], dtype=float) - float(centre_m[1])
    return (X * X + Y * Y <= float(radius_m) ** 2).astype(float)


def nominal_order_position_m(
    *,
    wavelength_m: float,
    focal_length_m: float,
    carrier_cpm: float,
) -> tuple[float, float]:
    sx = float(wavelength_m) * float(carrier_cpm)
    if abs(sx) >= 1.0:
        raise ValueError("nominal carrier order is non-propagating")
    sz = np.sqrt(1.0 - sx * sx)
    return float(focal_length_m) * sx / sz, 0.0


def _propagate(
    field: np.ndarray,
    grid: Mapping[str, Any],
    wavelength_m: float,
    distance_m: float,
) -> np.ndarray:
    if abs(float(distance_m)) < 1e-15:
        return np.asarray(field, dtype=np.complex128)
    return np.asarray(
        angular_spectrum_propagate_bl(
            np.asarray(field, dtype=np.complex128),
            dict(grid),
            float(wavelength_m),
            float(distance_m),
            n_medium=1.0,
            bandlimit=True,
            include_evanescent=True,
        ),
        dtype=np.complex128,
    )


def explicit_4f_relay(
    field_in: np.ndarray,
    grid: Mapping[str, Any],
    *,
    wavelength_m: float,
    nominal_focal_length_m: float,
    nominal_iris_radius_m: float,
    nominal_carrier_cpm: float = 0.0,
    error: FourFError = FourFError(),
    lens1_opd_map_m: np.ndarray | None = None,
    lens2_opd_map_m: np.ndarray | None = None,
) -> dict[str, Any]:
    """Propagate object -> L1 -> fixed physical +1 iris -> L2 -> output."""

    error.validate(float(nominal_focal_length_m))
    f = float(nominal_focal_length_m)
    d_obj_l1 = f + float(error.lens1_axial_shift_m)
    d_l1_iris = f - float(error.lens1_axial_shift_m)
    d_iris_l2 = f + float(error.lens2_axial_shift_m)
    d_l2_out = f - float(error.lens2_axial_shift_m)

    f1 = f * float(error.lens1.focal_length_scale)
    f2 = f * float(error.lens2.focal_length_scale)
    u0 = np.asarray(field_in, dtype=np.complex128)
    p0 = field_power(u0, grid)

    pre_l1 = _propagate(u0, grid, wavelength_m, d_obj_l1)
    post_l1, lens1_meta = _apply_lens_plane(
        pre_l1,
        grid,
        wavelength_m=wavelength_m,
        focal_length_m=f1,
        error=error.lens1,
        opd_map_m=lens1_opd_map_m,
    )
    pre_iris = _propagate(post_l1, grid, wavelength_m, d_l1_iris)

    nominal_centre = nominal_order_position_m(
        wavelength_m=wavelength_m,
        focal_length_m=f,
        carrier_cpm=nominal_carrier_cpm,
    )
    iris_centre = (
        nominal_centre[0] + float(error.iris_offset_m[0]),
        nominal_centre[1] + float(error.iris_offset_m[1]),
    )
    iris_radius = float(nominal_iris_radius_m) * float(error.iris_radius_scale)
    iris = physical_iris(grid, radius_m=iris_radius, centre_m=iris_centre)
    pre_iris_power = field_power(pre_iris, grid)
    post_iris = pre_iris * iris
    selected_fraction = field_power(post_iris, grid) / max(pre_iris_power, EPS)

    pre_l2 = _propagate(post_iris, grid, wavelength_m, d_iris_l2)
    post_l2, lens2_meta = _apply_lens_plane(
        pre_l2,
        grid,
        wavelength_m=wavelength_m,
        focal_length_m=f2,
        error=error.lens2,
        opd_map_m=lens2_opd_map_m,
    )
    output = _propagate(post_l2, grid, wavelength_m, d_l2_out)

    return {
        "input": u0,
        "pre_lens1": pre_l1,
        "post_lens1": post_l1,
        "fourier_plane_before_iris": pre_iris,
        "iris_mask": iris,
        "fourier_plane_after_iris": post_iris,
        "pre_lens2": pre_l2,
        "post_lens2": post_l2,
        "output": output,
        "metadata": {
            "model": "explicit_4f_ASM_with_rotated_thin_lens_planes",
            "nominal_focal_length_m": f,
            "nominal_carrier_cpm": float(nominal_carrier_cpm),
            "nominal_selected_order_centre_m": tuple(map(float, nominal_centre)),
            "physical_iris_centre_m": tuple(map(float, iris_centre)),
            "distances_m": {
                "object_to_lens1": d_obj_l1,
                "lens1_to_iris": d_l1_iris,
                "iris_to_lens2": d_iris_l2,
                "lens2_to_output": d_l2_out,
            },
            "lens1": lens1_meta,
            "lens2": lens2_meta,
            "iris_radius_m": iris_radius,
            "iris_error_offset_m": tuple(map(float, error.iris_offset_m)),
            "iris_selected_power_fraction": float(selected_fraction),
            "input_power": float(p0),
            "output_power": float(field_power(output, grid)),
        },
    }
