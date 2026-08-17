"""Independent vectorial Debye/Richards-Wolf objective reference.

The input is a transverse Jones field sampled in the entrance-pupil plane.
The default ``sqrt_cosine`` convention treats that sampled field as the
collimated pre-objective field; an ideal aplanatic sine-condition objective
therefore contributes ``sqrt(cos(theta))`` to the angular amplitude.

The implementation deliberately does not call the repository scalar focal
FFT.  It resamples the Cartesian pupil onto a polar Gauss-Legendre/trapezoid
quadrature and evaluates the solid-angle integral in bounded output chunks.
Its raw amplitude is a deterministic relative reference: the quadrature
weights are physical, but the omitted objective prefactor means it must not
be inserted into the Phase 2A absolute energy ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np


TWOPI = 2.0 * np.pi
EPS = np.finfo(float).tiny


@dataclass(frozen=True)
class DebyeConfig:
    """Configuration for an aplanatic vectorial objective reference."""

    wavelength_m: float
    refractive_index: float
    numerical_aperture: float
    focal_length_m: float
    pupil_radius_m: float
    apodisation: Literal["sqrt_cosine", "none"] = "sqrt_cosine"
    propagation_direction: Literal["+z", "-z"] = "+z"
    backend: Literal["polar_quadrature", "cartesian_fft"] = "polar_quadrature"
    quadrature_order_r: int = 256
    quadrature_order_phi: int = 720
    max_output_points: int = 1024
    fft_pad_factor: int = 1

    def validate(self) -> None:
        if not np.isfinite(self.wavelength_m) or self.wavelength_m <= 0.0:
            raise ValueError("wavelength_m must be finite and positive")
        if not np.isfinite(self.refractive_index) or self.refractive_index <= 0.0:
            raise ValueError("refractive_index must be finite and positive")
        if not 0.0 < self.numerical_aperture <= self.refractive_index:
            raise ValueError("numerical_aperture must lie in (0, refractive_index]")
        if not np.isfinite(self.focal_length_m) or self.focal_length_m <= 0.0:
            raise ValueError("focal_length_m must be finite and positive")
        if not np.isfinite(self.pupil_radius_m) or self.pupil_radius_m <= 0.0:
            raise ValueError("pupil_radius_m must be finite and positive")
        if self.apodisation not in {"sqrt_cosine", "none"}:
            raise ValueError("unsupported apodisation convention")
        if self.propagation_direction not in {"+z", "-z"}:
            raise ValueError("propagation_direction must be '+z' or '-z'")
        if self.backend not in {"polar_quadrature", "cartesian_fft"}:
            raise ValueError("unsupported Debye backend")
        if int(self.quadrature_order_r) < 4 or int(self.quadrature_order_phi) < 8:
            raise ValueError("quadrature orders are too small")
        if int(self.max_output_points) < 1:
            raise ValueError("max_output_points must be positive")
        if int(self.fft_pad_factor) < 1:
            raise ValueError("fft_pad_factor must be a positive integer")


@dataclass
class VectorFieldPlane:
    """Three-component vector field on one transverse output plane."""

    x_m: np.ndarray
    y_m: np.ndarray
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def intensity(self) -> np.ndarray:
        return np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2 + np.abs(self.Ez) ** 2

    @property
    def transverse_intensity(self) -> np.ndarray:
        return np.abs(self.Ex) ** 2 + np.abs(self.Ey) ** 2

    @property
    def component_power_fractions(self) -> dict[str, float]:
        powers = np.asarray(
            [np.sum(np.abs(self.Ex) ** 2), np.sum(np.abs(self.Ey) ** 2), np.sum(np.abs(self.Ez) ** 2)],
            dtype=float,
        )
        total = max(float(np.sum(powers)), EPS)
        return {
            "Ex_power_fraction": float(powers[0] / total),
            "Ey_power_fraction": float(powers[1] / total),
            "Ez_power_fraction": float(powers[2] / total),
        }


@dataclass(frozen=True)
class _AngularQuadrature:
    sample_x_m: np.ndarray
    sample_y_m: np.ndarray
    sin_theta: np.ndarray
    cos_theta: np.ndarray
    cos_phi: np.ndarray
    sin_phi: np.ndarray
    solid_angle_weight: np.ndarray
    theta_max_rad: float


def _axis(values: np.ndarray, name: str, *, require_uniform: bool) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be a finite one-dimensional axis")
    delta = np.diff(axis)
    if np.any(delta <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if require_uniform and not np.allclose(delta, delta[0], rtol=1e-9, atol=1e-15):
        raise ValueError(f"{name} must be uniformly sampled")
    return axis


def _complex_bilinear(
    values: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample one complex field on a regular Cartesian grid."""

    field_values = np.asarray(values, dtype=np.complex128)
    if field_values.shape != (y_axis.size, x_axis.size):
        raise ValueError("pupil field shape does not match pupil axes")
    dx = float(x_axis[1] - x_axis[0])
    dy = float(y_axis[1] - y_axis[0])
    ux = (np.asarray(xq, dtype=float) - float(x_axis[0])) / dx
    uy = (np.asarray(yq, dtype=float) - float(y_axis[0])) / dy
    ix = np.floor(ux).astype(np.int64)
    iy = np.floor(uy).astype(np.int64)
    valid = (ix >= 0) & (iy >= 0) & (ix < x_axis.size - 1) & (iy < y_axis.size - 1)
    ix_safe = np.clip(ix, 0, x_axis.size - 2)
    iy_safe = np.clip(iy, 0, y_axis.size - 2)
    tx = ux - ix
    ty = uy - iy
    sampled = (
        (1.0 - tx) * (1.0 - ty) * field_values[iy_safe, ix_safe]
        + tx * (1.0 - ty) * field_values[iy_safe, ix_safe + 1]
        + (1.0 - tx) * ty * field_values[iy_safe + 1, ix_safe]
        + tx * ty * field_values[iy_safe + 1, ix_safe + 1]
    )
    return np.where(valid, sampled, 0.0 + 0.0j)


def _angular_quadrature(config: DebyeConfig) -> _AngularQuadrature:
    theta_max = float(np.arcsin(config.numerical_aperture / config.refractive_index))
    nodes, weights = np.polynomial.legendre.leggauss(int(config.quadrature_order_r))
    theta = 0.5 * theta_max * (nodes + 1.0)
    theta_weight = 0.5 * theta_max * weights
    phi = TWOPI * np.arange(int(config.quadrature_order_phi), dtype=float) / float(config.quadrature_order_phi)
    sin_theta_1d = np.sin(theta)
    cos_theta_1d = np.cos(theta)
    rho_1d = sin_theta_1d / max(float(np.sin(theta_max)), EPS)
    sin_theta, phi_grid = np.meshgrid(sin_theta_1d, phi, indexing="ij")
    cos_theta, _ = np.meshgrid(cos_theta_1d, phi, indexing="ij")
    rho, _ = np.meshgrid(rho_1d, phi, indexing="ij")
    cos_phi = np.cos(phi_grid)
    sin_phi = np.sin(phi_grid)
    radial_weight, _ = np.meshgrid(theta_weight, phi, indexing="ij")
    solid_angle = radial_weight * sin_theta * (TWOPI / float(config.quadrature_order_phi))
    return _AngularQuadrature(
        sample_x_m=(config.pupil_radius_m * rho * cos_phi).ravel(),
        sample_y_m=(config.pupil_radius_m * rho * sin_phi).ravel(),
        sin_theta=sin_theta.ravel(),
        cos_theta=cos_theta.ravel(),
        cos_phi=cos_phi.ravel(),
        sin_phi=sin_phi.ravel(),
        solid_angle_weight=solid_angle.ravel(),
        theta_max_rad=theta_max,
    )


def _focused_angular_amplitudes(
    ex_sampled: np.ndarray,
    ey_sampled: np.ndarray,
    quadrature: _AngularQuadrature,
    config: DebyeConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cos_phi = quadrature.cos_phi
    sin_phi = quadrature.sin_phi
    sin_theta = quadrature.sin_theta
    cos_theta = quadrature.cos_theta
    e_rho = ex_sampled * cos_phi + ey_sampled * sin_phi
    e_phi = -ex_sampled * sin_phi + ey_sampled * cos_phi
    direction = 1.0 if config.propagation_direction == "+z" else -1.0
    e_theta_x = direction * cos_theta * cos_phi
    e_theta_y = direction * cos_theta * sin_phi
    e_theta_z = -sin_theta
    e_phi_x = -sin_phi
    e_phi_y = cos_phi
    if config.apodisation == "sqrt_cosine":
        apodisation = np.sqrt(np.maximum(cos_theta, 0.0))
    else:
        apodisation = np.ones_like(cos_theta)
    ax = apodisation * (e_rho * e_theta_x + e_phi * e_phi_x)
    ay = apodisation * (e_rho * e_theta_y + e_phi * e_phi_y)
    az = apodisation * e_rho * e_theta_z
    return ax, ay, az


def _positive_centered_fft(values: np.ndarray, pad_factor: int = 1) -> np.ndarray:
    """Centred positive-exponent discrete Fourier transform."""

    array = np.asarray(values, dtype=np.complex128)
    factor = int(pad_factor)
    if factor > 1:
        target_y = array.shape[0] * factor
        target_x = array.shape[1] * factor
        before_y = (target_y - array.shape[0]) // 2
        before_x = (target_x - array.shape[1]) // 2
        array = np.pad(
            array,
            (
                (before_y, target_y - array.shape[0] - before_y),
                (before_x, target_x - array.shape[1] - before_x),
            ),
        )
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(array))) * array.size


def _cartesian_debye_focus_plane(
    ex: np.ndarray,
    ey: np.ndarray,
    pupil_x: np.ndarray,
    pupil_y: np.ndarray,
    output_x: np.ndarray,
    output_y: np.ndarray,
    z_m: float,
    config: DebyeConfig,
) -> VectorFieldPlane:
    """Evaluate the sine-condition Debye integral on the native pupil grid."""

    Xp, Yp = np.meshgrid(pupil_x, pupil_y, indexing="xy")
    radius = np.hypot(Xp, Yp)
    phi = np.arctan2(Yp, Xp)
    rho = radius / config.pupil_radius_m
    valid = rho <= 1.0
    sine_max = config.numerical_aperture / config.refractive_index
    sin_theta = np.clip(sine_max * rho, 0.0, 1.0)
    cos_theta = np.sqrt(np.maximum(1.0 - sin_theta**2, 0.0))
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    e_rho = ex * cos_phi + ey * sin_phi
    e_phi = -ex * sin_phi + ey * cos_phi
    direction = 1.0 if config.propagation_direction == "+z" else -1.0
    e_theta_x = direction * cos_theta * cos_phi
    e_theta_y = direction * cos_theta * sin_phi
    e_theta_z = -sin_theta
    if config.apodisation == "sqrt_cosine":
        apodisation = np.sqrt(np.maximum(cos_theta, 0.0))
    else:
        apodisation = np.ones_like(cos_theta)
    ax = apodisation * (e_rho * e_theta_x - e_phi * sin_phi)
    ay = apodisation * (e_rho * e_theta_y + e_phi * cos_phi)
    az = apodisation * e_rho * e_theta_z
    khat_dot_a = (
        sin_theta * cos_phi * ax
        + sin_theta * sin_phi * ay
        + direction * cos_theta * az
    )
    angular_amplitude = np.sqrt(np.abs(ax) ** 2 + np.abs(ay) ** 2 + np.abs(az) ** 2)
    transversality_ratio = np.divide(
        np.abs(khat_dot_a),
        angular_amplitude,
        out=np.zeros_like(angular_amplitude, dtype=float),
        where=angular_amplitude > EPS,
    )
    active_transversality = valid & (
        angular_amplitude > 1e-13 * max(float(np.max(angular_amplitude)), EPS)
    )
    transversality_residual = (
        float(np.max(transversality_ratio[active_transversality]))
        if np.any(active_transversality)
        else 0.0
    )
    # r_pupil=(R/sin(theta_max))*sin(theta), therefore
    # sin(theta)dtheta dphi = sin(theta_max)^2 dxp dyp /(R^2 cos(theta)).
    jacobian_density = np.zeros_like(cos_theta)
    jacobian_density[valid] = sine_max**2 / (
        config.pupil_radius_m**2 * cos_theta[valid]
    )
    wave_number = TWOPI * config.refractive_index / config.wavelength_m
    z_phase = np.exp(1j * wave_number * direction * cos_theta * float(z_m))
    common = np.zeros_like(z_phase, dtype=np.complex128)
    common[valid] = jacobian_density[valid] * z_phase[valid]
    dx = float(pupil_x[1] - pupil_x[0])
    dy = float(pupil_y[1] - pupil_y[0])
    padded_nx = pupil_x.size * int(config.fft_pad_factor)
    padded_ny = pupil_y.size * int(config.fft_pad_factor)
    native_fx = np.fft.fftshift(np.fft.fftfreq(padded_nx, d=dx))
    native_fy = np.fft.fftshift(np.fft.fftfreq(padded_ny, d=dy))
    coordinate_scale = config.wavelength_m * config.pupil_radius_m / config.numerical_aperture
    native_x = coordinate_scale * native_fx
    native_y = coordinate_scale * native_fy
    Xo, Yo = np.meshgrid(output_x, output_y, indexing="xy")
    requested_native_grid = bool(
        output_x.size == native_x.size
        and output_y.size == native_y.size
        and np.allclose(output_x, native_x, rtol=0.0, atol=1e-15)
        and np.allclose(output_y, native_y, rtol=0.0, atol=1e-15)
    )
    sampled: list[np.ndarray] = []
    for component in (ax, ay, az):
        native_component = _positive_centered_fft(
            component * common, pad_factor=config.fft_pad_factor
        ) * dx * dy
        if requested_native_grid:
            sampled.append(native_component)
        else:
            sampled.append(
                _complex_bilinear(
                    native_component, native_x, native_y, Xo.ravel(), Yo.ravel()
                ).reshape(Xo.shape)
            )
    sine_radius = config.focal_length_m * config.numerical_aperture / config.refractive_index
    metadata: dict[str, Any] = {
        "method": "native_cartesian_sine_condition_vector_debye_fft",
        "pupil_field_convention": "collimated transverse Jones field immediately before an ideal aplanatic objective",
        "apodisation": config.apodisation,
        "apodisation_definition": (
            "W(theta)=sqrt(cos(theta))" if config.apodisation == "sqrt_cosine" else "W(theta)=1"
        ),
        "mapping": "sine condition: sin(theta)=(NA/n)*(r_pupil/pupil_radius)",
        "jacobian": "sin(theta_max)^2 dxp dyp /(pupil_radius^2 cos(theta))",
        "normalisation": "relative_morphology_reference",
        "normalisation_detail": "native Cartesian weighted Debye integral; deterministic relative amplitude; objective prefactor omitted",
        "absolute_energy_ledger_eligible": False,
        "vector_transversality_residual": transversality_residual,
        "native_pupil_shape": [int(ex.shape[0]), int(ex.shape[1])],
        "native_pupil_dx_m": dx,
        "native_pupil_dy_m": dy,
        "fft_pad_factor": int(config.fft_pad_factor),
        "native_focal_dx_m": float(native_x[1] - native_x[0]),
        "native_valid_sample_count": int(np.count_nonzero(valid)),
        "interpolation_to_requested_output": (
            "none; requested coordinates equal native Debye FFT grid"
            if requested_native_grid
            else "bilinear complex-field interpolation from native Debye FFT coordinates"
        ),
        "requested_output_equals_native_debye_grid": requested_native_grid,
        "propagation_direction": config.propagation_direction,
        "backend": config.backend,
        "phase_convention": "exp(+i k.r - i omega t)",
        "declared_pupil_radius_m": config.pupil_radius_m,
        "sine_condition_pupil_radius_m": sine_radius,
        "declared_to_sine_radius_ratio": float(config.pupil_radius_m / max(sine_radius, EPS)),
        "z_m": float(z_m),
    }
    return VectorFieldPlane(
        x_m=output_x,
        y_m=output_y,
        Ex=sampled[0],
        Ey=sampled[1],
        Ez=sampled[2],
        metadata=metadata,
    )


def debye_focus_plane(
    Ex_pupil: np.ndarray,
    Ey_pupil: np.ndarray,
    pupil_x_m: np.ndarray,
    pupil_y_m: np.ndarray,
    output_x_m: np.ndarray,
    output_y_m: np.ndarray,
    z_m: float,
    config: DebyeConfig,
) -> VectorFieldPlane:
    """Evaluate an independent vectorial focal-field reference.

    The quadrature is

    ``integral A(theta, phi) exp(i k.r) sin(theta) dtheta dphi``.

    Gauss-Legendre weights provide ``dtheta`` and the periodic trapezoid rule
    provides ``dphi``.  No unweighted pupil sum is used.
    """

    config.validate()
    pupil_x = _axis(pupil_x_m, "pupil_x_m", require_uniform=True)
    pupil_y = _axis(pupil_y_m, "pupil_y_m", require_uniform=True)
    output_x = _axis(output_x_m, "output_x_m", require_uniform=False)
    output_y = _axis(output_y_m, "output_y_m", require_uniform=False)
    ex = np.asarray(Ex_pupil, dtype=np.complex128)
    ey = np.asarray(Ey_pupil, dtype=np.complex128)
    if ex.shape != ey.shape or ex.shape != (pupil_y.size, pupil_x.size):
        raise ValueError("Ex_pupil and Ey_pupil must match the pupil axes")
    if not np.isfinite(z_m):
        raise ValueError("z_m must be finite")
    if (
        pupil_x[0] > -config.pupil_radius_m
        or pupil_x[-1] < config.pupil_radius_m
        or pupil_y[0] > -config.pupil_radius_m
        or pupil_y[-1] < config.pupil_radius_m
    ):
        raise ValueError("pupil axes do not span the declared pupil radius")

    if config.backend == "cartesian_fft":
        return _cartesian_debye_focus_plane(
            ex, ey, pupil_x, pupil_y, output_x, output_y, float(z_m), config
        )

    quadrature = _angular_quadrature(config)
    ex_q = _complex_bilinear(ex, pupil_x, pupil_y, quadrature.sample_x_m, quadrature.sample_y_m)
    ey_q = _complex_bilinear(ey, pupil_x, pupil_y, quadrature.sample_x_m, quadrature.sample_y_m)
    ax, ay, az = _focused_angular_amplitudes(ex_q, ey_q, quadrature, config)
    direction = 1.0 if config.propagation_direction == "+z" else -1.0
    khat_dot_a = (
        quadrature.sin_theta * quadrature.cos_phi * ax
        + quadrature.sin_theta * quadrature.sin_phi * ay
        + direction * quadrature.cos_theta * az
    )
    angular_amplitude = np.sqrt(np.abs(ax) ** 2 + np.abs(ay) ** 2 + np.abs(az) ** 2)
    transversality_ratio = np.divide(
        np.abs(khat_dot_a),
        angular_amplitude,
        out=np.zeros_like(angular_amplitude, dtype=float),
        where=angular_amplitude > EPS,
    )
    active_transversality = angular_amplitude > 1e-13 * max(float(np.max(angular_amplitude)), EPS)
    transversality_residual = (
        float(np.max(transversality_ratio[active_transversality]))
        if np.any(active_transversality)
        else 0.0
    )
    weighted = (
        ax * quadrature.solid_angle_weight,
        ay * quadrature.solid_angle_weight,
        az * quadrature.solid_angle_weight,
    )
    wave_number = TWOPI * config.refractive_index / config.wavelength_m
    kx = wave_number * quadrature.sin_theta * quadrature.cos_phi
    ky = wave_number * quadrature.sin_theta * quadrature.sin_phi
    kz = wave_number * direction * quadrature.cos_theta
    X, Y = np.meshgrid(output_x, output_y, indexing="xy")
    flat_x = X.ravel()
    flat_y = Y.ravel()
    out = [np.empty(flat_x.size, dtype=np.complex128) for _ in range(3)]
    chunk_size = int(config.max_output_points)
    for start in range(0, flat_x.size, chunk_size):
        stop = min(start + chunk_size, flat_x.size)
        phase_argument = (
            flat_x[start:stop, None] * kx[None, :]
            + flat_y[start:stop, None] * ky[None, :]
            + float(z_m) * kz[None, :]
        )
        kernel = np.exp(1j * phase_argument)
        for component, angular_values in zip(out, weighted):
            component[start:stop] = kernel @ angular_values
    shape = (output_y.size, output_x.size)
    sine_radius = config.focal_length_m * config.numerical_aperture / config.refractive_index
    metadata: dict[str, Any] = {
        "method": "chunked_direct_vector_debye_solid_angle_quadrature",
        "pupil_field_convention": "collimated transverse Jones field immediately before an ideal aplanatic objective",
        "apodisation": config.apodisation,
        "apodisation_definition": (
            "W(theta)=sqrt(cos(theta))" if config.apodisation == "sqrt_cosine" else "W(theta)=1"
        ),
        "mapping": "sine condition: sin(theta)=(NA/n)*(r_pupil/pupil_radius)",
        "jacobian": "sin(theta) dtheta dphi with Gauss-Legendre theta and periodic trapezoid phi weights",
        "normalisation": "relative_morphology_reference",
        "normalisation_detail": "raw weighted Debye integral; deterministic relative amplitude; objective prefactor omitted",
        "absolute_energy_ledger_eligible": False,
        "vector_transversality_residual": transversality_residual,
        "quadrature_order_r": int(config.quadrature_order_r),
        "quadrature_order_phi": int(config.quadrature_order_phi),
        "angular_sample_count": int(quadrature.solid_angle_weight.size),
        "max_output_points_per_chunk": chunk_size,
        "theta_max_rad": quadrature.theta_max_rad,
        "propagation_direction": config.propagation_direction,
        "backend": config.backend,
        "phase_convention": "exp(+i k.r - i omega t)",
        "declared_pupil_radius_m": config.pupil_radius_m,
        "sine_condition_pupil_radius_m": sine_radius,
        "declared_to_sine_radius_ratio": float(config.pupil_radius_m / max(sine_radius, EPS)),
        "z_m": float(z_m),
    }
    return VectorFieldPlane(
        x_m=output_x,
        y_m=output_y,
        Ex=out[0].reshape(shape),
        Ey=out[1].reshape(shape),
        Ez=out[2].reshape(shape),
        metadata=metadata,
    )


def debye_config_dict(config: DebyeConfig) -> Mapping[str, Any]:
    """Return a JSON-friendly record of the declared solver configuration."""

    return {
        "wavelength_m": float(config.wavelength_m),
        "refractive_index": float(config.refractive_index),
        "numerical_aperture": float(config.numerical_aperture),
        "focal_length_m": float(config.focal_length_m),
        "pupil_radius_m": float(config.pupil_radius_m),
        "apodisation": config.apodisation,
        "propagation_direction": config.propagation_direction,
        "quadrature_order_r": int(config.quadrature_order_r),
        "quadrature_order_phi": int(config.quadrature_order_phi),
        "max_output_points": int(config.max_output_points),
        "fft_pad_factor": int(config.fft_pad_factor),
    }


__all__ = ["DebyeConfig", "VectorFieldPlane", "debye_config_dict", "debye_focus_plane"]
