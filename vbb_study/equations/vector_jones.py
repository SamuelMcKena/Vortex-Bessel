"""Jones-vector equations for the vector-beam pipeline.

The production vector helpers live in ``vbb_vector``.  This module keeps the
formula pieces small enough to compare directly with the paper's eqs. 9-17 and
with the conventions document.

Convention:
  Jones vectors are represented in the fixed transverse linear basis
  ``[Ex, Ey]^T``.  A scalar Bessel/SAS envelope is the complex amplitude shared
  by the field; the vector part is the local Jones direction.  Intensity is
  ``S0 = |Ex|^2 + |Ey|^2``.  The radial basis is
  ``e_r = (cos(phi), sin(phi))`` and the azimuthal basis is
  ``e_phi = (-sin(phi), cos(phi))``.  At the coordinate singularity
  ``r = 0`` these bases are explicitly set to zero unless callers provide a
  different centre policy.  Stokes convention is documented in
  :func:`stokes_from_linear_components`, including the sign of ``S3``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

EPS = 1.0e-30


def linear_polarizer_matrix(theta_rad: float) -> np.ndarray:
    """Return the ideal linear-polarizer Jones matrix at angle ``theta``."""

    c = math.cos(float(theta_rad))
    s = math.sin(float(theta_rad))
    return np.array([[c * c, c * s], [c * s, s * s]], dtype=complex)


def linear_polarizer_45_matrix() -> np.ndarray:
    """Return the 45 degree polarizer matrix used in paper eq. 10."""

    return linear_polarizer_matrix(math.pi / 4.0)


def first_slm_half_matrix(beta_rad: Any) -> np.ndarray:
    """Return the first SLM-half matrix from paper eq. 11."""

    phase = np.exp(1j * np.asarray(beta_rad, dtype=float))
    one = np.ones_like(phase, dtype=complex)
    zero = np.zeros_like(phase, dtype=complex)
    return np.stack([np.stack([phase, zero], axis=0), np.stack([zero, one], axis=0)], axis=0)


def reflection_helicity_flip_matrix() -> np.ndarray:
    """Return the reflection/helicity-flip matrix used between the SLM halves."""

    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def hwp_45_swap_matrix() -> np.ndarray:
    """Return the 45 degree HWP component-swap matrix from paper eq. 12."""

    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)


def second_slm_half_matrix(beta_rad: Any) -> np.ndarray:
    """Return the second SLM-half matrix from paper eq. 13."""

    phase = np.exp(-1j * np.asarray(beta_rad, dtype=float))
    one = np.ones_like(phase, dtype=complex)
    zero = np.zeros_like(phase, dtype=complex)
    return np.stack([np.stack([one, zero], axis=0), np.stack([zero, phase], axis=0)], axis=0)


def qwp1_matrix(mode: str) -> np.ndarray:
    """Return QWP1 for radial or azimuthal generation from paper eqs. 14-15."""

    key = str(mode).lower().strip()
    if key == "radial":
        return np.array([[1.0, 0.0], [0.0, -1j]], dtype=complex)
    if key == "azimuthal":
        return np.array([[-1j, 0.0], [0.0, 1.0]], dtype=complex)
    raise ValueError("mode must be 'radial' or 'azimuthal'.")


def qwp2_matrix() -> np.ndarray:
    """Return QWP2 at 45 degrees from paper eq. 16."""

    return 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=complex)


def compose_jones_matrices(*matrices: Any) -> np.ndarray:
    """Compose Jones matrices in physical order, left-multiplying each optic."""

    result: np.ndarray | None = None
    for matrix in matrices:
        arr = np.asarray(matrix, dtype=complex)
        if arr.shape[0:2] != (2, 2):
            raise ValueError("Jones matrices must have leading shape (2, 2).")
        result = arr if result is None else np.einsum("ab...,bc...->ac...", arr, result, optimize=True)
    if result is None:
        raise ValueError("At least one matrix is required.")
    return result


def apply_jones_matrix(Ex: Any, Ey: Any, matrix: Any) -> tuple[np.ndarray, np.ndarray]:
    """Apply a Jones matrix to two linear-basis field components."""

    arr = np.asarray(matrix, dtype=complex)
    if arr.shape[0:2] != (2, 2):
        raise ValueError("Jones matrices must have leading shape (2, 2).")
    Ex_arr = np.asarray(Ex, dtype=complex)
    Ey_arr = np.asarray(Ey, dtype=complex)
    return arr[0, 0] * Ex_arr + arr[0, 1] * Ey_arr, arr[1, 0] * Ex_arr + arr[1, 1] * Ey_arr


def circular_to_linear(E_sigma_plus: Any, E_sigma_minus: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert circular-basis components into the linear x/y basis."""

    plus = np.asarray(E_sigma_plus, dtype=complex)
    minus = np.asarray(E_sigma_minus, dtype=complex)
    scale = 1.0 / math.sqrt(2.0)
    return scale * (plus + minus), 1j * scale * (plus - minus)


def radial_basis_from_xy(X: Any, Y: Any, *, centre_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the radial Jones basis ``(cos(phi), sin(phi))``.

    The basis is undefined at ``X=Y=0``.  This function handles that centre
    singularity explicitly by assigning both components to ``centre_value``.
    The default zero value keeps intensity finite and prevents an accidental
    preferred polarisation direction on axis.
    """

    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    R = np.hypot(X_arr, Y_arr)
    ex = np.divide(X_arr, R, out=np.full_like(X_arr, float(centre_value)), where=R > EPS)
    ey = np.divide(Y_arr, R, out=np.full_like(Y_arr, float(centre_value)), where=R > EPS)
    return ex.astype(float), ey.astype(float)


def azimuthal_basis_from_xy(X: Any, Y: Any, *, centre_value: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the azimuthal Jones basis ``(-sin(phi), cos(phi))``.

    The centre singularity is handled with the same explicit policy as
    :func:`radial_basis_from_xy`.
    """

    er_x, er_y = radial_basis_from_xy(X, Y, centre_value=centre_value)
    return -er_y, er_x


def cylindrical_vector_field(
    envelope: Any,
    X: Any,
    Y: Any,
    *,
    mode: str,
    centre_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a radial or azimuthal Jones basis to a scalar envelope."""

    key = str(mode).lower().strip()
    if key == "radial":
        bx, by = radial_basis_from_xy(X, Y, centre_value=centre_value)
    elif key == "azimuthal":
        bx, by = azimuthal_basis_from_xy(X, Y, centre_value=centre_value)
    else:
        raise ValueError("mode must be 'radial' or 'azimuthal'.")
    env = np.asarray(envelope, dtype=complex)
    return env * bx, env * by


def linear_to_circular(Ex: Any, Ey: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert linear x/y components into sigma-plus/sigma-minus components."""

    Ex_arr = np.asarray(Ex, dtype=complex)
    Ey_arr = np.asarray(Ey, dtype=complex)
    scale = 1.0 / math.sqrt(2.0)
    return scale * (Ex_arr - 1j * Ey_arr), scale * (Ex_arr + 1j * Ey_arr)


def vector_intensity(Ex: Any, Ey: Any) -> np.ndarray:
    """Return ``S0 = |Ex|^2 + |Ey|^2`` for a linear-basis Jones field."""

    Ex_arr = np.asarray(Ex, dtype=complex)
    Ey_arr = np.asarray(Ey, dtype=complex)
    return np.abs(Ex_arr) ** 2 + np.abs(Ey_arr) ** 2


def stokes_from_linear_components(Ex: Any, Ey: Any) -> dict[str, np.ndarray]:
    """Return Stokes parameters ``S0..S3`` from linear Jones components.

    I use ``S3 = -2 Im(Ex Ey*)`` to match the handedness convention in
    ``vbb_vector``.  The ellipse angles below record the paper typo explicitly:
    orientation uses ``tan(2 psi)=S2/S1`` and ellipticity uses
    ``sin(2 chi)=S3/S0``.
    """

    Ex_arr = np.asarray(Ex, dtype=complex)
    Ey_arr = np.asarray(Ey, dtype=complex)
    cross = Ex_arr * np.conj(Ey_arr)
    S0 = vector_intensity(Ex_arr, Ey_arr)
    S1 = np.abs(Ex_arr) ** 2 - np.abs(Ey_arr) ** 2
    S2 = 2.0 * np.real(cross)
    S3 = -2.0 * np.imag(cross)
    return {"S0": S0, "S1": S1, "S2": S2, "S3": S3}


def normalized_stokes(stokes: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return finite normalized Stokes parameters where ``S0 > 0``.

    Output keys are ``s1``, ``s2``, and ``s3``.  Pixels with zero intensity are
    set to zero, which keeps singular vector-beam centres finite without
    hiding them from callers that inspect ``S0``.
    """

    S0 = np.asarray(stokes["S0"], dtype=float)
    out: dict[str, np.ndarray] = {}
    for source, target in (("S1", "s1"), ("S2", "s2"), ("S3", "s3")):
        values = np.asarray(stokes[source], dtype=float)
        out[target] = np.divide(values, S0, out=np.zeros_like(values, dtype=float), where=S0 > EPS)
    return out


def ellipse_angles_from_stokes(stokes: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Return orientation ``psi`` and ellipticity ``chi`` in radians.

    The paper text says ``sin(2 psi)=S3/S0``; I treat that as the typo noted in
    the study docs and use the standard corrected form
    ``tan(2 psi)=S2/S1`` and ``sin(2 chi)=S3/S0``.
    """

    S0 = np.asarray(stokes["S0"], dtype=float)
    S1 = np.asarray(stokes["S1"], dtype=float)
    S2 = np.asarray(stokes["S2"], dtype=float)
    S3 = np.asarray(stokes["S3"], dtype=float)
    psi = 0.5 * np.arctan2(S2, S1)
    chi = 0.5 * np.arcsin(np.clip(S3 / np.maximum(S0, EPS), -1.0, 1.0))
    return {"psi_rad": psi, "chi_rad": chi}


def linear_analyzer_intensity(Ex: Any, Ey: Any, theta_rad: float) -> np.ndarray:
    """Return intensity transmitted by an ideal linear analyzer."""

    c = math.cos(float(theta_rad))
    s = math.sin(float(theta_rad))
    projected = c * np.asarray(Ex, dtype=complex) + s * np.asarray(Ey, dtype=complex)
    return np.abs(projected) ** 2
