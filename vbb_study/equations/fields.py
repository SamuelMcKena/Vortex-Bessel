"""Scalar field, grid, FFT, and phase primitives for the engine facade."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

TWOPI = 2.0 * np.pi
EPS = 1e-30


def make_xy_grid(N: int, dx_m: float) -> Dict[str, Any]:
    """Square centered grid with `fftfreq` frequency coordinates."""

    x = (np.arange(int(N)) - int(N) / 2 + 0.5) * float(dx_m)
    X, Y = np.meshgrid(x, x, indexing="xy")
    R = np.hypot(X, Y)
    PHI = np.arctan2(Y, X)
    fx = np.fft.fftshift(np.fft.fftfreq(int(N), d=float(dx_m)))
    FX, FY = np.meshgrid(fx, fx, indexing="xy")
    return {"N": int(N), "dx": float(dx_m), "x": x, "X": X, "Y": Y, "R": R, "PHI": PHI, "FX": FX, "FY": FY}


def make_rect_grid(nx: int, ny: int, dx_m: float) -> Dict[str, Any]:
    """Rectangular centered grid for the physical SLM device."""

    x = (np.arange(int(nx)) - int(nx) / 2 + 0.5) * float(dx_m)
    y = (np.arange(int(ny)) - int(ny) / 2 + 0.5) * float(dx_m)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    PHI = np.arctan2(Y, X)
    return {"nx": int(nx), "ny": int(ny), "dx": float(dx_m), "x": x, "y": y, "X": X, "Y": Y, "R": R, "PHI": PHI}


def fft2c(U: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U)))


def ifft2c(A: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(A)))


def phase_wrap(phi: np.ndarray) -> np.ndarray:
    return np.mod(phi, TWOPI)


def quantize_phase(phi: np.ndarray, bits: int) -> np.ndarray:
    """Round wrapped phase to `2**bits` phase levels."""

    if int(bits) <= 0:
        raise ValueError("phase bit depth must be positive")
    levels = 1 << int(bits)
    wrapped = phase_wrap(phi)
    idx = np.floor(wrapped / TWOPI * levels + 0.5).astype(np.int64) % levels
    return idx.astype(float) * TWOPI / float(levels)


def phase_to_gray(phase: np.ndarray, bits: int = 8, invert: bool = False) -> np.ndarray:
    """Convert wrapped phase to an uploadable grayscale array."""

    levels = 1 << int(bits)
    idx = np.floor(phase_wrap(phase) / TWOPI * levels + 0.5).astype(np.int64) % levels
    if levels == 256:
        gray = idx.astype(np.uint8)
    else:
        gray = np.round(idx * 255.0 / max(levels - 1, 1)).astype(np.uint8)
    if invert:
        gray = np.uint8(255) - gray
    return gray


def gray_to_phase(gray: np.ndarray, bits: int = 8, invert: bool = False) -> np.ndarray:
    """Approximate phase represented by a grayscale hologram."""

    g = np.asarray(gray, dtype=np.uint8)
    if invert:
        g = np.uint8(255) - g
    levels = 1 << int(bits)
    if levels == 256:
        idx = g.astype(np.int64)
    else:
        idx = np.round(g.astype(float) * (levels - 1) / 255.0).astype(np.int64)
    return idx.astype(float) * TWOPI / float(levels)


def gaussian_amplitude(R: np.ndarray, radius_m: float) -> np.ndarray:
    return np.exp(-(R ** 2) / max(float(radius_m), EPS) ** 2)


def next_power_of_two(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def compute_kr(k0: float, n_axicon: float, n_medium: float, gamma_rad: float) -> float:
    """Transverse wavevector from the selected tangent axicon convention."""

    return float(k0 * (float(n_axicon) - float(n_medium)) * math.tan(float(gamma_rad)))
