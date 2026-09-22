from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from .config import SlmGeometry, SlmPhaseConfig

TWOPI = 2.0 * np.pi


@dataclass
class PhaseResult:
    phase_rad: np.ndarray
    gray_float: np.ndarray
    gray_uint8: np.ndarray
    components: Dict[str, np.ndarray]
    warnings: List[str]
    stats: Dict[str, float]


def _mesh(config: SlmPhaseConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return pixel and metre coordinates centered on the editable SLM centre."""

    g = config.geometry
    yy, xx = np.indices((g.height_px, g.width_px), dtype=np.float64)
    x_px = xx - config.center_x_px
    y_px = yy - config.center_y_px

    # Optional rotation for deliberately misaligned/rotated computational terms.
    if abs(config.term_rotation_deg) > 1e-12:
        theta = np.deg2rad(config.term_rotation_deg)
        c, s = np.cos(theta), np.sin(theta)
        xr = c * x_px + s * y_px
        yr = -s * x_px + c * y_px
        x_px, y_px = xr, yr

    x_m = x_px * g.pixel_pitch_m
    y_m = y_px * g.pixel_pitch_m
    return x_px, y_px, x_m, y_m


def pupil_radius_px(config: SlmPhaseConfig) -> float:
    return max(1.0, (config.pupil_diameter_mm * 1e-3) / (2.0 * config.geometry.pixel_pitch_m))


def pupil_mask(config: SlmPhaseConfig) -> np.ndarray:
    x_px, y_px, _, _ = _mesh(config)
    return np.sqrt(x_px**2 + y_px**2) <= pupil_radius_px(config)


def phase_to_gray(phase_rad: np.ndarray, bit_depth: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Wrap phase to one SLM period and convert to grayscale."""

    levels = (2**bit_depth) - 1
    wrapped = np.mod(phase_rad, TWOPI)
    gray_float = wrapped / TWOPI * levels
    gray_uint8 = np.rint(np.clip(gray_float, 0, 255)).astype(np.uint8)
    return gray_float, gray_uint8


def gray_to_phase(gray: np.ndarray) -> np.ndarray:
    return (gray.astype(np.float64) / 255.0) * TWOPI


def _resize_wrapped_phase_phasor(phase_rad: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize wrapped phase through the complex unit phasor, not across a 2π jump."""

    src = np.asarray(phase_rad, dtype=np.float64)
    if src.shape == target_shape:
        return src.copy()
    # Pillow can resize floating real/imaginary images robustly.
    real = Image.fromarray(np.cos(src).astype(np.float32), mode="F")
    imag = Image.fromarray(np.sin(src).astype(np.float32), mode="F")
    target_xy = (int(target_shape[1]), int(target_shape[0]))
    real = np.asarray(real.resize(target_xy, Image.Resampling.BILINEAR), dtype=np.float64)
    imag = np.asarray(imag.resize(target_xy, Image.Resampling.BILINEAR), dtype=np.float64)
    return np.angle(real + 1j * imag)


def load_retrieved_correction_phase(path: str, config: SlmPhaseConfig, warnings: List[str]) -> np.ndarray:
    """Load an additive inverse/retrieved correction in radians.

    Preferred input is ``.npy`` containing a native SLM-sized signed phase map in
    radians. Image files remain supported as wrapped 8-bit phase for convenience.
    No pupil gate is applied here; zero contribution outside a correction region
    leaves the pre-existing blaze/wavefront phase untouched.
    """

    g = config.geometry
    target_shape = (g.height_px, g.width_px)
    if not path:
        warnings.append(f"{config.name}: retrieved correction enabled but no file selected.")
        return np.zeros(target_shape, dtype=np.float64)

    p = Path(path)
    if not p.exists():
        warnings.append(f"{config.name}: retrieved correction file not found: {path}")
        return np.zeros(target_shape, dtype=np.float64)

    if p.suffix.lower() == ".npy":
        arr = np.asarray(np.load(p, allow_pickle=False), dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"{config.name}: retrieved correction .npy must be 2-D, got shape {arr.shape}.")
        bad = ~np.isfinite(arr)
        if np.any(bad):
            warnings.append(
                f"{config.name}: retrieved correction contains {int(np.count_nonzero(bad))} non-finite pixels; replacing them with zero phase."
            )
            arr = arr.copy()
            arr[bad] = 0.0
    else:
        img = Image.open(p).convert("L")
        arr = gray_to_phase(np.asarray(img, dtype=np.float64))

    if arr.shape != target_shape:
        warnings.append(
            f"{config.name}: retrieved correction resized from {arr.shape[::-1]} to "
            f"{(g.width_px, g.height_px)} through complex-phase interpolation. "
            "For a hardware correction, a native-size calibrated map is strongly preferred."
        )
        arr = _resize_wrapped_phase_phasor(arr, target_shape)

    return float(config.retrieved_correction_gain) * arr


def load_wavefront_phase(path: str, config: SlmPhaseConfig, warnings: List[str]) -> np.ndarray:
    g = config.geometry
    if not path:
        warnings.append(f"{config.name}: wavefront term enabled but no file selected.")
        return np.zeros((g.height_px, g.width_px), dtype=np.float64)

    p = Path(path)
    if not p.exists():
        warnings.append(f"{config.name}: wavefront file not found: {path}")
        return np.zeros((g.height_px, g.width_px), dtype=np.float64)

    img = Image.open(p).convert("L")
    if img.size != (g.width_px, g.height_px):
        warnings.append(
            f"{config.name}: wavefront image resized from {img.size} to "
            f"{(g.width_px, g.height_px)}. Confirm this is intentional."
        )
        img = img.resize((g.width_px, g.height_px), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float64)
    return config.wavefront_gain * gray_to_phase(arr)


def background_phase(config: SlmPhaseConfig) -> np.ndarray:
    g = config.geometry
    gray = np.clip(config.background_gray, 0.0, 255.0)
    return np.full((g.height_px, g.width_px), gray / 255.0 * TWOPI, dtype=np.float64)


def blaze_phase(config: SlmPhaseConfig) -> np.ndarray:
    period = float(config.blaze_period_px)
    if abs(period) < 1e-9:
        return np.zeros((config.geometry.height_px, config.geometry.width_px), dtype=np.float64)
    x_px, y_px, _, _ = _mesh(config)
    coord = x_px if config.blaze_axis == "x" else y_px
    return config.blaze_sign * TWOPI * coord / period


def focus_phase(config: SlmPhaseConfig) -> np.ndarray:
    g = config.geometry
    f_mm = float(config.focus_focal_length_mm)
    if abs(f_mm) < 1e-12:
        return np.zeros((g.height_px, g.width_px), dtype=np.float64)
    _, _, x_m, y_m = _mesh(config)
    f_m = f_mm * 1e-3
    k = TWOPI / g.wavelength_m
    return config.focus_sign * k * (x_m**2 + y_m**2) / (2.0 * f_m)


def axicon_phase(config: SlmPhaseConfig) -> np.ndarray:
    period = float(config.axicon_period_px)
    if abs(period) < 1e-9:
        return np.zeros((config.geometry.height_px, config.geometry.width_px), dtype=np.float64)
    x_px, y_px, _, _ = _mesh(config)
    r_px = np.sqrt(x_px**2 + y_px**2)
    return config.axicon_sign * TWOPI * r_px / period


def vortex_phase(config: SlmPhaseConfig) -> np.ndarray:
    charge = int(config.vortex_charge)
    if charge == 0:
        return np.zeros((config.geometry.height_px, config.geometry.width_px), dtype=np.float64)
    x_px, y_px, _, _ = _mesh(config)
    theta = np.arctan2(y_px, x_px)
    return charge * theta


def low_order_zernike_phase(config: SlmPhaseConfig) -> np.ndarray:
    """Return a compact low-order aberration correction basis.

    Coefficients are entered in waves.  The polynomials are deliberately kept in
    simple unnormalised radial form so legacy Z4^0 values preserve their meaning.
    The correction is only applied inside the configured circular pupil radius.
    """

    coeffs = (
        config.z20_amp_waves,
        config.z22_cos_amp_waves,
        config.z22_sin_amp_waves,
        config.z31_cos_amp_waves,
        config.z31_sin_amp_waves,
        config.z40_amp_waves,
    )
    if all(abs(float(v)) < 1e-15 for v in coeffs):
        return np.zeros((config.geometry.height_px, config.geometry.width_px), dtype=np.float64)

    x_px, y_px, _, _ = _mesh(config)
    rho = np.sqrt(x_px**2 + y_px**2) / pupil_radius_px(config)
    theta = np.arctan2(y_px, x_px)
    mask = rho <= 1.0

    z20 = 2.0 * rho**2 - 1.0
    z22c = rho**2 * np.cos(2.0 * theta)
    z22s = rho**2 * np.sin(2.0 * theta)
    z31c = (3.0 * rho**3 - 2.0 * rho) * np.cos(theta)
    z31s = (3.0 * rho**3 - 2.0 * rho) * np.sin(theta)
    z40 = 6.0 * rho**4 - 6.0 * rho**2 + 1.0

    waves = (
        config.z20_amp_waves * z20
        + config.z22_cos_amp_waves * z22c
        + config.z22_sin_amp_waves * z22s
        + config.z31_cos_amp_waves * z31c
        + config.z31_sin_amp_waves * z31s
        + config.z40_amp_waves * z40
    )
    phase = TWOPI * waves
    phase[~mask] = 0.0
    return phase


def zernike_z40_phase(config: SlmPhaseConfig) -> np.ndarray:
    """Backward-compatible alias for the expanded low-order correction term."""

    return low_order_zernike_phase(config)


def custom_phase_overlay(config: SlmPhaseConfig, warnings: List[str]) -> np.ndarray:
    if not config.custom_phase_path:
        warnings.append(f"{config.name}: custom phase overlay enabled but no file selected.")
        return np.zeros((config.geometry.height_px, config.geometry.width_px), dtype=np.float64)

    p = Path(config.custom_phase_path)
    if not p.exists():
        warnings.append(f"{config.name}: custom phase file not found: {config.custom_phase_path}")
        return np.zeros((config.geometry.height_px, config.geometry.width_px), dtype=np.float64)

    img = Image.open(p).convert("L")
    target = (config.geometry.width_px, config.geometry.height_px)
    if img.size != target:
        warnings.append(
            f"{config.name}: custom phase image resized from {img.size} to {target}. Confirm this is intentional."
        )
        img = img.resize(target, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float64)
    return config.custom_phase_gain * gray_to_phase(arr)

def spherical_interface_phase(config: SlmPhaseConfig, warnings: List[str]) -> np.ndarray:
    """Return only the additive sample-interface correction term.

    Outside the configured computational pupil this function returns zero phase,
    which means the already composed blaze/wavefront/other terms remain untouched.
    It never gates or blanks the SLM. The separate ``circular_pupil`` switch is
    the only intentional outside-pupil blanking operation.
    """
    g = config.geometry
    NA = float(config.interface_NA)
    depth_um = float(config.interface_depth_um)
    if abs(depth_um) < 1e-15 or abs(NA) < 1e-15:
        return np.zeros((g.height_px, g.width_px), dtype=np.float64)

    n1 = float(config.interface_n1)
    n2 = float(config.interface_n2)
    if n1 <= 0 or n2 <= 0:
        warnings.append(f"{config.name}: interface refractive indices must be positive.")
        return np.zeros((g.height_px, g.width_px), dtype=np.float64)

    x_px, y_px, _, _ = _mesh(config)
    rho = np.sqrt(x_px**2 + y_px**2) / pupil_radius_px(config)
    pupil = rho <= 1.0

    phi = np.zeros((g.height_px, g.width_px), dtype=np.float64)
    arg1 = n1**2 - (NA * rho[pupil]) ** 2
    arg2 = n2**2 - (NA * rho[pupil]) ** 2
    valid = (arg1 >= 0.0) & (arg2 >= 0.0)

    if not np.all(valid):
        warnings.append(
            f"{config.name}: interface correction has evanescent/invalid high-NA samples; clipped invalid pupil points."
        )

    local_rho = rho[pupil]
    local_phi = np.zeros_like(local_rho)
    s = n1 / n2
    wl_um = g.wavelength_um
    local_phi[valid] = (
        config.interface_sign
        * ((TWOPI * depth_um) / (s * wl_um))
        * (s * np.sqrt(arg1[valid]) - np.sqrt(arg2[valid]))
    )
    phi[pupil] = local_phi
    return phi


def n_fold_phase(config: SlmPhaseConfig) -> np.ndarray:
    amp = float(config.n_fold_amp_waves)
    order = max(1, int(config.n_fold_order))
    if abs(amp) < 1e-15:
        return np.zeros((config.geometry.height_px, config.geometry.width_px), dtype=np.float64)
    x_px, y_px, _, _ = _mesh(config)
    theta = np.arctan2(y_px, x_px)
    r_norm = np.sqrt(x_px**2 + y_px**2) / pupil_radius_px(config)
    envelope = np.clip(r_norm, 0.0, 1.0)
    phase = TWOPI * amp * envelope * np.cos(order * theta)
    phase[r_norm > 1.0] = 0.0
    return phase


def compose_phase(config: SlmPhaseConfig) -> PhaseResult:
    warnings: List[str] = []
    components: Dict[str, np.ndarray] = {}

    phase = background_phase(config)
    components["background"] = phase.copy()

    if config.switches.wavefront:
        comp = load_wavefront_phase(config.wavefront_path, config, warnings)
        phase += comp
        components["wavefront"] = comp

    if config.switches.blaze:
        if abs(config.blaze_period_px) < 2.0:
            warnings.append(f"{config.name}: blaze period < 2 px is likely aliased.")
        comp = blaze_phase(config)
        phase += comp
        components["blaze"] = comp

    if config.switches.focus:
        comp = focus_phase(config)
        phase += comp
        components["focus"] = comp

    if config.switches.axicon:
        if abs(config.axicon_period_px) < 2.0:
            warnings.append(f"{config.name}: axicon radial period < 2 px is likely aliased.")
        comp = axicon_phase(config)
        phase += comp
        components["axicon"] = comp

    if config.switches.vortex:
        comp = vortex_phase(config)
        phase += comp
        components["vortex"] = comp

    if config.switches.spherical_interface:
        comp = spherical_interface_phase(config, warnings)
        phase += comp
        components["spherical_interface"] = comp

    if config.switches.zernike_z40:
        comp = low_order_zernike_phase(config)
        phase += comp
        components["low_order_zernike"] = comp

    if config.switches.retrieved_correction:
        comp = load_retrieved_correction_phase(config.retrieved_correction_path, config, warnings)
        phase += comp
        components["retrieved_correction"] = comp

    if config.switches.custom_phase:
        comp = custom_phase_overlay(config, warnings)
        phase += comp
        components["custom_phase"] = comp

    if config.switches.n_fold:
        comp = n_fold_phase(config)
        phase += comp
        components["n_fold"] = comp

    phase *= config.global_phase_gain

    if config.switches.circular_pupil:
        warnings.append(
            f"{config.name}: CIRCULAR PUPIL GATE IS ON — phase outside the configured pupil is intentionally replaced by the background. "
            "Turn this OFF if you want blaze/wavefront/corrections across the full panel."
        )
        mask = pupil_mask(config)
        outside = background_phase(config)
        phase = np.where(mask, phase, outside)
        components["circular_pupil_gate"] = mask.astype(np.float64)

    gray_float, gray_uint8 = phase_to_gray(phase, config.output_bit_depth)
    stats = {
        "gray_min": float(np.min(gray_float)),
        "gray_max": float(np.max(gray_float)),
        "gray_mean": float(np.mean(gray_float)),
        "gray_std": float(np.std(gray_float)),
        "wrapped_phase_min": float(np.min(np.mod(phase, TWOPI))),
        "wrapped_phase_max": float(np.max(np.mod(phase, TWOPI))),
        "component_count": float(len(components)),
    }
    return PhaseResult(phase, gray_float, gray_uint8, components, warnings, stats)


def save_phase_png(path: Path, gray_uint8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray_uint8, mode="L").save(path)
