"""Conservative dual-SLM diagnostics and correction-safety primitives.

This module deliberately separates optical hypotheses from hardware commands.
No uint8 SLM export is permitted until the relevant calibration is complete.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import json
import numpy as np
import pandas as pd


class Evidence(str, Enum):
    SUPPORTED = "SUPPORTED"
    POSSIBLE = "POSSIBLE"
    WEAK = "WEAK EVIDENCE"
    RULED_OUT = "RULED OUT"
    UNMEASURED = "UNMEASURED"
    DEGENERATE = "DEGENERATE"


class Correctability(str, Enum):
    YES = "YES"
    PARTIAL = "PARTIALLY"
    NO = "NO"
    UNKNOWN = "UNKNOWN UNTIL CALIBRATED"


@dataclass(frozen=True)
class ScanConfig:
    name: str
    data_dir: Path
    z_hex_mm: tuple[float, ...]
    acquisition_index: tuple[int, ...]
    reference_hex_mm: float = 6.0

    def validate(self) -> None:
        n = len(self.z_hex_mm)
        if n == 0 or len(self.acquisition_index) != n:
            raise ValueError("Scan coordinates and acquisition indices must be non-empty and equal length")
        if len(set(self.z_hex_mm)) != n:
            raise ValueError("Duplicate physical z coordinates are forbidden")
        if len(set(self.acquisition_index)) != n:
            raise ValueError("Duplicate acquisition indices are forbidden")
        if not self.data_dir.is_dir():
            raise FileNotFoundError(self.data_dir)

    @property
    def z_rel_mm(self) -> np.ndarray:
        return np.asarray(self.z_hex_mm, float) - self.reference_hex_mm


@dataclass(frozen=True)
class SLMCalibration:
    name: str
    pixel_pitch_um: float | None = None
    active_nx: int | None = None
    active_ny: int | None = None
    phase_stroke_rad: float | None = None
    phase_lut: tuple[float, ...] | None = None
    camera_magnification: float | None = None
    camera_rotation_deg: float | None = None
    flip_x: bool | None = None
    flip_y: bool | None = None
    polarization_verified: bool = False
    static_correction_path: Path | None = None

    @property
    def hardware_ready(self) -> bool:
        return all(v is not None for v in (
            self.pixel_pitch_um, self.active_nx, self.active_ny,
            self.phase_stroke_rad, self.phase_lut, self.camera_magnification,
            self.camera_rotation_deg, self.flip_x, self.flip_y,
        )) and self.polarization_verified

    def blockers(self) -> list[str]:
        missing = []
        labels = {
            "pixel_pitch_um": "pixel pitch", "active_nx": "active width",
            "active_ny": "active height", "phase_stroke_rad": "phase stroke",
            "phase_lut": "measured phase LUT", "camera_magnification": "camera magnification",
            "camera_rotation_deg": "camera rotation", "flip_x": "x parity", "flip_y": "y parity",
        }
        for attr, label in labels.items():
            if getattr(self, attr) is None:
                missing.append(label)
        if not self.polarization_verified:
            missing.append("polarization/phase-modulation validity")
        return missing


def angle_to_phase_gradient(theta_rad: float, wavelength_m: float) -> float:
    return 2 * np.pi * np.sin(theta_rad) / wavelength_m


def phase_gradient_to_angle(gradient_rad_m: float, wavelength_m: float) -> float:
    value = gradient_rad_m * wavelength_m / (2 * np.pi)
    if abs(value) > 1:
        raise ValueError("Phase gradient implies a non-propagating steering angle")
    return float(np.arcsin(value))


def phase_gradient_to_grating_period(gradient_rad_m: float) -> float:
    return np.inf if gradient_rad_m == 0 else float(2 * np.pi / abs(gradient_rad_m))


def predict_free_space_shift(theta_rad: float, distance_m: float) -> float:
    return float(distance_m * np.tan(theta_rad))


def make_tiptilt_phase(x_m: np.ndarray, y_m: np.ndarray, theta_x_rad: float,
                       theta_y_rad: float, wavelength_m: float) -> np.ndarray:
    k = 2 * np.pi / wavelength_m
    return k * (np.sin(theta_x_rad) * x_m + np.sin(theta_y_rad) * y_m)


def make_vortex_phase(x: np.ndarray, y: np.ndarray, ell: int,
                      x0: float = 0.0, y0: float = 0.0) -> np.ndarray:
    return ell * np.arctan2(y - y0, x - x0)


def transform_points(points_xy: np.ndarray, scale: float, rotation_deg: float,
                     flip_x: bool, flip_y: bool, offset_xy=(0.0, 0.0)) -> np.ndarray:
    pts = np.asarray(points_xy, float).copy()
    pts[:, 0] *= -1 if flip_x else 1
    pts[:, 1] *= -1 if flip_y else 1
    a = np.deg2rad(rotation_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return scale * pts @ rot.T + np.asarray(offset_xy, float)


def topology_winding(phase: np.ndarray) -> float:
    wrapped_steps = np.angle(np.exp(1j * np.diff(np.r_[phase, phase[0]])))
    return float(wrapped_steps.sum() / (2 * np.pi))


def dark_core_ratio(image: np.ndarray, cy: float, cx: float, ring_radius_px: float,
                    core_fraction: float = 0.35, ring_half_width_px: float = 2.0) -> float:
    """Mean central intensity divided by mean principal-annulus intensity."""
    yy, xx = np.indices(image.shape)
    rr = np.hypot(xx - cx, yy - cy)
    core = rr <= max(1.0, core_fraction * ring_radius_px)
    annulus = np.abs(rr - ring_radius_px) <= max(1.0, ring_half_width_px)
    if not np.any(core) or not np.any(annulus):
        return np.nan
    return float(np.mean(image[core]) / max(np.mean(image[annulus]), 1e-12))


def radial_profile_1d(image: np.ndarray, cy: float, cx: float, dr: float = 1.0):
    yy, xx = np.indices(image.shape)
    rr = np.hypot(xx - cx, yy - cy)
    bins = np.arange(0, rr.max() + dr, dr)
    idx = np.digitize(rr.ravel(), bins)
    sums = np.bincount(idx, weights=image.ravel(), minlength=len(bins) + 1)
    nums = np.bincount(idx, minlength=len(bins) + 1)
    return 0.5 * (bins[:-1] + bins[1:]), sums[1:len(bins)] / np.maximum(nums[1:len(bins)], 1)


def vortex_bessel_target_profile(r_px: np.ndarray, ell: int, principal_radius_px: float,
                                 envelope_radius_px: float | None = None) -> np.ndarray:
    """Level-3 finite-envelope J_ell target, scaled only by measured ring radius."""
    if ell == 0:
        raise ValueError("Zero-order Bessel target is forbidden for this correction framework")
    try:
        from scipy.special import jnp_zeros, jv
    except ImportError as exc:
        raise RuntimeError("SciPy is required for the vortex-Bessel target") from exc
    # First positive maximum of J_l occurs at first positive zero of J_l'.
    peak_argument = float(jnp_zeros(abs(ell), 1)[0])
    kr = peak_argument / max(principal_radius_px, 1e-12)
    envelope_radius_px = envelope_radius_px or max(3 * principal_radius_px, 1.0)
    target = jv(abs(ell), kr * r_px) ** 2 * np.exp(-2 * (r_px / envelope_radius_px) ** 2)
    return target / max(float(np.max(target)), 1e-12)


def radial_profile_similarity(profile: np.ndarray, target: np.ndarray) -> float:
    a = np.asarray(profile, float); b = np.asarray(target, float)
    n = min(len(a), len(b)); a = a[:n]; b = b[:n]
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def contour_winding(field: np.ndarray, cy: float, cx: float, radius_px: float,
                    samples: int = 2048) -> float:
    """Numerical phase winding on a circular contour around the singularity."""
    try:
        from scipy.ndimage import map_coordinates
    except ImportError as exc:
        raise RuntimeError("SciPy is required for contour winding") from exc
    theta = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    xs = cx + radius_px * np.cos(theta); ys = cy + radius_px * np.sin(theta)
    real = map_coordinates(np.real(field), [ys, xs], order=1, mode="constant", cval=0)
    imag = map_coordinates(np.imag(field), [ys, xs], order=1, mode="constant", cval=0)
    phase = np.angle(real + 1j * imag)
    return topology_winding(phase)


def vortex_stack_gate(before_rows: pd.DataFrame, after_rows: pd.DataFrame,
                      target_ell: int, minimum_improvement_percent: float = 5.0,
                      max_dark_core_increase: float = 0.05,
                      max_ring_change_percent: float = 5.0,
                      max_width_change_percent: float = 10.0,
                      winding_tolerance: float = 0.75,
                      axis_improvement_pass: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Hard-veto, full-stack gate for one correction applied at every z plane."""
    if target_ell == 0:
        raise ValueError("TARGET_ELL=0 is forbidden")
    required = {"z_rel_mm", "azimuthal", "dark_core", "ring_radius", "ring_width",
                "radial_similarity", "power", "winding"}
    if not required.issubset(before_rows) or not required.issubset(after_rows):
        raise ValueError(f"Missing vortex gate columns: {required - set(before_rows)}")
    rows = []
    for b, a in zip(before_rows.itertuples(index=False), after_rows.itertuples(index=False)):
        az_gain = 100 * (b.azimuthal - a.azimuthal) / max(b.azimuthal, 1e-12)
        ring_change = 100 * abs(a.ring_radius / max(b.ring_radius, 1e-12) - 1)
        width_change = 100 * abs(a.ring_width / max(b.ring_width, 1e-12) - 1)
        rows.append({
            "z_rel_mm": b.z_rel_mm,
            "axis_improvement_pass": bool(axis_improvement_pass),
            "azimuthal_uniformity_pass": az_gain >= minimum_improvement_percent,
            "ring_radius_pass": ring_change <= max_ring_change_percent,
            "ring_width_pass": width_change <= max_width_change_percent,
            "dark_core_pass": a.dark_core <= b.dark_core + max_dark_core_increase,
            "radial_profile_pass": a.radial_similarity >= b.radial_similarity - 0.02,
            "topology_pass": abs(a.winding - target_ell) <= winding_tolerance,
            "power_pass": a.power >= 0.90 * b.power,
            "azimuthal_improvement_percent": az_gain,
            "dark_core_before": b.dark_core, "dark_core_after": a.dark_core,
            "winding_after": a.winding,
        })
    per_plane = pd.DataFrame(rows)
    hard = ["ring_radius_pass", "ring_width_pass", "dark_core_pass",
            "radial_profile_pass", "topology_pass", "power_pass"]
    all_hard = bool(per_plane[hard].all(axis=None))
    stack_az_gain = 100 * (before_rows.azimuthal.mean() - after_rows.azimuthal.mean()) / max(before_rows.azimuthal.mean(), 1e-12)
    accepted = all_hard and axis_improvement_pass and stack_az_gain >= minimum_improvement_percent
    failed = [c for c in hard if not per_plane[c].all()]
    return per_plane, {
        "accepted": accepted,
        "target_ell": target_ell,
        "stack_azimuthal_improvement_percent": float(stack_az_gain),
        "all_planes_hard_gates_pass": all_hard,
        "axis_improvement_pass": bool(axis_improvement_pass),
        "failed_hard_gates": failed,
        "reason": "ACCEPTED" if accepted else "REJECTED — TARGET VORTEX MORPHOLOGY/STACK GATE FAILED",
    }


def correction_gate(before: dict[str, float], after: dict[str, float],
                    minimum_improvement_percent: float = 5.0,
                    maximum_scale_change_percent: float = 5.0,
                    maximum_power_loss_percent: float = 10.0) -> dict[str, Any]:
    asym_gain = 100 * (before["asymmetry"] - after["asymmetry"]) / max(before["asymmetry"], 1e-12)
    scale_change = 100 * abs(after["ring_radius"] / max(before["ring_radius"], 1e-12) - 1)
    power_loss = 100 * max(0.0, 1 - after["power"] / max(before["power"], 1e-12))
    accepted = (asym_gain >= minimum_improvement_percent and
                scale_change <= maximum_scale_change_percent and
                power_loss <= maximum_power_loss_percent)
    return {"accepted": accepted, "asymmetry_improvement_percent": asym_gain,
            "ring_scale_change_percent": scale_change, "power_loss_percent": power_loss}


def correlation_table(df: pd.DataFrame, observables: Iterable[str],
                      physical_col="z_rel_mm", acquisition_col="acquisition_index") -> pd.DataFrame:
    rows = []
    for name in observables:
        clean = df[[name, physical_col, acquisition_col]].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"observable": name, "n": len(clean),
                     "corr_physical_z": clean[name].corr(clean[physical_col]),
                     "corr_acquisition_index": clean[name].corr(clean[acquisition_col]),
                     "interpretation": "DEGENERATE: acquisition order is monotonic with physical z"
                     if abs(clean[physical_col].corr(clean[acquisition_col])) > 0.95 else "SEPARABLE"})
    return pd.DataFrame(rows)


def build_error_budget(metrics: pd.DataFrame, qc: pd.DataFrame,
                       ring_angle_mrad: float, correction_accepted: bool,
                       slm1: SLMCalibration, slm2: SLMCalibration) -> pd.DataFrame:
    sat = float(qc.saturation_fraction.max())
    asym = float(metrics.azimuthal_modulation.max())
    coverage = float(metrics.ring_angular_coverage.min())
    rows = [
        ("geometric tip/tilt", Evidence.SUPPORTED if ring_angle_mrad > .25 else Evidence.WEAK,
         f"{ring_angle_mrad:.3f} mrad", Correctability.PARTIAL, "mechanically align first; test low-gain ramp only after SLM dither"),
        ("amplitude imbalance", Evidence.SUPPORTED if asym > .35 else Evidence.WEAK,
         f"max modulation {asym:.3f}", Correctability.PARTIAL, "inspect illumination/order selection; phase-only repair is not guaranteed"),
        ("camera saturation", Evidence.SUPPORTED if sat >= .98 else Evidence.RULED_OUT,
         f"max ADC fraction {sat:.3f}", Correctability.NO, "repeat affected planes at lower exposure"),
        ("physical clipping", Evidence.POSSIBLE if coverage < .75 else Evidence.WEAK,
         f"minimum angular coverage {coverage:.3f}", Correctability.NO, "inspect iris, apertures, and ROI"),
        ("low-order aberration correction", Evidence.WEAK if not correction_accepted else Evidence.SUPPORTED,
         "numerical hypothesis", Correctability.UNKNOWN, "do not apply unless all morphology gates pass"),
        ("SLM1 vortex-origin offset", Evidence.DEGENERATE, "no SLM1-only perturbation", Correctability.UNKNOWN, "capture SLM1 vortex-origin dither"),
        ("SLM2 vortex-origin offset", Evidence.DEGENERATE, "no SLM2-only perturbation", Correctability.UNKNOWN, "capture SLM2 vortex-origin dither"),
        ("relative SLM rotation/scale", Evidence.UNMEASURED, "no response matrix", Correctability.UNKNOWN, "capture rotation/scale dither"),
        ("4F order contamination", Evidence.POSSIBLE, "not identifiable from z-scan alone", Correctability.NO, "image Fourier plane and block orders independently"),
        ("polarization validity", Evidence.UNMEASURED, "not recorded", Correctability.NO, "verify input and both SLM analyser settings"),
        ("SLM1 calibration", Evidence.UNMEASURED if slm1.blockers() else Evidence.SUPPORTED,
         "; ".join(slm1.blockers()) or "complete", Correctability.UNKNOWN, "measure LUT and camera transform"),
        ("SLM2 calibration", Evidence.UNMEASURED if slm2.blockers() else Evidence.SUPPORTED,
         "; ".join(slm2.blockers()) or "complete", Correctability.UNKNOWN, "measure LUT and camera transform"),
        ("coherent ghost / zero order", Evidence.POSSIBLE, "phase-sensitive/order data absent", Correctability.NO, "capture carrier/iris and blocked-order controls"),
    ]
    return pd.DataFrame(rows, columns=["error_source", "evidence", "measured_magnitude",
                                       "slm_correctable", "recommended_action"])


def calibration_queue() -> pd.DataFrame:
    items = [
        (1, "SLM1 steering response", "capture baseline and +/-x, +/-y phase-ramp dithers at low gain"),
        (2, "SLM2 steering response", "repeat identical independent ramp dithers"),
        (3, "SLM1 vortex-origin response", "capture +/-x, +/-y digital vortex-origin shifts"),
        (4, "SLM2 vortex-origin response", "repeat independent origin shifts"),
        (5, "Gaussian Shack-Hartmann wavefront", "measure ordinary system aberration without the l=20 target"),
        (6, "phase LUT and polarization", "measure grey-to-phase curves and verify LC alignment on both panels"),
        (7, "4F order isolation", "image Fourier plane; capture zero/+1/-1 orders separately and verify iris"),
        (8, "lower-exposure z scan", "repeat saturated planes with identical geometry"),
    ]
    return pd.DataFrame(items, columns=["priority", "experiment", "minimum_measurement"])


def require_hardware_ready(slm: SLMCalibration) -> None:
    if not slm.hardware_ready:
        raise RuntimeError(f"{slm.name} HARDWARE EXPORT BLOCKED: " + ", ".join(slm.blockers()))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
