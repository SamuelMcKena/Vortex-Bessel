"""Discrete N-fold lattice/kaleidoscope non-diffracting beam helpers.

I keep this module as a labelled reference for the finite-plane-wave family:
discrete deltas on the Bessel k-ring make periodic lattices that tile space.
They are useful kaleidoscope/discrete-nondiffracting beams, but they are **not**
the localized polygonal vortex ring built by `vbb_polygonal`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vbb_study.config import EPS as BT_EPS, um as BT_UM
from vbb_study.equations.fields import gaussian_amplitude
from vbb_study.facade import core as _bt
from . import vbb_materials, vbb_metrics, vbb_style
from .equations import holography as holography_eq


@dataclass(frozen=True)
class DiscretePattern:
    """One N-wave lattice/kaleidoscope preset definition."""

    name: str
    N: int
    phases_rad: tuple[float, ...]
    vortex_charge: int = 0
    orientation_rad: float = 0.0


def pattern_preset(name: str, *, orientation_rad: float = 0.0, vortex_charge: int = 1) -> DiscretePattern:
    """Return one named N-fold lattice/kaleidoscope phase preset."""

    key = str(name).lower().strip().replace("-", "_")
    if key == "triangular":
        phases = tuple(float(2.0 * np.pi * n / 3.0) for n in range(3))
        return DiscretePattern("triangular", 3, phases, 0, orientation_rad)
    if key == "square":
        return DiscretePattern("square", 4, (0.0, 0.0, 0.0, 0.0), 0, orientation_rad)
    if key == "hexagonal":
        return DiscretePattern("hexagonal", 6, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0, orientation_rad)
    if key == "kagome":
        return DiscretePattern("kagome", 6, (0.0, 0.0, 2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0, 4.0 * np.pi / 3.0), 0, orientation_rad)
    if key == "honeycomb":
        return DiscretePattern("honeycomb", 6, (0.0, np.pi, 0.0, np.pi, 0.0, np.pi), 0, orientation_rad)
    if key in {"hexagonal_vortex", "hex_vortex"}:
        angles = 2.0 * np.pi * np.arange(6) / 6.0 + float(orientation_rad)
        phases = tuple(float(vortex_charge * a) for a in angles)
        return DiscretePattern("hexagonal_vortex", 6, phases, int(vortex_charge), orientation_rad)
    raise ValueError("Unknown discrete pattern preset.")


def wave_angles(pattern: DiscretePattern) -> np.ndarray:
    """Return the azimuthal plane-wave angles in radians."""

    return 2.0 * np.pi * np.arange(int(pattern.N)) / int(pattern.N) + float(pattern.orientation_rad)


def n_wave_complex_field(
    grid: Mapping[str, Any],
    kr_m_inv: float,
    pattern: DiscretePattern,
    *,
    waist_m: float | None = None,
) -> np.ndarray:
    """Return the ideal finite N-wave lattice field.

    This intentionally produces a periodic tiled pattern, not one localized
    polygonal ring.
    """

    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    U = np.zeros_like(X, dtype=complex)
    for angle, phase in zip(wave_angles(pattern), pattern.phases_rad):
        U += np.exp(1j * (float(kr_m_inv) * (X * np.cos(angle) + Y * np.sin(angle)) + float(phase)))
    U /= np.sqrt(float(pattern.N))
    if waist_m is not None:
        U *= gaussian_amplitude(np.asarray(grid["R"], dtype=float), float(waist_m))
    return U


def hexagonal_axicon_phase(grid: Mapping[str, Any], kr_m_inv: float, pattern: DiscretePattern) -> np.ndarray:
    """Return a piecewise linear generalized-axicon phase for one N-fold beam."""

    phi = np.asarray(grid["PHI"], dtype=float)
    angles = wave_angles(pattern)
    diffs = np.angle(np.exp(1j * (phi[..., None] - angles[None, None, :])))
    nearest = np.argmin(np.abs(diffs), axis=-1)
    selected_angles = angles[nearest]
    selected_phase = np.asarray(pattern.phases_rad, dtype=float)[nearest]
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    return float(kr_m_inv) * (X * np.cos(selected_angles) + Y * np.sin(selected_angles)) + selected_phase


def hexagonal_axicon_field(
    grid: Mapping[str, Any],
    kr_m_inv: float,
    pattern: DiscretePattern,
    *,
    waist_m: float | None = None,
) -> np.ndarray:
    """Return the phase-only generalized-axicon alternative field.

    I use this as the notebook's second encoding: it should preserve the same
    N-fold symmetry order as the explicit N-wave sum even when the detailed
    lattice contrast differs.
    """

    U = np.exp(1j * hexagonal_axicon_phase(grid, kr_m_inv, pattern))
    if waist_m is not None:
        U *= gaussian_amplitude(np.asarray(grid["R"], dtype=float), float(waist_m))
    return U


def phase_only_lab_field(
    target_field: np.ndarray,
    *,
    phase_bits: int = 8,
    method: str = "phase_only",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a phase-only lab proxy field and encoding metadata."""

    target = np.asarray(target_field, dtype=complex)
    phase = np.angle(target)
    amp = np.abs(target)
    amp_norm = amp / (float(np.max(amp)) + BT_EPS)
    key = str(method).lower().strip()
    if key in {"complex", "complex_amplitude", "method_b", "b"}:
        encoded_power_fraction = float(np.mean(amp_norm**2))
        phase = phase + np.arccos(np.clip(amp_norm, 0.0, 1.0))
        method_label = "complex_amplitude"
    else:
        encoded_power_fraction = 1.0
        method_label = "phase_only"
    wrapped = holography_eq.wrap_phase_rad(phase)
    quantized = holography_eq.quantize_phase_rad(wrapped, int(phase_bits))
    return np.exp(1j * quantized), {
        "phase_wrapped_rad": wrapped,
        "phase_quantized_rad": quantized,
        "gray": holography_eq.phase_to_gray(quantized, bits=int(phase_bits)),
        "encoded_power_fraction": encoded_power_fraction,
        "encoding_method": method_label,
    }


def propagate_discrete_case(
    grid: Mapping[str, Any],
    pattern: DiscretePattern,
    *,
    kr_m_inv: float,
    waist_m: float,
    wavelength_m: float,
    n_medium: float,
    z_values_m: Sequence[float],
    path: str,
    phase_bits: int = 8,
    crop_pixels: int | None = None,
    encoding_method: str = "phase_only",
) -> dict[str, Any]:
    """Build and propagate one ideal or lab-realistic discrete-beam case."""

    target = n_wave_complex_field(grid, kr_m_inv, pattern, waist_m=waist_m)
    if str(path).lower().strip() == "ideal":
        U0 = target
        encoding = {"encoded_power_fraction": 1.0, "encoding_method": "complex_ideal"}
    else:
        U0, encoding = phase_only_lab_field(target, phase_bits=phase_bits, method=encoding_method)
        U0 = U0 * gaussian_amplitude(np.asarray(grid["R"], dtype=float), waist_m)
    volume = _bt().propagate_volume(
        U0,
        dict(grid),
        wavelength_m,
        z_values_m,
        n_medium=n_medium,
        crop_pixels=crop_pixels,
        bandlimit=True,
        method="bl_asm",
    )
    return {
        "pattern": pattern,
        "path": str(path),
        "grid": dict(grid),
        "U0": U0,
        "target_field": target,
        "volume": volume,
        "encoding": encoding,
        "kr_m_inv": float(kr_m_inv),
        "waist_m": float(waist_m),
        "wavelength_m": float(wavelength_m),
        "n_medium": float(n_medium),
    }


def angular_profile(image: np.ndarray, grid: Mapping[str, Any], radius_m: float, *, samples: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """Sample intensity on a circular contour for azimuthal-FFT metrics."""

    x = np.asarray(grid["x"], dtype=float)
    dx = float(grid["dx"])
    theta = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False)
    xs = float(radius_m) * np.cos(theta)
    ys = float(radius_m) * np.sin(theta)
    arr = np.asarray(image, dtype=float)
    ix = np.clip(np.round((xs - float(x[0])) / dx).astype(int), 0, arr.shape[1] - 1)
    iy = np.clip(np.round((ys - float(x[0])) / dx).astype(int), 0, arr.shape[0] - 1)
    return theta, arr[iy, ix]


def symmetry_metrics(
    image: np.ndarray,
    grid: Mapping[str, Any],
    *,
    expected_N: int,
    radius_m: float,
) -> dict[str, float]:
    """Measure rotational order, contrast, and vertex/edge radius proxies."""

    theta, prof = angular_profile(image, grid, radius_m)
    centered = prof - float(np.mean(prof))
    coeff = np.fft.rfft(centered)
    amps = np.abs(coeff)
    amps[0] = 0.0
    peak_order = int(np.argmax(amps)) if amps.size else 0
    expected_amp = float(amps[int(expected_N)]) if int(expected_N) < amps.size else 0.0
    other = np.array(amps, copy=True)
    if int(expected_N) < other.size:
        other[int(expected_N)] = 0.0
    order_fidelity = float(expected_amp / (float(np.max(other)) + expected_amp + BT_EPS))
    contrast = float((np.max(prof) - np.min(prof)) / (np.max(prof) + np.min(prof) + BT_EPS))
    phase = float(np.angle(coeff[int(expected_N)])) if int(expected_N) < coeff.size else 0.0
    orientation = float((-phase / max(int(expected_N), 1)) % (2.0 * np.pi / max(int(expected_N), 1)))
    return {
        "expected_N": float(expected_N),
        "azimuthal_fft_peak_order": float(peak_order),
        "order_N_fidelity": order_fidelity,
        "azimuthal_contrast": contrast,
        "orientation_rad": orientation,
        "vertex_radius_um": float(radius_m / BT_UM),
        "flat_to_flat_radius_um": float(radius_m * np.cos(np.pi / max(int(expected_N), 3)) / BT_UM),
        "lattice_spacing_um": float(2.0 * np.pi * float(radius_m) / max(float(expected_N), 1.0) / BT_UM),
    }


def best_symmetry_metrics(
    image: np.ndarray,
    grid: Mapping[str, Any],
    *,
    expected_N: int,
    kr_m_inv: float,
    radius_m: float | None = None,
    min_radius_m: float | None = None,
    max_radius_m: float | None = None,
    candidates: int = 56,
) -> dict[str, float]:
    """Choose a documented analysis contour for the order-N FFT metric.

    A finite plane-wave lattice can have weak or zero order-N content on some
    radii even when the global motif has that symmetry. I therefore scan a
    modest annular interval and keep the contour with the strongest expected
    order-N harmonic. The selected radius is reported, so the notebook can
    distinguish physics from a plotting choice.
    """

    if radius_m is not None:
        metrics = symmetry_metrics(image, grid, expected_N=expected_N, radius_m=float(radius_m))
        metrics["analysis_radius_um"] = float(radius_m / BT_UM)
        return metrics

    dx = float(grid["dx"])
    n = int(grid["N"])
    half_window = 0.48 * n * dx
    low = float(min_radius_m) if min_radius_m is not None else max(0.55 / max(float(kr_m_inv), BT_EPS), 2.5 * dx)
    high = float(max_radius_m) if max_radius_m is not None else min(8.0 / max(float(kr_m_inv), BT_EPS), half_window)
    if not np.isfinite(high) or high <= low:
        high = min(half_window, max(low + 4.0 * dx, 2.0 * np.pi / max(float(kr_m_inv), BT_EPS)))
    best: dict[str, float] | None = None
    for r_m in np.linspace(low, high, int(candidates)):
        m = symmetry_metrics(image, grid, expected_N=expected_N, radius_m=float(r_m))
        score = float(m["order_N_fidelity"]) + 0.05 * float(m["azimuthal_contrast"])
        if best is None or score > float(best["_score"]):
            best = dict(m)
            best["_score"] = score
            best["analysis_radius_um"] = float(r_m / BT_UM)
    assert best is not None
    best.pop("_score", None)
    return best


def per_z_symmetry(case: Mapping[str, Any], *, radius_m: float) -> pd.DataFrame:
    """Return symmetry metrics along z for one propagated discrete beam."""

    vol = case["volume"]
    grid = vol["crop_grid"]
    rows = []
    for idx, z_m in enumerate(np.asarray(vol["z"], dtype=float)):
        metrics = symmetry_metrics(
            np.asarray(vol["intensity_stack"][idx], dtype=float),
            grid,
            expected_N=int(case["pattern"].N),
            radius_m=radius_m,
        )
        rows.append({"z_um": float(z_m / BT_UM), **metrics})
    return pd.DataFrame(rows)


def threshold_polygon_metrics(case: Mapping[str, Any], pulse_energy_J: float, threshold_J_cm2: float) -> dict[str, Any]:
    """Map a transverse N-fold profile to threshold-contour material metrics."""

    vol = case["volume"]
    idx = int(vol["peak_index"])
    plane = np.asarray(vol["intensity_stack"][idx], dtype=float)
    grid = vol["crop_grid"]
    F = vbb_materials.fluence_from_intensity(plane, grid["dx"], pulse_energy_J)
    mask = F >= float(threshold_J_cm2)
    theta = np.mod(np.arctan2(grid["Y"], grid["X"]), 2.0 * np.pi)
    sector_means = []
    for n in range(int(case["pattern"].N)):
        lo = 2.0 * np.pi * n / int(case["pattern"].N)
        hi = 2.0 * np.pi * (n + 1) / int(case["pattern"].N)
        sector = mask & (theta >= lo) & (theta < hi)
        sector_means.append(float(np.mean(F[sector])) if np.any(sector) else np.nan)
    vals = np.asarray(sector_means, dtype=float)
    uniformity = float(1.0 / (1.0 + np.nanstd(vals) / (np.nanmean(vals) + BT_EPS))) if np.any(np.isfinite(vals)) else np.nan
    return {
        "threshold_area_um2": float(np.count_nonzero(mask) * grid["dx"] ** 2 / (BT_UM**2)),
        "vertex_edge_fluence_uniformity": uniformity,
        "sector_fluence_J_cm2": vals,
        "fluence_xy_J_cm2": F,
        "threshold_mask": mask,
    }


def continuous_limit_error(
    grid: Mapping[str, Any],
    *,
    kr_m_inv: float,
    waist_m: float,
    N_values: Sequence[int] = (6, 12, 24, 48),
) -> pd.DataFrame:
    """Compare large-N plane-wave sums against the continuous J0 Bessel limit.

    I compare azimuthally averaged, normalized intensity profiles. The absolute
    field phase is not the observable here; the useful check is that the radial
    intensity tends toward `J_0(k_r r)^2` under the same Gaussian envelope.
    """

    try:
        from scipy import special as sp
    except Exception:  # pragma: no cover
        sp = None

    R = np.asarray(grid["R"], dtype=float)
    bessel = np.cos(float(kr_m_inv) * R) if sp is None else sp.j0(float(kr_m_inv) * R)
    target = (bessel * gaussian_amplitude(R, waist_m)) ** 2
    target_avg = vbb_metrics.azimuthal_average(target, grid, center_mode="origin")
    target_profile = np.asarray(target_avg["profile"], dtype=float)
    target_profile = target_profile / (float(np.nanmax(target_profile)) + BT_EPS)
    rows: list[dict[str, float]] = []
    for N in N_values:
        pattern = DiscretePattern(f"N{int(N)}", int(N), tuple(0.0 for _ in range(int(N))))
        I = np.abs(n_wave_complex_field(grid, kr_m_inv, pattern, waist_m=waist_m)) ** 2
        avg = vbb_metrics.azimuthal_average(I, grid, center_mode="origin")
        profile = np.asarray(avg["profile"], dtype=float)
        profile = profile / (float(np.nanmax(profile)) + BT_EPS)
        finite = np.isfinite(profile) & np.isfinite(target_profile)
        err = float(
            np.sqrt(np.mean((profile[finite] - target_profile[finite]) ** 2))
            / (np.sqrt(np.mean(target_profile[finite] ** 2)) + BT_EPS)
        )
        rows.append({"N": int(N), "continuous_N_bessel_l2_error": err})
    return pd.DataFrame(rows)


def export_cgh(case: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write an 8-bit CGH and params JSON for one discrete pattern."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pattern = case["pattern"]
    encoding = case.get("encoding", {})
    gray = np.asarray(encoding.get("gray", holography_eq.phase_to_gray(np.angle(case["U0"]), bits=8)), dtype=np.uint8)
    name = f"10_discrete_{pattern.name}_{case['path']}".lower()
    png_path = out / f"{name}.png"
    try:
        from PIL import Image

        Image.fromarray(gray, mode="L").save(png_path)
    except Exception:  # pragma: no cover
        plt.imsave(png_path, gray, cmap="gray", vmin=0, vmax=255)
    params = {
        "pattern": pattern.name,
        "N": int(pattern.N),
        "phases_rad": list(map(float, pattern.phases_rad)),
        "orientation_rad": float(pattern.orientation_rad),
        "vortex_charge": int(pattern.vortex_charge),
        "path": case["path"],
        "kr_m_inv": float(case["kr_m_inv"]),
        "waist_m": float(case["waist_m"]),
        "encoded_power_fraction": float(encoding.get("encoded_power_fraction", np.nan)),
        "encoding_method": str(encoding.get("encoding_method", "")),
    }
    json_path = out / f"{name}.json"
    json_path.write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")
    return {"phase_png": png_path, "params_json": json_path}


def run_pattern_suite(
    grid: Mapping[str, Any],
    patterns: Sequence[DiscretePattern],
    *,
    kr_m_inv: float,
    waist_m: float,
    wavelength_m: float,
    n_medium: float,
    z_values_m: Sequence[float],
    crop_pixels: int | None,
    pulse_energy_J: float,
    threshold_J_cm2: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run ideal and lab-realistic discrete-pattern cases and return summary rows."""

    cases: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        for path in ("ideal", "lab_realistic"):
            case = propagate_discrete_case(
                grid,
                pattern,
                kr_m_inv=kr_m_inv,
                waist_m=waist_m,
                wavelength_m=wavelength_m,
                n_medium=n_medium,
                z_values_m=z_values_m,
                path="ideal" if path == "ideal" else "lab",
                crop_pixels=crop_pixels,
                encoding_method="complex_amplitude" if path != "ideal" else "phase_only",
            )
            sym = best_symmetry_metrics(
                case["volume"]["planes"]["peak"],
                case["volume"]["crop_grid"],
                expected_N=pattern.N,
                kr_m_inv=kr_m_inv,
            )
            radius_m = float(sym["analysis_radius_um"]) * BT_UM
            per_z = per_z_symmetry(case, radius_m=radius_m)
            mat = threshold_polygon_metrics(case, pulse_energy_J, threshold_J_cm2)
            row = {
                "pattern": pattern.name,
                "N": int(pattern.N),
                "path": path,
                "peak_order": int(round(sym["azimuthal_fft_peak_order"])),
                "order_N_fidelity": sym["order_N_fidelity"],
                "azimuthal_contrast": sym["azimuthal_contrast"],
                "symmetry_stability_std": float(per_z["order_N_fidelity"].std()),
                "contrast_stability_std": float(per_z["azimuthal_contrast"].std()),
                "vertex_radius_um": sym["vertex_radius_um"],
                "flat_to_flat_radius_um": sym["flat_to_flat_radius_um"],
                "analysis_radius_um": sym["analysis_radius_um"],
                "lattice_spacing_um": float(2.0 * np.pi / max(float(kr_m_inv), BT_EPS) / BT_UM),
                "encoded_power_fraction": float(case["encoding"].get("encoded_power_fraction", np.nan)),
                "threshold_area_um2": mat["threshold_area_um2"],
                "vertex_edge_fluence_uniformity": mat["vertex_edge_fluence_uniformity"],
            }
            case["summary_metrics"] = row
            case["per_z_symmetry"] = per_z
            case["material_metrics"] = mat
            case["analysis_radius_m"] = radius_m
            cases.append(case)
            rows.append(row)
    return pd.DataFrame(rows), cases


def plot_pattern_gallery(cases: Sequence[Mapping[str, Any]], output_path: str | Path) -> Path:
    """Save a transverse ideal-vs-lab gallery for discrete beams."""

    vbb_style.apply_style()
    patterns = list(dict.fromkeys(c["pattern"].name for c in cases))
    fig, axes = plt.subplots(len(patterns), 2, figsize=(7.6, 3.0 * len(patterns)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    artist = None
    for row, name in enumerate(patterns):
        subset = [c for c in cases if c["pattern"].name == name]
        for col, path in enumerate(("ideal", "lab")):
            case = next(c for c in subset if c["path"] == path)
            img = np.asarray(case["volume"]["planes"]["peak"], dtype=float)
            grid = case["volume"]["crop_grid"]
            x_um = np.asarray(grid["x"], dtype=float) / BT_UM
            artist = axes[row, col].imshow(
                vbb_style.display_scale(img, gamma=0.45),
                origin="lower",
                extent=[float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])],
                cmap=vbb_style.INTENSITY_CMAP,
                vmin=0.0,
                vmax=1.0,
            )
            axes[row, col].set_title(f"{name} | {path}")
            axes[row, col].set_xlabel("x [um, sample plane]")
            axes[row, col].set_ylabel("y [um, sample plane]")
    if artist is not None:
        cb = fig.colorbar(artist, ax=axes, shrink=0.94)
        cb.set_label("display intensity, gamma=0.45 [a.u.]")
    caption = "Ideal and lab-realistic transverse intensity gallery for discrete N-fold nondiffracting beams."
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "discrete_pattern_gallery"})
    plt.close(fig)
    return out


def plot_symmetry_summary(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save azimuthal-order fidelity and contrast degradation summary."""

    vbb_style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    for path, group in df.groupby("path"):
        axes[0].plot(group["pattern"], group["order_N_fidelity"], marker="o", label=path)
        axes[1].plot(group["pattern"], group["azimuthal_contrast"], marker="o", label=path)
    axes[0].set_ylabel("order-N fidelity [0-1]")
    axes[1].set_ylabel("azimuthal contrast [0-1]")
    for ax in axes:
        ax.set_xlabel("pattern")
        ax.tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False)
    caption = "Discrete-beam symmetry fidelity and azimuthal contrast for ideal and lab-realistic encodings."
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "discrete_symmetry_summary"})
    plt.close(fig)
    return out


__all__ = [
    "DiscretePattern",
    "export_cgh",
    "best_symmetry_metrics",
    "continuous_limit_error",
    "hexagonal_axicon_field",
    "hexagonal_axicon_phase",
    "n_wave_complex_field",
    "pattern_preset",
    "per_z_symmetry",
    "phase_only_lab_field",
    "plot_pattern_gallery",
    "plot_symmetry_summary",
    "propagate_discrete_case",
    "run_pattern_suite",
    "symmetry_metrics",
    "threshold_polygon_metrics",
    "wave_angles",
]
