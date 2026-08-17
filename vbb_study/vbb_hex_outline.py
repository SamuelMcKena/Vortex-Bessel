"""Hollow regular-hexagon outline targets and phase-only hologram design."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import vbb_style
from vbb_study.config import EPS as BT_EPS, TWOPI as BT_TWOPI, TwinConfig, um as BT_UM
from vbb_study.equations.fields import (
    fft2c,
    gaussian_amplitude,
    ifft2c,
    make_rect_grid,
    make_xy_grid,
    phase_to_gray,
    phase_wrap,
    quantize_phase,
)
from vbb_study.equations.holography import fill_factor_amplitude as _fill_factor_amplitude
from vbb_study.equations.interface import interface_aberration_pupil, interface_correction_phase
from vbb_study.equations.propagation import bandlimit_mask_matsushima, focus_to_focal_plane

try:
    from scipy.ndimage import label as nd_label
except Exception:  # pragma: no cover
    nd_label = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


def _fill_factor_amplitude_grid(grid: Mapping[str, Any], pixel_pitch_m: float, fill_factor: float) -> np.ndarray:
    return _fill_factor_amplitude(
        grid["X"],
        grid["Y"],
        pixel_pitch_m=pixel_pitch_m,
        fill_factor=fill_factor,
        sample_spacing_m=float(grid["dx"]),
    )


@dataclass(frozen=True)
class HexOutlineConfig:
    """Physical target: one hollow regular-hexagon perimeter."""

    flat_radius_m: float = 7.0e-6
    line_sigma_m: float = 0.65e-6
    orientation_rad: float = 0.0
    retrieval_iterations: int = 180
    roi_margin_m: float = 4.0e-6
    signal_widths: float = 1.5
    target_truncation_widths: float = 3.0
    core_clearance_widths: float = 2.2
    threshold_fraction: float = 0.35
    phase_bits: int = 8
    random_seed: int = 12345


def hex_vertices(config: HexOutlineConfig) -> np.ndarray:
    """Return regular-hexagon vertices as an ``(6, 2)`` array in metres."""

    corner_radius = float(config.flat_radius_m / np.cos(np.pi / 6.0))
    phi = float(config.orientation_rad) + (np.arange(6, dtype=float) + 0.5) * (2.0 * np.pi / 6.0)
    return np.column_stack((corner_radius * np.cos(phi), corner_radius * np.sin(phi)))


def inside_hex_mask(grid: Mapping[str, Any], config: HexOutlineConfig) -> np.ndarray:
    """Return the filled regular-hexagon mask defined by the apothem."""

    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    max_signed = np.full_like(X, -np.inf, dtype=float)
    for idx in range(6):
        theta = float(config.orientation_rad) + idx * (2.0 * np.pi / 6.0)
        signed = X * np.cos(theta) + Y * np.sin(theta) - float(config.flat_radius_m)
        max_signed = np.maximum(max_signed, signed)
    return max_signed <= 0.0


def distance_to_hex_outline(grid: Mapping[str, Any], config: HexOutlineConfig) -> np.ndarray:
    """Return Euclidean distance from each grid point to the hexagon perimeter."""

    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    points_x = X[..., None]
    points_y = Y[..., None]
    verts = hex_vertices(config)
    a = verts
    b = np.roll(verts, -1, axis=0)
    ab = b - a
    ab2 = np.sum(ab * ab, axis=1)
    apx = points_x - a[:, 0]
    apy = points_y - a[:, 1]
    t = np.clip((apx * ab[:, 0] + apy * ab[:, 1]) / np.maximum(ab2, BT_EPS), 0.0, 1.0)
    closest_x = a[:, 0] + t * ab[:, 0]
    closest_y = a[:, 1] + t * ab[:, 1]
    d2 = (points_x - closest_x) ** 2 + (points_y - closest_y) ** 2
    return np.sqrt(np.min(d2, axis=-1))


def hex_outline_target(grid: Mapping[str, Any], config: HexOutlineConfig) -> dict[str, np.ndarray]:
    """Return the outline target amplitude and masks on a sample/focal grid."""

    distance = distance_to_hex_outline(grid, config)
    sigma = max(float(config.line_sigma_m), BT_EPS)
    target = np.exp(-0.5 * (distance / sigma) ** 2)
    target = np.where(distance <= float(config.target_truncation_widths) * sigma, target, 0.0)
    target = target / (float(np.max(target)) + BT_EPS)

    inside = inside_hex_mask(grid, config)
    corner_radius = float(config.flat_radius_m / np.cos(np.pi / 6.0))
    R = np.asarray(grid["R"], dtype=float)
    outline_mask = distance <= float(config.signal_widths) * sigma
    core_mask = inside & (distance >= float(config.core_clearance_widths) * sigma)
    roi_mask = R <= corner_radius + float(config.roi_margin_m)
    outside_mask = roi_mask & (~outline_mask) & (~core_mask)
    return {
        "target_amplitude": target,
        "distance_m": distance,
        "inside_mask": inside,
        "outline_mask": outline_mask,
        "core_mask": core_mask,
        "outside_mask": outside_mask,
        "roi_mask": roi_mask,
    }


def _bilinear_sample(image: np.ndarray, grid: Mapping[str, Any], xs_m: np.ndarray, ys_m: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    x = np.asarray(grid["x"], dtype=float)
    dx = float(grid["dx"])
    fx = (np.asarray(xs_m, dtype=float) - float(x[0])) / dx
    fy = (np.asarray(ys_m, dtype=float) - float(x[0])) / dx
    x0 = np.clip(np.floor(fx).astype(int), 0, arr.shape[1] - 2)
    y0 = np.clip(np.floor(fy).astype(int), 0, arr.shape[0] - 2)
    tx = np.clip(fx - x0, 0.0, 1.0)
    ty = np.clip(fy - y0, 0.0, 1.0)
    return (
        (1.0 - tx) * (1.0 - ty) * arr[y0, x0]
        + tx * (1.0 - ty) * arr[y0, x0 + 1]
        + (1.0 - tx) * ty * arr[y0 + 1, x0]
        + tx * ty * arr[y0 + 1, x0 + 1]
    )


def hex_outline_samples(config: HexOutlineConfig, *, samples_per_side: int = 160) -> dict[str, np.ndarray]:
    verts = hex_vertices(config)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    side_id: list[np.ndarray] = []
    t = np.linspace(0.0, 1.0, int(samples_per_side), endpoint=False)
    for idx in range(6):
        a = verts[idx]
        b = verts[(idx + 1) % 6]
        pts = a[None, :] * (1.0 - t[:, None]) + b[None, :] * t[:, None]
        xs.append(pts[:, 0])
        ys.append(pts[:, 1])
        side_id.append(np.full_like(t, idx, dtype=int))
    return {"x_m": np.concatenate(xs), "y_m": np.concatenate(ys), "side": np.concatenate(side_id)}


def hex_outline_metrics(
    field_or_intensity: np.ndarray,
    grid: Mapping[str, Any],
    config: HexOutlineConfig,
) -> dict[str, Any]:
    """Measure whether intensity is a hollow hexagon outline."""

    if np.iscomplexobj(field_or_intensity):
        intensity = np.abs(np.asarray(field_or_intensity, dtype=complex)) ** 2
    else:
        intensity = np.asarray(field_or_intensity, dtype=float)
    target = hex_outline_target(grid, config)
    outline_mask = target["outline_mask"]
    core_mask = target["core_mask"]
    outside_mask = target["outside_mask"]
    roi_mask = target["roi_mask"]
    peak = float(np.max(intensity[roi_mask])) + BT_EPS if np.any(roi_mask) else float(np.max(intensity)) + BT_EPS
    norm = intensity / peak

    threshold_mask = norm >= float(config.threshold_fraction)
    if nd_label is not None:
        labels, count = nd_label(threshold_mask & roi_mask)
        areas = np.bincount(labels.ravel())[1:]
    else:  # pragma: no cover
        count = int(np.any(threshold_mask & roi_mask))
        areas = np.asarray([np.count_nonzero(threshold_mask & roi_mask)], dtype=int)
    largest = int(np.max(areas)) if areas.size else 0
    largest_fraction = float(largest / (np.sum(areas) + BT_EPS)) if areas.size else 0.0

    hit = threshold_mask & outline_mask
    precision = float(np.count_nonzero(hit) / (np.count_nonzero(threshold_mask & roi_mask) + BT_EPS))
    recall = float(np.count_nonzero(hit) / (np.count_nonzero(outline_mask) + BT_EPS))
    f1 = float(2.0 * precision * recall / (precision + recall + BT_EPS))

    samples = hex_outline_samples(config)
    contour_values = _bilinear_sample(norm, grid, samples["x_m"], samples["y_m"])
    side_means = np.asarray(
        [np.mean(contour_values[samples["side"] == idx]) for idx in range(6)],
        dtype=float,
    )
    contour_mean = float(np.mean(contour_values)) + BT_EPS
    edge_uniformity = float(1.0 / (1.0 + np.std(contour_values) / contour_mean))
    side_balance = float(np.min(side_means) / (np.max(side_means) + BT_EPS))

    outline_energy = float(np.sum(intensity[outline_mask]))
    roi_energy = float(np.sum(intensity[roi_mask])) + BT_EPS
    core_peak = float(np.max(norm[core_mask])) if np.any(core_mask) else np.nan
    outside_peak = float(np.max(norm[outside_mask])) if np.any(outside_mask) else np.nan
    return {
        "component_count": int(count),
        "largest_component_fraction": largest_fraction,
        "single_outline_component_pass": bool(count == 1 and largest_fraction >= 0.95),
        "outline_precision": precision,
        "outline_recall": recall,
        "outline_f1": f1,
        "outline_f1_pass": bool(f1 >= 0.50),
        "core_peak_ratio": core_peak,
        "dark_core_pass": bool(np.isfinite(core_peak) and core_peak <= 0.08),
        "side_lobe_peak_ratio": outside_peak,
        "side_lobe_pass": bool(np.isfinite(outside_peak) and outside_peak <= 0.25),
        "outline_energy_fraction": float(outline_energy / roi_energy),
        "outline_energy_pass": bool(outline_energy / roi_energy >= 0.55),
        "edge_uniformity": edge_uniformity,
        "edge_uniformity_pass": bool(edge_uniformity >= 0.55),
        "side_balance": side_balance,
        "side_balance_pass": bool(side_balance >= 0.35),
        "target_flat_radius_um": float(config.flat_radius_m / BT_UM),
        "target_line_sigma_um": float(config.line_sigma_m / BT_UM),
        "target_line_fwhm_um": float(2.355 * config.line_sigma_m / BT_UM),
        "threshold_fraction": float(config.threshold_fraction),
    }


def _pad_rect_to_square(U_rect: np.ndarray, N: int) -> np.ndarray:
    out = np.zeros((int(N), int(N)), dtype=U_rect.dtype)
    ny, nx = U_rect.shape
    y0 = (int(N) - ny) // 2
    x0 = (int(N) - nx) // 2
    if y0 < 0 or x0 < 0:
        raise ValueError("square grid is too small for the rectangular SLM field")
    out[y0 : y0 + ny, x0 : x0 + nx] = U_rect
    return out


def lab_pupil_amplitude(twin_config: TwinConfig) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    """Return square pupil grid, lab amplitude mask, and rectangular metadata."""

    ds = max(1, int(twin_config.grid.device_downsample))
    nx = int(np.ceil(twin_config.slm.resolution_x / ds))
    ny = int(np.ceil(twin_config.slm.resolution_y / ds))
    rect_grid = make_rect_grid(nx, ny, twin_config.slm.pixel_pitch_m * ds)
    amp = gaussian_amplitude(rect_grid["R"], twin_config.laser.beam_radius_on_slm_m)
    if twin_config.include_fill_factor:
        amp = amp * _fill_factor_amplitude_grid(rect_grid, twin_config.slm.pixel_pitch_m, twin_config.slm.fill_factor)
    if twin_config.include_active_aperture:
        aperture = (
            (np.abs(rect_grid["X"]) <= 0.5 * twin_config.slm.active_width_m)
            & (np.abs(rect_grid["Y"]) <= 0.5 * twin_config.slm.active_height_m)
        ).astype(float)
        amp = amp * aperture
    pupil_grid = make_xy_grid(twin_config.grid.N, rect_grid["dx"])
    amp_sq = _pad_rect_to_square(amp, int(twin_config.grid.N))
    pupil = (pupil_grid["R"] <= twin_config.objective.pupil_radius_m).astype(float)
    return pupil_grid, amp_sq * pupil, {"rect_grid": rect_grid, "device_downsample": ds}


def focus_grid_from_pupil(twin_config: TwinConfig, pupil_grid: Mapping[str, Any]) -> dict[str, Any]:
    dummy = np.zeros((int(pupil_grid["N"]), int(pupil_grid["N"])), dtype=complex)
    _field, focal_grid = focus_to_focal_plane(dummy, dict(pupil_grid), twin_config.laser, twin_config.objective)
    return focal_grid


def phase_retrieve_outline(
    pupil_amplitude: np.ndarray,
    config: HexOutlineConfig,
    *,
    iterations: int | None = None,
    target_grid: Mapping[str, Any],
    initial_pupil_phase: np.ndarray | None = None,
    initial_focus_field: np.ndarray | None = None,
) -> dict[str, Any]:
    """Phase-only Gerchberg-Saxton/MRAF retrieval for the outline target."""

    rng = np.random.default_rng(int(config.random_seed))
    amp = np.asarray(pupil_amplitude, dtype=float)
    support = amp > 0.0
    target = hex_outline_target(target_grid, config)
    target_amp = np.asarray(target["target_amplitude"], dtype=float)
    roi = np.asarray(target["roi_mask"], dtype=bool)
    signal = np.asarray(target["outline_mask"], dtype=bool)
    if initial_pupil_phase is not None:
        seed_phase = np.asarray(initial_pupil_phase, dtype=float)
        if seed_phase.shape != amp.shape:
            raise ValueError("initial_pupil_phase must match the pupil amplitude shape.")
        phase = seed_phase
    elif initial_focus_field is not None:
        seed_focus = np.asarray(initial_focus_field, dtype=complex)
        if seed_focus.shape != amp.shape:
            raise ValueError("initial_focus_field must match the pupil amplitude shape.")
        phase = np.angle(ifft2c(seed_focus))
    else:
        phase = rng.uniform(0.0, BT_TWOPI, size=amp.shape)
    U = amp * np.exp(1j * phase)
    history: list[dict[str, float]] = []
    n_iter = int(config.retrieval_iterations if iterations is None else iterations)
    for idx in range(max(1, n_iter)):
        F = fft2c(U)
        current_amp = np.abs(F)
        current_phase = np.angle(F)
        signal_scale = float(np.mean(current_amp[signal]) / (np.mean(target_amp[signal]) + BT_EPS)) if np.any(signal) else 1.0
        desired_amp = signal_scale * target_amp
        next_amp = current_amp.copy()
        next_amp[roi] = desired_amp[roi]
        F_next = next_amp * np.exp(1j * current_phase)
        U_back = ifft2c(F_next)
        U = amp * np.exp(1j * np.angle(U_back))
        U = np.where(support, U, 0.0)
        if idx == 0 or (idx + 1) % 25 == 0 or idx + 1 == n_iter:
            I = np.abs(fft2c(U)) ** 2
            m = hex_outline_metrics(I, target_grid, config)
            history.append(
                {
                    "iteration": float(idx + 1),
                    "outline_f1": float(m["outline_f1"]),
                    "core_peak_ratio": float(m["core_peak_ratio"]),
                    "side_lobe_peak_ratio": float(m["side_lobe_peak_ratio"]),
                    "outline_energy_fraction": float(m["outline_energy_fraction"]),
                }
            )
    F_final = fft2c(U)
    metrics = hex_outline_metrics(np.abs(F_final) ** 2, target_grid, config)
    return {
        "pupil_field": U,
        "pupil_phase": np.angle(U),
        "focus_field_fft": F_final,
        "target": target,
        "metrics": metrics,
        "history": history,
    }


def _asm_transfer_stack(
    grid: Mapping[str, Any],
    wavelength_m: float,
    z_values_m: Sequence[float],
    *,
    n_medium: float = 1.0,
    bandlimit: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return forward and backward ASM transfer functions for one grid."""

    k = BT_TWOPI * float(n_medium) / float(wavelength_m)
    kx = BT_TWOPI * np.asarray(grid["FX"], dtype=float)
    ky = BT_TWOPI * np.asarray(grid["FY"], dtype=float)
    arg = k * k - kx * kx - ky * ky
    propagating = arg >= 0.0
    kz = np.zeros_like(arg, dtype=float)
    kz[propagating] = np.sqrt(np.maximum(arg[propagating], 0.0))
    evanescent_alpha = np.sqrt(np.maximum(-arg, 0.0))
    forward = []
    backward = []
    for z in np.asarray(z_values_m, dtype=float):
        H = np.zeros_like(arg, dtype=complex)
        H_back = np.zeros_like(arg, dtype=complex)
        H[propagating] = np.exp(1j * kz[propagating] * float(z))
        H_back[propagating] = np.exp(-1j * kz[propagating] * float(z))
        H[~propagating] = np.exp(-evanescent_alpha[~propagating] * abs(float(z)))
        H_back[~propagating] = H[~propagating]
        if bandlimit:
            mask = bandlimit_mask_matsushima(dict(grid), wavelength_m, float(z), n_medium=n_medium)
            H = H * mask
            H_back = H_back * mask
        forward.append(H)
        backward.append(H_back)
    return np.asarray(forward), np.asarray(backward)


def _propagate_with_transfer(field: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    return ifft2c(fft2c(field) * transfer)


def phase_retrieve_outline_multiplane(
    pupil_amplitude: np.ndarray,
    config: HexOutlineConfig,
    *,
    target_grid: Mapping[str, Any],
    z_values_m: Sequence[float],
    wavelength_m: float,
    n_medium: float,
    iterations: int | None = None,
    initial_focus_field: np.ndarray | None = None,
) -> dict[str, Any]:
    """Phase-only retrieval with the same hollow outline enforced over z.

    This is the Bessel-like gate: the field is constrained to the same hollow
    hexagon outline at each requested axial plane, then back-projected to a
    single phase-only pupil.
    """

    rng = np.random.default_rng(int(config.random_seed))
    amp = np.asarray(pupil_amplitude, dtype=float)
    support = amp > 0.0
    target = hex_outline_target(target_grid, config)
    target_amp = np.asarray(target["target_amplitude"], dtype=float)
    roi = np.asarray(target["roi_mask"], dtype=bool)
    signal = np.asarray(target["outline_mask"], dtype=bool)
    if initial_focus_field is not None:
        seed_focus = np.asarray(initial_focus_field, dtype=complex)
        if seed_focus.shape != amp.shape:
            raise ValueError("initial_focus_field must match the pupil amplitude shape.")
        phase = np.angle(ifft2c(seed_focus))
    else:
        phase = rng.uniform(0.0, BT_TWOPI, size=amp.shape)
    U = amp * np.exp(1j * phase)
    H_forward, H_backward = _asm_transfer_stack(
        target_grid,
        wavelength_m,
        z_values_m,
        n_medium=n_medium,
        bandlimit=True,
    )
    history: list[dict[str, float]] = []
    n_iter = int(config.retrieval_iterations if iterations is None else iterations)
    for idx in range(max(1, n_iter)):
        focus0 = fft2c(U)
        back_focus_fields = []
        for Hf, Hb in zip(H_forward, H_backward):
            plane = _propagate_with_transfer(focus0, Hf)
            current_amp = np.abs(plane)
            current_phase = np.angle(plane)
            scale = float(np.mean(current_amp[signal]) / (np.mean(target_amp[signal]) + BT_EPS)) if np.any(signal) else 1.0
            next_amp = current_amp.copy()
            next_amp[roi] = scale * target_amp[roi]
            constrained = next_amp * np.exp(1j * current_phase)
            back_focus_fields.append(_propagate_with_transfer(constrained, Hb))
        focus_next = np.mean(np.asarray(back_focus_fields), axis=0)
        U_back = ifft2c(focus_next)
        U = amp * np.exp(1j * np.angle(U_back))
        U = np.where(support, U, 0.0)
        if idx == 0 or (idx + 1) % 20 == 0 or idx + 1 == n_iter:
            stack = []
            focus_eval = fft2c(U)
            for Hf in H_forward:
                stack.append(np.abs(_propagate_with_transfer(focus_eval, Hf)) ** 2)
            summary = outline_z_survival(np.asarray(stack), target_grid, config, z_values_m)
            history.append(
                {
                    "iteration": float(idx + 1),
                    "accepted_depth_um": float(summary["accepted_depth_um"]),
                    "mean_outline_f1": float(summary["mean_outline_f1"]),
                    "min_outline_f1": float(summary["min_outline_f1"]),
                    "max_side_lobe_peak_ratio": float(summary["max_side_lobe_peak_ratio"]),
                }
            )
    focus_final = fft2c(U)
    fields = np.asarray([_propagate_with_transfer(focus_final, Hf) for Hf in H_forward])
    intensity_stack = np.abs(fields) ** 2
    metrics_z = outline_z_survival(intensity_stack, target_grid, config, z_values_m)
    return {
        "pupil_field": U,
        "pupil_phase": np.angle(U),
        "focus_field_fft": focus_final,
        "fields_z": fields,
        "intensity_stack": intensity_stack,
        "z_values_m": np.asarray(z_values_m, dtype=float),
        "target": target,
        "metrics_z": metrics_z,
        "history": history,
    }


def build_lab_outline_case(
    twin_config: TwinConfig,
    config: HexOutlineConfig,
    pupil_phase: np.ndarray,
    *,
    correct_interface: bool = True,
    quantize: bool = True,
    include_interface: bool = True,
) -> dict[str, Any]:
    """Build the lab-realistic focal field from a retrieved pupil phase."""

    pupil_grid, amp, meta = lab_pupil_amplitude(twin_config)
    phase = np.asarray(pupil_phase, dtype=float)
    if correct_interface and include_interface:
        phase = phase + interface_correction_phase(
            pupil_grid,
            twin_config.laser,
            twin_config.objective,
            twin_config.material,
        )
    encoded_phase = quantize_phase(phase, int(config.phase_bits)) if quantize else phase_wrap(phase)
    U = amp * np.exp(1j * encoded_phase)
    if include_interface:
        W = interface_aberration_pupil(pupil_grid, twin_config.laser, twin_config.objective, twin_config.material)
        U = U * np.exp(1j * W)
    focus, focal_grid = focus_to_focal_plane(U, pupil_grid, twin_config.laser, twin_config.objective)
    metrics = hex_outline_metrics(focus, focal_grid, config)
    gray = phase_to_gray(encoded_phase, int(config.phase_bits), invert=twin_config.slm.invert_gray)
    rect_grid = meta["rect_grid"]
    ny = int(rect_grid["ny"])
    nx = int(rect_grid["nx"])
    y0 = (int(pupil_grid["N"]) - ny) // 2
    x0 = (int(pupil_grid["N"]) - nx) // 2
    return {
        "path": "lab_realistic",
        "config": config,
        "grid": focal_grid,
        "field": focus,
        "pupil_grid": pupil_grid,
        "pupil_amplitude": amp,
        "encoded_phase": encoded_phase,
        "gray": gray,
        "gray_rect": gray[y0 : y0 + ny, x0 : x0 + nx],
        "metrics": metrics,
        **meta,
    }


def propagate_outline_case_z(
    field: np.ndarray,
    grid: Mapping[str, Any],
    config: HexOutlineConfig,
    z_values_m: Sequence[float],
    *,
    wavelength_m: float,
    n_medium: float,
) -> dict[str, Any]:
    """Propagate an outline field over z and score each plane."""

    H_forward, _ = _asm_transfer_stack(grid, wavelength_m, z_values_m, n_medium=n_medium, bandlimit=True)
    fields = np.asarray([_propagate_with_transfer(np.asarray(field, dtype=complex), Hf) for Hf in H_forward])
    intensity_stack = np.abs(fields) ** 2
    survival = outline_z_survival(intensity_stack, grid, config, z_values_m)
    return {
        "fields_z": fields,
        "intensity_stack": intensity_stack,
        "z_values_m": np.asarray(z_values_m, dtype=float),
        "metrics_z": survival,
    }


def outline_z_survival(
    intensity_stack: np.ndarray,
    grid: Mapping[str, Any],
    config: HexOutlineConfig,
    z_values_m: Sequence[float],
    *,
    f1_threshold: float = 0.65,
    core_threshold: float = 0.08,
    side_lobe_threshold: float = 0.25,
) -> dict[str, Any]:
    """Return per-plane and contiguous-depth hollow-outline survival metrics."""

    rows = []
    for idx, z_m in enumerate(np.asarray(z_values_m, dtype=float)):
        metrics = hex_outline_metrics(np.asarray(intensity_stack[idx], dtype=float), grid, config)
        accepted = (
            float(metrics["outline_f1"]) >= float(f1_threshold)
            and float(metrics["core_peak_ratio"]) <= float(core_threshold)
            and float(metrics["side_lobe_peak_ratio"]) <= float(side_lobe_threshold)
            and bool(metrics["single_outline_component_pass"])
        )
        rows.append(
            {
                "z_um": float(z_m / BT_UM),
                "accepted": bool(accepted),
                "outline_f1": float(metrics["outline_f1"]),
                "core_peak_ratio": float(metrics["core_peak_ratio"]),
                "side_lobe_peak_ratio": float(metrics["side_lobe_peak_ratio"]),
                "outline_energy_fraction": float(metrics["outline_energy_fraction"]),
                "edge_uniformity": float(metrics["edge_uniformity"]),
                "side_balance": float(metrics["side_balance"]),
                "component_count": int(metrics["component_count"]),
            }
        )
    accepted = np.asarray([row["accepted"] for row in rows], dtype=bool)
    z = np.asarray(z_values_m, dtype=float)
    best_depth = 0.0
    start_um = end_um = np.nan
    if accepted.size and np.any(accepted):
        padded = np.r_[False, accepted, False]
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        starts = changes[0::2]
        ends = changes[1::2] - 1
        spans = z[ends] - z[starts]
        best_idx = int(np.argmax(spans))
        best_depth = float(spans[best_idx] / BT_UM)
        start_um = float(z[starts[best_idx]] / BT_UM)
        end_um = float(z[ends[best_idx]] / BT_UM)
    f1 = np.asarray([row["outline_f1"] for row in rows], dtype=float)
    side = np.asarray([row["side_lobe_peak_ratio"] for row in rows], dtype=float)
    core = np.asarray([row["core_peak_ratio"] for row in rows], dtype=float)
    return {
        "rows": rows,
        "accepted_plane_count": int(np.count_nonzero(accepted)),
        "accepted_any": bool(np.any(accepted)),
        "accepted_depth_um": best_depth,
        "accepted_z_start_um": start_um,
        "accepted_z_end_um": end_um,
        "mean_outline_f1": float(np.mean(f1)) if f1.size else np.nan,
        "min_outline_f1": float(np.min(f1)) if f1.size else np.nan,
        "max_side_lobe_peak_ratio": float(np.max(side)) if side.size else np.nan,
        "max_core_peak_ratio": float(np.max(core)) if core.size else np.nan,
        "f1_threshold": float(f1_threshold),
        "core_threshold": float(core_threshold),
        "side_lobe_threshold": float(side_lobe_threshold),
    }


def export_outline_hologram(case: Mapping[str, Any], out_dir: str | Path, *, label: str = "hollow_hex_outline") -> dict[str, Path]:
    if Image is None:
        raise RuntimeError("Pillow is required for PNG export.")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gray = np.asarray(case.get("gray_rect", case["gray"]), dtype=np.uint8)
    png_path = out / f"{label}_phase.png"
    Image.fromarray(gray).save(png_path)
    json_path = out / f"{label}_params.json"
    payload = {
        "label": label,
        "target": "hollow_regular_hexagon_outline",
        "hex_outline_config": asdict(case.get("config", HexOutlineConfig())),
        "metrics": dict(case.get("metrics", {})),
    }
    import json

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"phase_png": png_path, "params_json": json_path}


def plot_outline_checkpoint(
    cases: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    title: str = "Hollow hexagon outline checkpoint",
) -> Path:
    """Save target/phase-only/lab comparison panels."""

    import matplotlib.pyplot as plt

    vbb_style.apply_style()
    fig, axes = plt.subplots(len(cases), 3, figsize=(10.8, 3.4 * len(cases)), constrained_layout=True)
    if len(cases) == 1:
        axes = axes[None, :]
    for row, case in enumerate(cases):
        config = case["config"]
        target_grid = case["target_grid"]
        target = hex_outline_target(target_grid, config)
        panels = [
            ("target outline", target["target_amplitude"] ** 2, target_grid, None),
            ("phase-only ideal", np.abs(case["ideal"]["focus_field_fft"]) ** 2, target_grid, case["ideal"]["metrics"]),
            ("lab corrected", np.abs(case["lab"]["field"]) ** 2, case["lab"]["grid"], case["lab"]["metrics"]),
        ]
        for col, (label, intensity, grid, metrics) in enumerate(panels):
            ax = axes[row, col]
            x_um = np.asarray(grid["x"], dtype=float) / BT_UM
            ax.imshow(
                vbb_style.display_scale(intensity, gamma=0.55),
                origin="lower",
                extent=[float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])],
                cmap=vbb_style.INTENSITY_CMAP,
                vmin=0.0,
                vmax=1.0,
            )
            verts = hex_vertices(config) / BT_UM
            closed = np.vstack([verts, verts[0]])
            ax.plot(closed[:, 0], closed[:, 1], color="white", lw=0.9, alpha=0.9)
            ax.set_xlim(-16.0, 16.0)
            ax.set_ylim(-16.0, 16.0)
            ax.set_aspect("equal")
            ax.set_xlabel("x [um]")
            ax.set_ylabel("y [um]")
            if metrics is None:
                subtitle = f"line FWHM={2.355 * config.line_sigma_m / BT_UM:.2f} um"
            else:
                subtitle = (
                    f"F1={metrics['outline_f1']:.2f}, "
                    f"core={metrics['core_peak_ratio']:.2f}, "
                    f"side={metrics['side_lobe_peak_ratio']:.2f}"
                )
            ax.set_title(f"{label}\n{subtitle}", fontsize=9)
    fig.suptitle(title, fontsize=14)
    caption = (
        "Direct hollow-hexagon outline target. The white line is the requested regular-hexagon perimeter; "
        "metrics score thresholded intensity against that outline and separately penalise core leakage and side lobes."
    )
    out = vbb_style.save_figure(
        fig,
        output_path,
        caption,
        metadata={"figure": "hollow_hex_outline_checkpoint"},
    )
    plt.close(fig)
    return out


__all__ = [
    "HexOutlineConfig",
    "build_lab_outline_case",
    "distance_to_hex_outline",
    "export_outline_hologram",
    "focus_grid_from_pupil",
    "hex_outline_metrics",
    "hex_outline_samples",
    "hex_outline_target",
    "hex_vertices",
    "inside_hex_mask",
    "lab_pupil_amplitude",
    "phase_retrieve_outline",
    "phase_retrieve_outline_multiplane",
    "plot_outline_checkpoint",
    "propagate_outline_case_z",
    "outline_z_survival",
]
