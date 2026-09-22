"""Source-scale two-SLM oscillating-polarization Bessel model (Fu et al. 2016).

This is a nominal numerical adaptation to the recorded 1029 nm lab source and
PLUTO-size panels. The sequential SLMs are assumed to be in conjugate planes;
their actual spacing, reflection parity and phase LUT need bench calibration.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vbb_study.digital_twin.nathan_mode2u3_hardware_closure import resolve_slm_hardware
from vbb_study.digital_twin.nathan_mode2v_lab_ready_build import LAB_WAVELENGTH_M, native_panel_grid
from vbb_study.equations.vector_jones import linear_analyzer_intensity


def wrap(phase: np.ndarray) -> np.ndarray:
    return np.mod(phase, 2 * np.pi)


def paper_masks(radius: np.ndarray, theta: np.ndarray, ell: int, d_m: float, D_m: float) -> tuple[np.ndarray, np.ndarray]:
    """HA1 + ell SPP and HA2 + 2ell SPP in a common design frame."""

    return wrap(-2 * np.pi * radius / d_m + ell * theta), wrap(-2 * np.pi * radius / D_m + 2 * ell * theta)


def source_fields(x: np.ndarray, y: np.ndarray, *, ell: int, beam_radius_m: float,
                  d_m: float, D_m: float, ha2: bool) -> tuple[np.ndarray, np.ndarray]:
    """Effective field after two reflective SLMs, HWP 22.5°, and QWP 45°.

    The reflection parity is written explicitly in a single receiver frame:
    after SLM2, the unmodulated V charge is +ell and the H charge is -ell.
    The quarter-cycle relative phase fixes the nominal QWP convention.
    """

    X, Y = np.meshgrid(x, y, indexing="xy")
    radius = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    gaussian = np.exp(-(radius / beam_radius_m) ** 2) / np.sqrt(2)
    common = -2 * np.pi * radius / d_m
    extra = -2 * np.pi * radius / D_m if ha2 else 0.0
    h = gaussian * np.exp(1j * (common + extra - ell * theta))
    v = gaussian * np.exp(1j * (common + ell * theta + np.pi / 2))
    # Nominal QWP fast-axis convention; global phase has no camera effect.
    ex = (h - 1j * v) / np.sqrt(2)
    ey = (-1j * h + v) / np.sqrt(2)
    return ex, ey


def harmonic_orientation(intensity: np.ndarray, theta: np.ndarray, ring_mask: np.ndarray, ell: int) -> tuple[float, float]:
    signal = np.sum(intensity[ring_mask] * np.exp(-2j * ell * theta[ring_mask]))
    dc = np.sum(intensity[ring_mask])
    return float(np.angle(signal)), float(2 * abs(signal) / max(float(dc), 1e-30))


def run(output_dir: Path, *, ell: int = 3, grid_n: int = 1024,
        window_m: float = 10e-3, beam_radius_m: float = 2e-3,
        d_m: float = 160e-6, D_m: float = 1200e-6) -> dict:
    hardware, _ = resolve_slm_hardware()
    pitch = float(hardware.pixel_pitch_m)
    if min(d_m, D_m) / pitch < 16:
        raise ValueError("holographic-axicon period must span at least 16 physical SLM pixels")
    if ell < 1 or grid_n < 256:
        raise ValueError("ell must be positive and grid_n at least 256")
    x = (np.arange(grid_n) - grid_n // 2) * window_m / grid_n
    X, Y = np.meshgrid(x, x, indexing="xy")
    radius = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    frequency = np.fft.fftfreq(grid_n, d=window_m / grid_n)
    FX, FY = np.meshgrid(frequency, frequency, indexing="xy")
    fsq = FX * FX + FY * FY
    z_m = np.linspace(0.04, 0.22, 10)
    wavelength = LAB_WAVELENGTH_M
    kr1, kr2 = 2 * np.pi / d_m, 2 * np.pi / D_m
    k = 2 * np.pi / wavelength
    kz_v = np.sqrt(k * k - kr1 * kr1)
    kz_h = np.sqrt(k * k - (kr1 + kr2) ** 2)
    predicted_period_m = 2 * np.pi / abs(kz_h - kz_v)

    source = {enabled: source_fields(x, x, ell=ell, beam_radius_m=beam_radius_m,
                                     d_m=d_m, D_m=D_m, ha2=enabled)
              for enabled in (False, True)}
    spectra = {enabled: tuple(np.fft.fft2(component) for component in fields)
               for enabled, fields in source.items()}
    kz = np.sqrt((k * k - (2 * np.pi) ** 2 * fsq) + 0j)
    # The ring is set by the first bright Bessel annulus, rather than by the
    # analyzer petals, so all rows use the same physically identified ROI.
    ring_centre_m = (ell + 1.0) / kr1
    ring_mask = (radius > 0.65 * ring_centre_m) & (radius < 1.45 * ring_centre_m)
    rows = []
    images: dict[bool, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {False: [], True: []}
    for enabled in (False, True):
        for z in z_m:
            transfer = np.exp(1j * kz * z)
            ex = np.fft.ifft2(spectra[enabled][0] * transfer)
            ey = np.fft.ifft2(spectra[enabled][1] * transfer)
            total = np.abs(ex) ** 2 + np.abs(ey) ** 2
            i0 = linear_analyzer_intensity(ex, ey, 0.0)
            i90 = linear_analyzer_intensity(ex, ey, np.pi / 2)
            phase, contrast = harmonic_orientation(i0, theta, ring_mask, ell)
            images[enabled].append((total, i0, i90))
            rows.append({"ha2_enabled": enabled, "z_m": float(z),
                         "harmonic_phase_rad": phase, "orientation_modulo_pi_over_ell_rad": -phase / (2 * ell),
                         "2ell_modulation_contrast": contrast,
                         "total_power": float(np.sum(total) * (window_m / grid_n) ** 2)})

    for enabled in (False, True):
        subset = [r for r in rows if r["ha2_enabled"] == enabled]
        unwrapped = np.unwrap([r["harmonic_phase_rad"] for r in subset])
        for row, phase in zip(subset, unwrapped):
            row["orientation_unwrapped_rad"] = float(-phase / (2 * ell))
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "z_rotation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    extent = [float(x[0] * 1e3), float(x[-1] * 1e3)] * 2
    selected = (0, 3, 6, 9)
    fig, axes = plt.subplots(len(selected), 3, figsize=(10.2, 12.0), constrained_layout=True)
    peak = max(float(np.max(images[True][i][0])) for i in selected)
    for row, index in enumerate(selected):
        for col, image in enumerate(images[True][index]):
            ax = axes[row, col]
            ax.imshow(np.clip(image / max(peak, 1e-30), 0, 1) ** 0.45,
                      origin="lower", extent=extent, cmap="inferno", vmin=0, vmax=1)
            ax.set_xlim(-0.30, 0.30)
            ax.set_ylim(-0.30, 0.30)
            ax.set_title(("total", "polarizer 0°", "polarizer 90°")[col])
            ax.set_xlabel("x [mm]")
            if col == 0:
                ax.set_ylabel(f"z={z_m[index]*1e3:.0f} mm\ny [mm]")
    montage = output_dir / f"ell{ell}_paper_route_z_montage.png"
    fig.savefig(montage, dpi=170)
    plt.close(fig)

    comparison_index = 3  # fixed 100 mm plane from the declared 40-220 mm scan
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.8), constrained_layout=True)
    comparison_peak = max(float(np.max(images[enabled][comparison_index][0])) for enabled in (False, True))
    for row, enabled in enumerate((False, True)):
        for col, image in enumerate(images[enabled][comparison_index]):
            ax = axes[row, col]
            ax.imshow(np.clip(image / max(comparison_peak, 1e-30), 0, 1) ** 0.45,
                      origin="lower", extent=extent, cmap="inferno", vmin=0, vmax=1)
            ax.set_xlim(-0.30, 0.30)
            ax.set_ylim(-0.30, 0.30)
            ax.set_title(("total", "polarizer 0°", "polarizer 90°")[col])
            ax.set_xlabel("x [mm]")
            if col == 0:
                ax.set_ylabel(f"HA2 {'on' if enabled else 'off'}\ny [mm]")
    comparison_figure = output_dir / f"ell{ell}_static_vs_oscillating_z100mm.png"
    fig.savefig(comparison_figure, dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for enabled, label in ((False, "HA2 off: static control"), (True, "HA2 on: oscillating polarization")):
        subset = [r for r in rows if r["ha2_enabled"] == enabled]
        ax.plot(z_m * 1e3, np.rad2deg([r["orientation_unwrapped_rad"] for r in subset]),
                marker="o", label=label)
    ax.set(xlabel="camera z after SLM2 [mm]", ylabel="0° analyzer lobe orientation [deg]",
           title=f"Polarization-order {ell}: expected axial rotation")
    ax.grid(alpha=0.3)
    ax.legend()
    rotation_figure = output_dir / f"ell{ell}_rotation_vs_z.png"
    fig.savefig(rotation_figure, dpi=170)
    plt.close(fig)

    native = native_panel_grid()
    mask1, mask2 = paper_masks(np.asarray(native["R"]), np.asarray(native["PHI"]), ell, d_m, D_m)
    mask_dir = output_dir / "native_phase_design"
    mask_dir.mkdir(exist_ok=True)
    slm1_path = mask_dir / f"SLM1_HA1_SPP_l{ell}_radians.npy"
    slm2_path = mask_dir / f"SLM2_HA2_SPP_2l{ell}_radians.npy"
    np.save(slm1_path, mask1.astype(np.float32))
    np.save(slm2_path, mask2.astype(np.float32))

    on = [r for r in rows if r["ha2_enabled"]]
    off = [r for r in rows if not r["ha2_enabled"]]
    power = np.asarray([r["total_power"] for r in rows], dtype=float)
    power_drift = float((np.max(power) - np.min(power)) / np.mean(power))
    report = {
        "status": "numerical_design_prediction_not_measured",
        "paper": "Fu, Zhang, Gao, Scientific Reports 6, 30765 (2016), doi:10.1038/srep30765",
        "architecture": "sequential SLM1(HA1+ell SPP) -> HWP 22.5deg -> SLM2(H-only HA2+2ell SPP) -> QWP 45deg -> polarizer -> camera",
        "source_scale_adaptation": {"wavelength_m": wavelength, "beam_radius_m": beam_radius_m,
                                    "grid_n": grid_n, "window_m": window_m, "ell": ell,
                                    "HA1_period_m": d_m, "HA2_period_m": D_m,
                                    "periods_in_slm_pixels": [d_m / pitch, D_m / pitch],
                                    "slm_model": hardware.model, "slm_native_pixels": [hardware.width_px, hardware.height_px]},
        "predicted_axial_polarization_period_m": float(predicted_period_m),
        "propagation_power_drift_fraction": power_drift,
        "simulated_rotation_deg_HA2_on": float(np.rad2deg(on[-1]["orientation_unwrapped_rad"] - on[0]["orientation_unwrapped_rad"])),
        "simulated_rotation_deg_HA2_off": float(np.rad2deg(off[-1]["orientation_unwrapped_rad"] - off[0]["orientation_unwrapped_rad"])),
        "assumptions": ["ideal conjugate relay between reflective SLMs", "equal H/V amplitudes after ideal HWP",
                        "nominal reflection parity in common receiver frame", "ideal QWP and analyzer",
                        "scalar angular-spectrum propagation of transverse Jones components"],
        "calibration_required": ["actual SLM1-to-SLM2 relay/spacing", "per-panel 1029 nm phase LUT and stroke",
                                 "reflection parity and LC director orientation", "waveplate angles and retardance",
                                 "camera scale and z origin"],
        "mask_status": "native phase designs only; not drive-ready",
        "figures": [str(montage), str(comparison_figure), str(rotation_figure)],
        "metrics_csv": str(csv_path),
        "native_phase_designs": [str(slm1_path), str(slm2_path)],
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
