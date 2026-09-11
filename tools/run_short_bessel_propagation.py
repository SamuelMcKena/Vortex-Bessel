from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

from vbb_study.short_bessel import ShortBesselDesignInput, design_short_bessel
from vbb_study.short_bessel_propagation import (
    ShortBesselPropagationConfig,
    run_short_bessel_propagation,
)


UM = 1.0e-6


def _normalised_sqrt(values: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), 0.0)
    vmax = float(np.max(arr))
    if vmax <= 0.0:
        return np.zeros_like(arr)
    return np.sqrt(arr / vmax)


def _crop_square(x_um: np.ndarray, image: np.ndarray, half_width_um: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_um, dtype=float)
    mask = np.abs(x) <= float(half_width_um)
    return x[mask], np.asarray(image)[np.ix_(mask, mask)]


def _metrics_text(comparison) -> str:
    d = comparison.design
    t = comparison.target.metrics
    c = comparison.current.metrics
    return "\n".join(
        [
            "Equivalent sample-plane propagation",
            f"target core: {d.target_fwhm_um:.2f} um FWHM",
            f"target geometric length: {d.target_length_um:.0f} um",
            f"k_r: {d.target_kr_sample_m_inv:,.0f} m^-1",
            "",
            f"required radius: {t.sample_beam_radius_um:.2f} um",
            f"current radius:  {c.sample_beam_radius_um:.2f} um",
            "",
            f"TARGET peak z: {t.peak_z_um:.1f} um",
            f"TARGET FWHM @ peak: {t.transverse_fwhm_at_peak_um:.3f} um",
            f"TARGET 50% on-axis zone: {t.on_axis_halfmax_zone_length_um:.1f} um",
            f"TARGET geometric ref: {t.geometric_reference_length_um:.1f} um",
            "",
            f"CURRENT peak z: {c.peak_z_um:.1f} um",
            f"CURRENT FWHM @ peak: {c.transverse_fwhm_at_peak_um:.3f} um",
            f"CURRENT 50% on-axis zone: {c.on_axis_halfmax_zone_length_um:.1f} um",
            f"CURRENT geometric ref: {c.geometric_reference_length_um:.1f} um",
            "",
            f"radial samples / 2pi period: {t.samples_per_radial_period:.1f}",
            f"target power ratio range: {t.min_power_ratio:.4f} .. {t.max_power_ratio:.4f}",
            f"current power ratio range: {c.min_power_ratio:.4f} .. {c.max_power_ratio:.4f}",
            "",
            "Claim boundary:",
            "equivalent homogeneous-medium scalar propagation only;",
            "the real SLM + 4F + physical axicon + objective train",
            "is not yet explicitly propagated/calibrated here.",
        ]
    )


def _comparison_figure(comparison, path: Path, x_half_um: float = 28.0) -> None:
    target = comparison.target
    current = comparison.current
    xmask_t = np.abs(target.x_um) <= float(x_half_um)
    xmask_c = np.abs(current.x_um) <= float(x_half_um)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "5 um short-Bessel scalar propagation check\n"
        "required beam radius versus current 2 mm upstream radius",
        fontsize=15,
    )

    ax = axes[0, 0]
    image = _normalised_sqrt(target.xz_intensity[:, xmask_t])
    im = ax.imshow(
        image,
        extent=(target.x_um[xmask_t][0], target.x_um[xmask_t][-1], target.z_um[0], target.z_um[-1]),
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    m = target.metrics
    ax.axhline(m.on_axis_halfmax_zone_start_um, linestyle="--", linewidth=1.0)
    ax.axhline(m.on_axis_halfmax_zone_end_um, linestyle="--", linewidth=1.0)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z in sample (um)")
    ax.set_title(
        f"Required radius {m.sample_beam_radius_um:.2f} um | "
        f"FWHM@peak {m.transverse_fwhm_at_peak_um:.2f} um"
    )
    fig.colorbar(im, ax=ax, label="sqrt(normalised intensity)")

    ax = axes[0, 1]
    image = _normalised_sqrt(current.xz_intensity[:, xmask_c])
    im = ax.imshow(
        image,
        extent=(current.x_um[xmask_c][0], current.x_um[xmask_c][-1], current.z_um[0], current.z_um[-1]),
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    m = current.metrics
    ax.axhline(m.on_axis_halfmax_zone_start_um, linestyle="--", linewidth=1.0)
    ax.axhline(m.on_axis_halfmax_zone_end_um, linestyle="--", linewidth=1.0)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z in sample (um)")
    ax.set_title(
        f"Current-equivalent radius {m.sample_beam_radius_um:.2f} um | "
        f"FWHM@peak {m.transverse_fwhm_at_peak_um:.2f} um"
    )
    fig.colorbar(im, ax=ax, label="sqrt(normalised intensity)")

    ax = axes[1, 0]
    tn = target.on_axis_intensity / max(float(np.max(target.on_axis_intensity)), 1.0e-30)
    cn = current.on_axis_intensity / max(float(np.max(current.on_axis_intensity)), 1.0e-30)
    ax.plot(target.z_um, tn, label="required radius")
    ax.plot(current.z_um, cn, label="current radius")
    ax.axhline(0.5, linestyle="--", linewidth=1.0, label="50% of each peak")
    ax.set_xlabel("z in sample (um)")
    ax.set_ylabel("normalised on-axis intensity")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Axial central-intensity envelope")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.0, 1.0, _metrics_text(comparison), va="top", ha="left", family="monospace", fontsize=9.5)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _snapshot_figure(comparison, path: Path, half_width_um: float = 24.0) -> None:
    target = comparison.target
    items = list(target.snapshots_xy_intensity.items())
    n = len(items)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 4.25 * nrows), squeeze=False)
    fig.suptitle(
        "Required-radius target: transverse evolution through the predicted Bessel zone\n"
        "each panel independently normalised to show structure",
        fontsize=15,
    )

    for ax, (z_um, image) in zip(axes.ravel(), items):
        xc, cropped = _crop_square(target.x_um, image, half_width_um)
        norm = cropped / max(float(np.max(cropped)), 1.0e-30)
        im = ax.imshow(
            norm,
            extent=(xc[0], xc[-1], xc[0], xc[-1]),
            origin="lower",
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(f"z = {z_um:.0f} um")
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_aspect("equal")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=list(axes.ravel()[:n]), fraction=0.025, pad=0.02, label="per-plane normalised intensity")
    fig.subplots_adjust(top=0.90, wspace=0.24, hspace=0.26, right=0.90)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _peak_profile_figure(comparison, path: Path, half_width_um: float = 28.0) -> None:
    target = comparison.target
    current = comparison.current
    d = comparison.design

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("Short-Bessel peak-plane checks", fontsize=15)

    ax = axes[0]
    xc, cropped = _crop_square(target.x_um, target.peak_xy_intensity, half_width_um)
    norm = cropped / max(float(np.max(cropped)), 1.0e-30)
    im = ax.imshow(
        norm,
        extent=(xc[0], xc[-1], xc[0], xc[-1]),
        origin="lower",
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(f"Target XY at peak z = {target.metrics.peak_z_um:.0f} um")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_aspect("equal")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    centre = int(np.argmin(np.abs(target.x_um)))
    tline = target.peak_xy_intensity[centre]
    cline = current.peak_xy_intensity[centre]
    tline = tline / max(float(np.max(tline)), 1.0e-30)
    cline = cline / max(float(np.max(cline)), 1.0e-30)
    mask = np.abs(target.x_um) <= 20.0
    ax.plot(target.x_um[mask], tline[mask], label="required radius")
    ax.plot(current.x_um[mask], cline[mask], label="current radius", linestyle="--")
    ax.axhline(0.5, linestyle=":", linewidth=1.0)
    ax.axvline(-0.5 * d.target_fwhm_um, linestyle=":", linewidth=1.0)
    ax.axvline(+0.5 * d.target_fwhm_um, linestyle=":", linewidth=1.0)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("normalised intensity")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Peak-plane central cross-section")
    ax.grid(True, alpha=0.25)
    ax.legend()

    ax = axes[2]
    ax.plot(target.z_um, target.power_ratio, label="required radius")
    ax.plot(current.z_um, current.power_ratio, label="current radius", linestyle="--")
    ax.axhline(1.0, linestyle=":", linewidth=1.0)
    ax.set_xlabel("z (um)")
    ax.set_ylabel("transverse power / input power")
    ax.set_title("BL-ASM power audit")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_outputs(*, design_input: ShortBesselDesignInput, propagation_config: ShortBesselPropagationConfig, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    design = design_short_bessel(design_input)
    comparison = run_short_bessel_propagation(design, config=propagation_config)

    metrics_path = output_dir / "short_bessel_propagation_metrics.json"
    metrics = comparison.metrics_dict()
    metrics["validation_flags"] = {
        "target_peak_fwhm_within_10_percent": bool(
            abs(comparison.target.metrics.transverse_fwhm_at_peak_um - design.target_fwhm_um)
            <= 0.10 * design.target_fwhm_um
        ),
        "target_halfmax_zone_longer_than_current": bool(
            comparison.target.metrics.on_axis_halfmax_zone_length_um
            > comparison.current.metrics.on_axis_halfmax_zone_length_um
        ),
        "target_geometric_reference_matches_requested_length": bool(
            abs(comparison.target.metrics.geometric_reference_length_um - design.target_length_um)
            <= 0.01 * design.target_length_um
        ),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    comparison_path = output_dir / "01_target_vs_current_xz.png"
    snapshots_path = output_dir / "02_target_xy_evolution.png"
    peak_path = output_dir / "03_peak_plane_checks.png"
    _comparison_figure(comparison, comparison_path)
    _snapshot_figure(comparison, snapshots_path)
    _peak_profile_figure(comparison, peak_path)

    npz_path = output_dir / "short_bessel_propagation_arrays.npz"
    snapshot_keys = sorted(comparison.target.snapshots_xy_intensity)
    np.savez_compressed(
        npz_path,
        x_um=comparison.target.x_um,
        z_um=comparison.target.z_um,
        target_xz_intensity=comparison.target.xz_intensity,
        current_xz_intensity=comparison.current.xz_intensity,
        target_on_axis_intensity=comparison.target.on_axis_intensity,
        current_on_axis_intensity=comparison.current.on_axis_intensity,
        target_peak_xy_intensity=comparison.target.peak_xy_intensity,
        current_peak_xy_intensity=comparison.current.peak_xy_intensity,
        snapshot_z_um=np.asarray(snapshot_keys, dtype=float),
        **{
            f"target_xy_z_{int(round(z_um)):04d}um": comparison.target.snapshots_xy_intensity[z_um]
            for z_um in snapshot_keys
        },
    )

    return {
        "metrics": str(metrics_path),
        "comparison_figure": str(comparison_path),
        "snapshot_figure": str(snapshots_path),
        "peak_figure": str(peak_path),
        "arrays": str(npz_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Propagate the equivalent short-Bessel conical field in the sample medium.")
    parser.add_argument("--fwhm-um", type=float, default=5.0)
    parser.add_argument("--length-um", type=float, default=500.0)
    parser.add_argument("--wavelength-nm", type=float, default=1029.0)
    parser.add_argument("--sample-index", type=float, default=1.45)
    parser.add_argument("--current-pre-radius-mm", type=float, default=2.0)
    parser.add_argument("--grid-n", type=int, default=513)
    parser.add_argument("--dx-um", type=float, default=0.25)
    parser.add_argument("--z-max-um", type=float, default=700.0)
    parser.add_argument("--z-points", type=int, default=141)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/short_bessel_propagation"))
    args = parser.parse_args()

    base = ShortBesselDesignInput()
    design_input = ShortBesselDesignInput(
        target_fwhm_um=float(args.fwhm_um),
        target_length_um=float(args.length_um),
        wavelength_nm=float(args.wavelength_nm),
        sample_index=float(args.sample_index),
        relay_demag=float(base.relay_demag),
        objective_na=float(base.objective_na),
        slm_pixel_pitch_um=float(base.slm_pixel_pitch_um),
        slm_resolution_x=int(base.slm_resolution_x),
        slm_resolution_y=int(base.slm_resolution_y),
        slm_safe_radius_fraction=float(base.slm_safe_radius_fraction),
        minimum_pixels_per_phase_wrap=float(base.minimum_pixels_per_phase_wrap),
        current_pre_beam_radius_mm=float(args.current_pre_radius_mm),
    )
    propagation_config = ShortBesselPropagationConfig(
        grid_n=int(args.grid_n),
        dx_um=float(args.dx_um),
        z_min_um=0.0,
        z_max_um=float(args.z_max_um),
        z_points=int(args.z_points),
        bandlimit=True,
        include_evanescent=True,
    )
    outputs = build_outputs(
        design_input=design_input,
        propagation_config=propagation_config,
        output_dir=args.output_dir,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
