from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

from vbb_study.short_bessel import ShortBesselDesignInput, design_short_bessel
from vbb_study.short_bessel_bench_candidate import BenchCandidateConfig
from vbb_study.short_bessel_selected_order_frame import run_selected_order_frame


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sqrt_norm(a: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(a, dtype=float), 0.0)
    m = float(np.max(x))
    return np.sqrt(x / m) if m > 0 else np.zeros_like(x)


def _norm(a: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(a, dtype=float), 0.0)
    m = float(np.max(x))
    return x / m if m > 0 else np.zeros_like(x)


def _global_vs_aligned_figure(run, path: Path) -> None:
    raw = run.upstream
    m = run.metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("Why the +1 diffraction order must be treated as the downstream optical axis", fontsize=14)
    raw_mask = np.abs(raw.sample_x_um) <= 45.0
    axes[0].imshow(
        _sqrt_norm(raw.sample_xz_intensity[:, raw_mask]),
        extent=(raw.sample_x_um[raw_mask][0], raw.sample_x_um[raw_mask][-1], raw.sample_z_um[0], raw.sample_z_um[-1]),
        origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=1,
    )
    axes[0].set_title("global lab-axis diagnostic\n(carrier tilt retained)")
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("z (um)")

    mask = np.abs(run.sample_x_um) <= 45.0
    axes[1].imshow(
        _sqrt_norm(run.sample_xz_intensity[:, mask]),
        extent=(run.sample_x_um[mask][0], run.sample_x_um[mask][-1], run.sample_z_um[0], run.sample_z_um[-1]),
        origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=1,
    )
    axes[1].axhline(m.sample_halfmax_zone_start_um, linestyle="--", linewidth=1)
    axes[1].axhline(m.sample_halfmax_zone_end_um, linestyle="--", linewidth=1)
    axes[1].set_title("selected-order frame\n(mean carrier ramp removed)")
    axes[1].set_xlabel("x (um)")
    axes[1].set_ylabel("z (um)")

    axes[2].axis("off")
    text = "\n".join([
        f"raw 4F mean tilt x = {m.raw_mean_tilt_x_cpm/1000:.3f} kcpm",
        f"raw 4F mean tilt y = {m.raw_mean_tilt_y_cpm/1000:.3f} kcpm",
        f"tilt angle = {m.raw_mean_tilt_angle_mrad:.3f} mrad",
        f"Fourier-order peak = ({m.fourier_order_peak_x_mm:.3f}, {m.fourier_order_peak_y_mm:.3f}) mm",
        f"pinhole r = {m.pinhole_radius_mm:.3f} mm",
        "",
        f"aligned B0 FWHM = {m.central_fwhm_um:.3f} um",
        f"aligned B0 L50 = {m.sample_halfmax_zone_length_um:.1f} um",
        f"peak z = {m.sample_peak_z_um:.1f} um",
        "",
        "The re-reference is a coordinate/beam-axis change,",
        "not a fitted phase correction.",
    ])
    axes[2].text(0, 1, text, va="top", family="monospace", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _sweep_figure(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    for ell in (0, 1, 3):
        rr = [r for r in rows if int(r["ell"]) == ell]
        x = np.array([float(r["pinhole_radius_mm"]) for r in rr])
        trans = np.array([float(r["pinhole_transmitted_fraction"]) for r in rr])
        l50 = np.array([float(r["sample_halfmax_zone_length_um"]) for r in rr])
        winding = np.array([abs(float(r["measured_phase_winding"])) for r in rr])
        axes[0, 0].plot(x, trans, marker="o", label=f"ell={ell}")
        axes[0, 1].plot(x, l50, marker="o", label=f"ell={ell}")
        axes[1, 0].plot(x, winding, marker="o", label=f"ell={ell}")
        if ell == 0:
            size = np.array([float(r["central_fwhm_um"]) for r in rr])
            axes[1, 1].plot(x, size, marker="o", label="B0 FWHM")
        else:
            ring = np.array([float(r["measured_ring_diameter_um"]) for r in rr])
            axes[1, 1].plot(x, ring, marker="o", label=f"V{ell} ring")
    axes[0, 1].axhline(500.0, linestyle="--", linewidth=1)
    axes[1, 0].axhline(1.0, linestyle=":", linewidth=1)
    axes[1, 0].axhline(3.0, linestyle=":", linewidth=1)
    axes[1, 1].axhline(5.0, linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("Fourier-stop transmitted fraction")
    axes[0, 1].set_ylabel("axial L50 (um)")
    axes[1, 0].set_ylabel("|measured phase winding|")
    axes[1, 1].set_ylabel("transverse feature diameter (um)")
    for ax in axes.ravel():
        ax.set_xlabel("pinhole radius (mm)")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[0, 0].set_title("Selected-order throughput")
    axes[0, 1].set_title("Useful axial region")
    axes[1, 0].set_title("Vortex topology after selection")
    axes[1, 1].set_title("B0 core / vortex ring scale")
    fig.suptitle("Selected-order-frame pinhole sweep: B0 / V1 / V3", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _physical_sweep_figure(rows: list[dict[str, object]], path: Path) -> None:
    x = np.array([float(r["physical_only_fwhm_um_assumption"]) for r in rows])
    fwhm = np.array([float(r["central_fwhm_um"]) for r in rows])
    l50 = np.array([float(r["sample_halfmax_zone_length_um"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].plot(x, fwhm, marker="o")
    axes[0].axhline(5.0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("B0 FWHM (um)")
    axes[1].plot(x, l50, marker="o")
    axes[1].axhline(500.0, linestyle="--", linewidth=1)
    axes[1].set_ylabel("B0 L50 (um)")
    for ax in axes:
        ax.set_xlabel("assumed measured physical-only FWHM (um)")
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Residual radial phase: transverse target")
    axes[1].set_title("Residual radial phase: axial target")
    fig.suptitle("Dependence on physical-axicon calibration assumption", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _final_figure(runs: dict[int, object], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Selected-order-frame candidate outputs after rectangular SLM + 4F + physical axicon", fontsize=14)
    for j, ell in enumerate((0, 1, 3)):
        run = runs[ell]
        m = run.metrics
        mask = np.abs(run.sample_x_um) <= 30.0
        axes[0, j].imshow(
            _sqrt_norm(run.sample_xz_intensity[:, mask]),
            extent=(run.sample_x_um[mask][0], run.sample_x_um[mask][-1], run.sample_z_um[0], run.sample_z_um[-1]),
            origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=1,
        )
        axes[0, j].axhline(m.sample_halfmax_zone_start_um, linestyle="--", linewidth=1)
        axes[0, j].axhline(m.sample_halfmax_zone_end_um, linestyle="--", linewidth=1)
        axes[0, j].set_title(f"ell={ell} | L50={m.sample_halfmax_zone_length_um:.0f} um")
        axes[0, j].set_xlabel("x (um)")
        axes[0, j].set_ylabel("z (um)")
        xy = _norm(run.sample_peak_xy_intensity)
        xm = np.abs(run.sample_x_um) <= 30.0
        xx = run.sample_x_um[xm]
        axes[1, j].imshow(xy[np.ix_(xm, xm)], extent=(xx[0], xx[-1], xx[0], xx[-1]), origin="lower", cmap="inferno", vmin=0, vmax=1)
        size = m.central_fwhm_um if ell == 0 else m.measured_ring_diameter_um
        axes[1, j].set_title(f"peak XY | size={size:.2f} um | |w|={abs(m.measured_phase_winding):.2f}")
        axes[1, j].set_xlabel("x (um)")
        axes[1, j].set_ylabel("y (um)")
        axes[1, j].set_aspect("equal")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _score_group(group: list[dict[str, object]]) -> float:
    by_ell = {int(r["ell"]): r for r in group}
    if set(by_ell) != {0, 1, 3}:
        return float("inf")
    b0, v1, v3 = by_ell[0], by_ell[1], by_ell[3]
    score = 0.0
    score += abs(float(b0["central_fwhm_um"]) - 5.0) / 0.5
    score += abs(float(b0["sample_halfmax_zone_length_um"]) - 500.0) / 75.0
    score += abs(abs(float(v1["measured_phase_winding"])) - 1.0) / 0.25
    score += abs(abs(float(v3["measured_phase_winding"])) - 3.0) / 0.25
    if v1["measured_ring_diameter_um"] is not None:
        score += abs(float(v1["measured_ring_diameter_um"]) - float(v1["predicted_ring_diameter_um"])) / 3.0
    if v3["measured_ring_diameter_um"] is not None:
        score += abs(float(v3["measured_ring_diameter_um"]) - float(v3["predicted_ring_diameter_um"])) / 4.0
    for row in group:
        t = float(row["pinhole_transmitted_fraction"])
        if t < 0.20:
            score += 10.0 * (0.20 - t)
    return float(score)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Re-reference the selected +1 order and evaluate the downstream short-Bessel candidate.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/short_bessel_selected_order_frame"))
    parser.add_argument("--final-grid-n", type=int, default=768)
    parser.add_argument("--physical-fwhm-um", type=float, default=3.0)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    design = design_short_bessel(ShortBesselDesignInput(target_fwhm_um=5.0, target_length_um=500.0))
    radii = (0.12, 0.18, 0.25, 0.35, 0.50, 0.65, 0.80)

    rows: list[dict[str, object]] = []
    for radius in radii:
        for ell in (0, 1, 3):
            cfg = BenchCandidateConfig(
                grid_n=512,
                pinhole_radius_mm=float(radius),
                measured_physical_only_fwhm_um=float(args.physical_fwhm_um),
                lens_clear_radius_mm=7.0,
            )
            run = run_selected_order_frame(design, ell=ell, config=cfg, sample_z_max_um=800.0, sample_z_points=121)
            rows.append(run.metrics.as_dict())
    _write_csv(out / "selected_order_pinhole_sweep_B0_V1_V3.csv", rows)
    _sweep_figure(rows, out / "01_selected_order_pinhole_sweep.png")

    scored = []
    for radius in radii:
        group = [r for r in rows if abs(float(r["pinhole_radius_mm"]) - radius) < 1e-12]
        scored.append((float(_score_group(group)), float(radius)))
    selected_score, selected_radius = min(scored)

    physical_rows: list[dict[str, object]] = []
    for pf in (3.0, 5.0, 8.0):
        cfg = BenchCandidateConfig(
            grid_n=512,
            pinhole_radius_mm=float(selected_radius),
            measured_physical_only_fwhm_um=pf,
            lens_clear_radius_mm=7.0,
        )
        run = run_selected_order_frame(design, ell=0, config=cfg, sample_z_max_um=800.0, sample_z_points=121)
        physical_rows.append(run.metrics.as_dict())
    _write_csv(out / "selected_order_physical_axicon_sweep.csv", physical_rows)
    _physical_sweep_figure(physical_rows, out / "02_selected_order_physical_calibration.png")

    final_runs = {}
    for ell in (0, 1, 3):
        cfg = BenchCandidateConfig(
            grid_n=int(args.final_grid_n),
            pinhole_radius_mm=float(selected_radius),
            measured_physical_only_fwhm_um=float(args.physical_fwhm_um),
            lens_clear_radius_mm=7.0,
        )
        final_runs[ell] = run_selected_order_frame(design, ell=ell, config=cfg, sample_z_max_um=800.0, sample_z_points=161)

    _global_vs_aligned_figure(final_runs[0], out / "03_global_axis_vs_selected_order_frame_B0.png")
    _final_figure(final_runs, out / "04_selected_order_final_B0_V1_V3.png")

    report = {
        "target": {"B0_fwhm_um": 5.0, "B0_L50_um": 500.0},
        "selected_pinhole_radius_mm": float(selected_radius),
        "selected_score": float(selected_score),
        "assumed_physical_only_fwhm_um": float(args.physical_fwhm_um),
        "final_grid_n": int(args.final_grid_n),
        "final": {str(ell): final_runs[ell].metrics.as_dict() for ell in (0, 1, 3)},
        "selection_note": "pinhole radius is selected numerically from the declared sweep by a transparent score on B0 core/L50, V1/V3 winding magnitude, ring scale and low-throughput penalty; this is not a calibrated hardware prescription.",
        "claim_boundary": final_runs[0].metrics.claim_boundary,
    }
    (out / "selected_order_frame_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        out / "selected_order_frame_final_arrays.npz",
        sample_x_um=final_runs[0].sample_x_um,
        sample_z_um=final_runs[0].sample_z_um,
        B0_xz=final_runs[0].sample_xz_intensity,
        V1_xz=final_runs[1].sample_xz_intensity,
        V3_xz=final_runs[3].sample_xz_intensity,
        B0_peak_xy=final_runs[0].sample_peak_xy_intensity,
        V1_peak_xy=final_runs[1].sample_peak_xy_intensity,
        V3_peak_xy=final_runs[3].sample_peak_xy_intensity,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
