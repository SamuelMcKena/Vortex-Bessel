from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vbb_study.short_bessel import ShortBesselDesignInput, design_short_bessel, kr_from_j0_fwhm_diameter_m
from vbb_study.short_bessel_bench_candidate import BenchCandidateConfig, run_bench_candidate
from vbb_study.short_bessel_propagation import UM, MM
from vbb_study.config import SLMConfig


TWOPI = 2.0 * np.pi


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _norm_sqrt(a: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(a, dtype=float), 0.0)
    vmax = float(np.max(arr))
    return np.sqrt(arr / vmax) if vmax > 0 else np.zeros_like(arr)


def _norm(a: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(a, dtype=float), 0.0)
    vmax = float(np.max(arr))
    return arr / vmax if vmax > 0 else np.zeros_like(arr)


def _phase_gray(phi: np.ndarray) -> np.ndarray:
    return np.floor(np.mod(phi, TWOPI) / TWOPI * 256.0 + 0.5).astype(np.int64).astype(np.uint8)


def _native_masks(design, *, ell: int, physical_fwhm_um: float, out_dir: Path) -> dict[str, str]:
    slm = SLMConfig()
    x = (np.arange(slm.resolution_x) - slm.resolution_x / 2 + 0.5) * slm.pixel_pitch_m
    y = (np.arange(slm.resolution_y) - slm.resolution_y / 2 + 0.5) * slm.pixel_pitch_m
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    PHI = np.arctan2(Y, X)
    phys = kr_from_j0_fwhm_diameter_m(float(physical_fwhm_um) * UM)
    residual_sample = float(design.target_kr_sample_m_inv) - float(phys)
    residual_pre = residual_sample * float(design.relay_demag)
    carrier = TWOPI * float(slm.carrier_cpm) * X
    slm1 = int(ell) * PHI
    slm2 = -float(residual_pre) * R + carrier
    p1 = out_dir / f"SLM1_ell{ell}_vortex_only_8bit.png"
    p2 = out_dir / f"SLM2_ell{ell}_residual_axicon_plus_20px_blaze_8bit.png"
    Image.fromarray(_phase_gray(slm1), mode="L").save(p1)
    Image.fromarray(_phase_gray(slm2), mode="L").save(p2)
    return {"slm1": str(p1), "slm2": str(p2)}


def _physical_scenario_figure(rows: list[dict[str, object]], path: Path) -> None:
    x = np.array([float(r["physical_only_fwhm_um_assumption"]) for r in rows])
    resid = np.array([float(r["residual_slm_kr_pre_m_inv"]) for r in rows])
    l50 = np.array([float(r["sample_halfmax_zone_length_um"]) for r in rows])
    fwhm = np.array([float(r["central_fwhm_um"]) for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].plot(x, resid, marker="o")
    axes[0].axhline(0.0, linestyle="--", linewidth=1)
    axes[0].set_xlabel("assumed measured physical-only FWHM (um)")
    axes[0].set_ylabel("SLM2 residual k_r at pre-plane (m^-1)")
    axes[0].set_title("Signed programmable residual")
    axes[1].plot(x, fwhm, marker="o")
    axes[1].axhline(5.0, linestyle="--", linewidth=1)
    axes[1].set_xlabel("physical-only FWHM (um)")
    axes[1].set_ylabel("propagated B0 FWHM (um)")
    axes[1].set_title("Does cancellation/addition hit 5 um?")
    axes[2].plot(x, l50, marker="o")
    axes[2].axhline(500.0, linestyle="--", linewidth=1)
    axes[2].set_xlabel("physical-only FWHM (um)")
    axes[2].set_ylabel("B0 L50 (um)")
    axes[2].set_title("Axial zone after hardware-aware route")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Physical-axicon calibration dependence | pinhole 0.25 mm | nominal f=300 mm 4F", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _filter_figure(rows: list[dict[str, object]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ell in (0, 1, 3):
        rr = [r for r in rows if int(r["ell"]) == ell]
        x = np.array([float(r["pinhole_radius_mm"]) for r in rr])
        t = np.array([float(r["pinhole_transmitted_fraction"]) for r in rr])
        l = np.array([float(r["sample_halfmax_zone_length_um"]) for r in rr])
        axes[0].plot(x, t, marker="o", label=f"ell={ell}")
        axes[1].plot(x, l, marker="o", label=f"ell={ell}")
        if ell == 0:
            size = np.array([float(r["central_fwhm_um"]) for r in rr])
        else:
            size = np.array([float(r["measured_ring_diameter_um"]) for r in rr])
        axes[2].plot(x, size, marker="o", label=f"ell={ell}")
    axes[0].set_ylabel("Fourier-stop transmitted fraction")
    axes[1].set_ylabel("axial L50 (um)")
    axes[1].axhline(500.0, linestyle="--", linewidth=1)
    axes[2].set_ylabel("B0 FWHM / vortex ring diameter (um)")
    for ax in axes:
        ax.set_xlabel("pinhole radius (mm)")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes[0].set_title("Order-selection throughput")
    axes[1].set_title("Useful axial region")
    axes[2].set_title("Transverse feature after filtering")
    fig.suptitle("B0 / V1 / V3 sensitivity to the Fourier-plane pinhole", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _lens_figure(rows: list[dict[str, object]], path: Path) -> None:
    x = np.array([float(r["lens_clear_radius_mm"]) for r in rows])
    l1 = np.array([float(r["lens1_transmitted_fraction"]) for r in rows])
    l2 = np.array([float(r["lens2_transmitted_fraction"]) for r in rows])
    l50 = np.array([float(r["sample_halfmax_zone_length_um"]) for r in rows])
    fwhm = np.array([float(r["central_fwhm_um"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    axes[0].plot(x, l1, marker="o", label="lens 1")
    axes[0].plot(x, l2, marker="o", label="lens 2")
    axes[0].set_ylabel("pupil transmitted fraction")
    axes[0].legend()
    ax2 = axes[1]
    ax2.plot(x, l50, marker="o", label="L50")
    ax2.axhline(500.0, linestyle="--", linewidth=1)
    ax2.set_ylabel("L50 (um)")
    ax2b = ax2.twinx()
    ax2b.plot(x, fwhm, marker="s", linestyle=":", label="FWHM")
    ax2b.set_ylabel("B0 FWHM (um)")
    for ax in axes:
        ax.set_xlabel("assumed lens clear radius (mm)")
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Lens clipping audit")
    axes[1].set_title("Downstream consequence")
    fig.suptitle("Unknown 4F lens clear aperture is a real hardware sensitivity", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _bench_path_figure(run, path: Path) -> None:
    g = run.pre_grid
    x_mm = np.asarray(g["x"]) / MM
    smask = np.abs(run.sample_x_um) <= 30.0
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    fig.suptitle(f"Hardware-aware candidate path | ell={run.ell}", fontsize=15)
    fields = [
        (np.abs(run.slm2_output_field) ** 2, "after SLM2"),
        (np.abs(run.fourier_pre_stop_field) ** 2, "Fourier plane before stop"),
        (np.abs(run.fourier_pre_stop_field * run.fourier_stop) ** 2, "Fourier plane selected order"),
        (np.abs(run.relay_output_field) ** 2, "4F relay output"),
    ]
    for ax, (image, title) in zip(axes.ravel()[:4], fields):
        im = ax.imshow(_norm_sqrt(image), extent=(x_mm[0], x_mm[-1], x_mm[0], x_mm[-1]), origin="lower", cmap="inferno", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
    ax = axes[1, 1]
    image = _norm_sqrt(run.sample_xz_intensity[:, smask])
    im = ax.imshow(image, extent=(run.sample_x_um[smask][0], run.sample_x_um[smask][-1], run.sample_z_um[0], run.sample_z_um[-1]), origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=1)
    ax.axhline(run.metrics.sample_halfmax_zone_start_um, linestyle="--", linewidth=1)
    ax.axhline(run.metrics.sample_halfmax_zone_end_um, linestyle="--", linewidth=1)
    ax.set_title("sample XZ")
    ax.set_xlabel("x (um)")
    ax.set_ylabel("z (um)")
    ax = axes[1, 2]
    ax.axis("off")
    m = run.metrics
    txt = "\n".join([
        f"ell = {m.ell}",
        f"beam radius = {m.input_pre_radius_mm:.3f} mm",
        f"SLM capture = {m.slm1_capture_fraction:.3f} / {m.slm2_capture_fraction:.3f}",
        f"physical-only core assumption = {m.physical_only_fwhm_um_assumption:.2f} um",
        f"residual k_r(pre) = {m.residual_slm_kr_pre_m_inv:+.1f} m^-1",
        f"carrier = {m.carrier_lp_per_mm:.2f} lp/mm",
        f"order shift predicted/measured = {m.predicted_order_shift_mm:+.3f}/{m.measured_order_peak_x_mm:+.3f} mm",
        f"pinhole r = {m.pinhole_radius_mm:.3f} mm",
        f"pinhole transmission = {m.pinhole_transmitted_fraction:.3f}",
        f"lens transmissions = {m.lens1_transmitted_fraction:.3f}, {m.lens2_transmitted_fraction:.3f}",
        f"L50 = {m.sample_halfmax_zone_length_um:.1f} um",
        f"FWHM = {m.central_fwhm_um if m.central_fwhm_um is not None else float('nan'):.3f} um",
        f"ring diameter = {m.measured_ring_diameter_um if m.measured_ring_diameter_um is not None else float('nan'):.3f} um",
        f"winding = {m.measured_phase_winding:.3f}",
        "",
        "candidate screen only; unmeasured geometry remains",
    ])
    ax.text(0, 1, txt, va="top", family="monospace", fontsize=9.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _vortex_final_figure(runs: dict[int, object], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Candidate full-route B0 / V1 / V3 outputs", fontsize=15)
    for j, ell in enumerate((0, 1, 3)):
        run = runs[ell]
        m = run.metrics
        mask = np.abs(run.sample_x_um) <= 28.0
        axes[0, j].imshow(_norm_sqrt(run.sample_xz_intensity[:, mask]), extent=(run.sample_x_um[mask][0], run.sample_x_um[mask][-1], run.sample_z_um[0], run.sample_z_um[-1]), origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=1)
        axes[0, j].axhline(m.sample_halfmax_zone_start_um, linestyle="--", linewidth=1)
        axes[0, j].axhline(m.sample_halfmax_zone_end_um, linestyle="--", linewidth=1)
        axes[0, j].set_title(f"ell={ell} | L50={m.sample_halfmax_zone_length_um:.0f} um")
        axes[0, j].set_xlabel("x (um)")
        axes[0, j].set_ylabel("z (um)")
        xy = _norm(run.sample_peak_xy_intensity)
        xmask = np.abs(run.sample_x_um) <= 28.0
        crop = xy[np.ix_(xmask, xmask)]
        xx = run.sample_x_um[xmask]
        axes[1, j].imshow(crop, extent=(xx[0], xx[-1], xx[0], xx[-1]), origin="lower", cmap="inferno", vmin=0, vmax=1)
        size = m.central_fwhm_um if ell == 0 else m.measured_ring_diameter_um
        axes[1, j].set_title(f"peak XY | size={size:.2f} um | winding={m.measured_phase_winding:.2f}")
        axes[1, j].set_xlabel("x (um)")
        axes[1, j].set_ylabel("y (um)")
        axes[1, j].set_aspect("equal")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _phase_figure(runs: dict[int, object], path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.suptitle("Numerical SLM phases used in the candidate screen", fontsize=15)
    for j, ell in enumerate((0, 1, 3)):
        run = runs[ell]
        x = np.asarray(run.pre_grid["x"]) / MM
        axes[0, j].imshow(np.mod(run.slm1_phase_rad, TWOPI), extent=(x[0], x[-1], x[0], x[-1]), origin="lower", cmap="twilight", vmin=0, vmax=TWOPI)
        axes[0, j].set_title(f"SLM1 ell={ell}: vortex")
        axes[1, j].imshow(np.mod(run.slm2_phase_rad, TWOPI), extent=(x[0], x[-1], x[0], x[-1]), origin="lower", cmap="twilight", vmin=0, vmax=TWOPI)
        axes[1, j].set_title("SLM2: residual axicon + 20-px blaze")
        for ax in (axes[0, j], axes[1, j]):
            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_aspect("equal")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Hardware-aware candidate screen for the 5 um / 500 um short-Bessel route.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/short_bessel_bench_candidate"))
    parser.add_argument("--highres-grid-n", type=int, default=1024)
    parser.add_argument("--physical-fwhm-um", type=float, default=3.0, help="Assumed measured physical-axicon-only B0 FWHM for the final candidate; replace with a real measurement before lab use.")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    design = design_short_bessel(ShortBesselDesignInput(target_fwhm_um=5.0, target_length_um=500.0))

    # 1) Physical-axicon calibration dependence: B0, moderate filter, coarse-but-carrier-resolving grid.
    physical_rows: list[dict[str, object]] = []
    for pf in (3.0, 5.0, 8.0):
        cfg = BenchCandidateConfig(grid_n=512, z_points=121, pinhole_radius_mm=0.25, measured_physical_only_fwhm_um=pf)
        run = run_bench_candidate(design, ell=0, config=cfg)
        physical_rows.append(run.metrics.as_dict())
    _write_csv(out / "physical_axicon_assumption_sweep.csv", physical_rows)
    _physical_scenario_figure(physical_rows, out / "01_physical_axicon_calibration_dependence.png")

    # 2) Pinhole sensitivity for B0/V1/V3 under the deliberately challenging 3 um physical-only case.
    filter_rows: list[dict[str, object]] = []
    radii = (0.12, 0.18, 0.25, 0.35, 0.50)
    for ell in (0, 1, 3):
        for radius in radii:
            cfg = BenchCandidateConfig(grid_n=512, z_points=121, pinhole_radius_mm=radius, measured_physical_only_fwhm_um=float(args.physical_fwhm_um))
            run = run_bench_candidate(design, ell=ell, config=cfg)
            filter_rows.append(run.metrics.as_dict())
    _write_csv(out / "pinhole_sweep_B0_V1_V3.csv", filter_rows)
    _filter_figure(filter_rows, out / "02_pinhole_sweep_B0_V1_V3.png")

    # Select the smallest filter radius that preserves topology and useful scale for all three cases.
    selected = None
    for radius in radii:
        rr = [r for r in filter_rows if abs(float(r["pinhole_radius_mm"]) - radius) < 1e-12]
        ok = True
        for row in rr:
            ell = int(row["ell"])
            if abs(float(row["measured_phase_winding"]) - ell) > 0.20:
                ok = False
            if float(row["sample_halfmax_zone_length_um"]) < 420.0:
                ok = False
            if ell == 0 and not (4.4 <= float(row["central_fwhm_um"]) <= 5.7):
                ok = False
            if float(row["pinhole_transmitted_fraction"]) < 0.25:
                ok = False
        if ok:
            selected = radius
            break
    if selected is None:
        selected = 0.50

    # 3) Lens clear-aperture sensitivity with selected filter radius.
    lens_rows: list[dict[str, object]] = []
    for lens_r in (3.2, 4.5, 6.0, 7.0):
        cfg = BenchCandidateConfig(grid_n=512, z_points=101, pinhole_radius_mm=float(selected), lens_clear_radius_mm=lens_r, measured_physical_only_fwhm_um=float(args.physical_fwhm_um))
        run = run_bench_candidate(design, ell=0, config=cfg)
        row = run.metrics.as_dict()
        row["lens_clear_radius_mm"] = lens_r
        lens_rows.append(row)
    _write_csv(out / "lens_clear_aperture_sweep.csv", lens_rows)
    _lens_figure(lens_rows, out / "03_lens_clear_aperture_sensitivity.png")

    # 4) High-resolution final B0/V1/V3 candidate at selected stop radius.
    final_runs = {}
    mask_paths = {}
    for ell in (0, 1, 3):
        cfg = BenchCandidateConfig(
            grid_n=int(args.highres_grid_n),
            z_points=161,
            pinhole_radius_mm=float(selected),
            lens_clear_radius_mm=7.0,
            measured_physical_only_fwhm_um=float(args.physical_fwhm_um),
        )
        final_runs[ell] = run_bench_candidate(design, ell=ell, config=cfg)
        mask_paths[str(ell)] = _native_masks(design, ell=ell, physical_fwhm_um=float(args.physical_fwhm_um), out_dir=out)

    _bench_path_figure(final_runs[0], out / "04_full_candidate_route_B0.png")
    _vortex_final_figure(final_runs, out / "05_full_route_B0_V1_V3.png")
    _phase_figure(final_runs, out / "06_SLM_phase_roles_B0_V1_V3.png")

    report = {
        "target": {"fwhm_um": 5.0, "requested_propagated_L50_um": 500.0},
        "selected_pinhole_radius_mm": float(selected),
        "final_assumed_measured_physical_only_fwhm_um": float(args.physical_fwhm_um),
        "final_highres_grid_n": int(args.highres_grid_n),
        "final": {str(ell): final_runs[ell].metrics.as_dict() for ell in (0, 1, 3)},
        "native_1920x1080_masks": mask_paths,
        "important_unknowns": [
            "real physical-axicon-only B0 FWHM at the sample/equivalent sample plane",
            "actual pinhole diameter and physical centre for the selected carrier order",
            "actual 4F lens clear apertures and exact separations",
            "measured SLM phase LUT / spatial correction map",
            "real SLM1-to-SLM2 distance (current 0.04 mm value is a diagnostic placeholder)",
            "objective aberrations and sample-interface calibration",
        ],
        "claim_boundary": final_runs[0].metrics.claim_boundary,
    }
    (out / "short_bessel_bench_candidate_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    np.savez_compressed(
        out / "short_bessel_bench_candidate_final_arrays.npz",
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
