from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
import numpy as np

from vbb_study.short_bessel import ShortBesselDesignInput, design_short_bessel
from vbb_study.short_bessel_bench_candidate import BenchCandidateConfig
from vbb_study.short_bessel_selected_order_frame import run_selected_order_frame


MM = 1e-3


@dataclass(frozen=True)
class TelescopeResult:
    input_diameter_1e2_mm: float
    input_radius_1e_field_mm: float
    f1_mm: float
    f2_mm: float
    separation_mm: float
    paraxial_magnification: float
    output_radius_1e_field_mm: float
    output_diameter_1e2_mm: float
    output_wavefront_radius_m: float
    lens_clear_radius_mm: float
    lens1_gaussian_capture: float
    lens2_gaussian_capture: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _lens_q(q: complex, f_m: float) -> complex:
    return q / (1.0 - q / f_m)


def _prop_q(q: complex, z_m: float) -> complex:
    return q + z_m


def _beam_radius_from_q(q: complex, wavelength_m: float) -> float:
    inv_q = 1.0 / q
    if inv_q.imag >= 0:
        raise ValueError("Gaussian q parameter has non-physical sign convention")
    return math.sqrt(-wavelength_m / (math.pi * inv_q.imag))


def _wavefront_radius_from_q(q: complex) -> float:
    inv_q = 1.0 / q
    if abs(inv_q.real) < 1e-15:
        return float("inf")
    return 1.0 / inv_q.real


def _gaussian_power_inside_circle(radius_aperture_m: float, beam_radius_m: float) -> float:
    # w is the 1/e field-amplitude radius = 1/e^2 intensity radius.
    return float(1.0 - math.exp(-2.0 * (radius_aperture_m / beam_radius_m) ** 2))


def galilean_telescope(
    *,
    input_diameter_1e2_mm: float,
    wavelength_nm: float,
    f1_mm: float,
    f2_mm: float,
    lens_clear_radius_mm: float,
) -> TelescopeResult:
    if f1_mm >= 0 or f2_mm <= 0:
        raise ValueError("Galilean telescope requires f1<0 and f2>0")
    w0 = 0.5 * float(input_diameter_1e2_mm) * MM
    lam = float(wavelength_nm) * 1e-9
    z_r = math.pi * w0**2 / lam
    q0 = 1j * z_r  # input assumed collimated at lens 1

    f1 = float(f1_mm) * MM
    f2 = float(f2_mm) * MM
    separation = f1 + f2  # f1 is negative; afocal Galilean spacing
    if separation <= 0:
        raise ValueError("f2 must exceed |f1| for a positive afocal spacing")

    q_after_l1 = _lens_q(q0, f1)
    q_at_l2 = _prop_q(q_after_l1, separation)
    w_l2 = _beam_radius_from_q(q_at_l2, lam)
    q_out = _lens_q(q_at_l2, f2)
    w_out = _beam_radius_from_q(q_out, lam)

    clear_r = float(lens_clear_radius_mm) * MM
    return TelescopeResult(
        input_diameter_1e2_mm=float(input_diameter_1e2_mm),
        input_radius_1e_field_mm=w0 / MM,
        f1_mm=float(f1_mm),
        f2_mm=float(f2_mm),
        separation_mm=separation / MM,
        paraxial_magnification=abs(float(f2_mm) / float(f1_mm)),
        output_radius_1e_field_mm=w_out / MM,
        output_diameter_1e2_mm=2.0 * w_out / MM,
        output_wavefront_radius_m=_wavefront_radius_from_q(q_out),
        lens_clear_radius_mm=float(lens_clear_radius_mm),
        lens1_gaussian_capture=_gaussian_power_inside_circle(clear_r, w0),
        lens2_gaussian_capture=_gaussian_power_inside_circle(clear_r, w_l2),
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _score(row: dict[str, object]) -> float:
    fwhm = float(row["central_fwhm_um"])
    l50 = float(row["sample_halfmax_zone_length_um"])
    slm_capture = min(float(row["slm1_capture_fraction"]), float(row["slm2_capture_fraction"]))
    score = abs(fwhm - 5.0) / 0.15 + abs(l50 - 500.0) / 20.0
    if slm_capture < 0.90:
        score += 20.0 * (0.90 - slm_capture)
    return float(score)


def _coarse_figure(rows: list[dict[str, object]], path: Path) -> None:
    mags = sorted({float(r["telescope_magnification"]) for r in rows})
    targets = sorted({float(r["programmed_target_fwhm_um"]) for r in rows})
    F = np.full((len(targets), len(mags)), np.nan)
    L = np.full_like(F, np.nan)
    C = np.full_like(F, np.nan)
    mi = {v: i for i, v in enumerate(mags)}
    ti = {v: i for i, v in enumerate(targets)}
    for row in rows:
        i = ti[float(row["programmed_target_fwhm_um"])]
        j = mi[float(row["telescope_magnification"])]
        F[i, j] = float(row["central_fwhm_um"])
        L[i, j] = float(row["sample_halfmax_zone_length_um"])
        C[i, j] = float(row["slm1_capture_fraction"])
    extent = (min(mags), max(mags), min(targets), max(targets))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    for ax, data, title, label in [
        (axes[0], F, "B0 core after complete candidate route", "FWHM (um)"),
        (axes[1], L, "B0 useful axial region", "L50 (um)"),
        (axes[2], C, "Rectangular SLM capture", "power fraction"),
    ]:
        im = ax.imshow(data, origin="lower", aspect="auto", extent=extent, cmap="viridis")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("Galilean magnification")
        ax.set_ylabel("programmed B0 FWHM target (um)")
        ax.set_title(title)
    axes[0].contour(mags, targets, F, levels=[5.0], linewidths=1.2)
    axes[1].contour(mags, targets, L, levels=[500.0], linewidths=1.2)
    fig.suptitle("4.0 mm input -> Galilean telescope -> SLM/4F/axicon candidate", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _final_figure(runs: dict[int, object], path: Path) -> None:
    def norm_sqrt(a: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(a, float), 0.0)
        m = float(np.max(x))
        return np.sqrt(x / m) if m > 0 else np.zeros_like(x)

    def norm(a: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(a, float), 0.0)
        m = float(np.max(x))
        return x / m if m > 0 else np.zeros_like(x)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, ell in enumerate((0, 1, 3)):
        run = runs[ell]
        m = run.metrics
        mask = np.abs(run.sample_x_um) <= 30.0
        axes[0, j].imshow(
            norm_sqrt(run.sample_xz_intensity[:, mask]),
            extent=(run.sample_x_um[mask][0], run.sample_x_um[mask][-1], run.sample_z_um[0], run.sample_z_um[-1]),
            origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=1,
        )
        axes[0, j].axhline(m.sample_halfmax_zone_start_um, linestyle="--", linewidth=1)
        axes[0, j].axhline(m.sample_halfmax_zone_end_um, linestyle="--", linewidth=1)
        axes[0, j].set_title(f"ell={ell} | L50={m.sample_halfmax_zone_length_um:.0f} um")
        axes[0, j].set_xlabel("x (um)")
        axes[0, j].set_ylabel("z (um)")

        xm = np.abs(run.sample_x_um) <= 30.0
        xx = run.sample_x_um[xm]
        xy = norm(run.sample_peak_xy_intensity)[np.ix_(xm, xm)]
        axes[1, j].imshow(xy, extent=(xx[0], xx[-1], xx[0], xx[-1]), origin="lower", cmap="inferno", vmin=0, vmax=1)
        size = m.central_fwhm_um if ell == 0 else m.measured_ring_diameter_um
        axes[1, j].set_title(f"peak XY | size={size:.2f} um | |w|={abs(m.measured_phase_winding):.2f}")
        axes[1, j].set_xlabel("x (um)")
        axes[1, j].set_ylabel("y (um)")
        axes[1, j].set_aspect("equal")
    fig.suptitle("4 mm-input telescope candidate: final B0 / V1 / V3", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Tie a measured-like 4 mm Gaussian input to an explicit Galilean telescope and the short-Bessel hardware candidate route.")
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/short_bessel_4mm_telescope"))
    ap.add_argument("--input-diameter-mm", type=float, default=4.0, help="1/e^2 intensity diameter = 2 x 1/e field radius")
    ap.add_argument("--wavelength-nm", type=float, default=1029.0)
    ap.add_argument("--f1-mm", type=float, default=-50.0)
    ap.add_argument("--lens-clear-radius-mm", type=float, default=10.0)
    ap.add_argument("--physical-fwhm-um", type=float, default=3.0, help="placeholder physical-axicon-only B0 FWHM until measured")
    ap.add_argument("--final-grid-n", type=int, default=768)
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # f2 values include a straightforward 2x pair and the approximately 2.11x
    # pair required to map a 4 mm input to the ~8.44 mm diameter suggested by
    # the earlier propagated-L50 optimization.
    f2_candidates_mm = (90.0, 95.0, 100.0, 105.0, 105.5)
    telescopes = [
        galilean_telescope(
            input_diameter_1e2_mm=float(args.input_diameter_mm),
            wavelength_nm=float(args.wavelength_nm),
            f1_mm=float(args.f1_mm),
            f2_mm=float(f2),
            lens_clear_radius_mm=float(args.lens_clear_radius_mm),
        )
        for f2 in f2_candidates_mm
    ]
    _write_csv(out / "telescope_prescriptions.csv", [t.as_dict() for t in telescopes])

    programmed_targets = (4.8, 5.0, 5.2, 5.4, 5.6)
    coarse_rows: list[dict[str, object]] = []
    for tel in telescopes:
        for programmed in programmed_targets:
            design = design_short_bessel(ShortBesselDesignInput(target_fwhm_um=float(programmed), target_length_um=500.0))
            cfg = BenchCandidateConfig(
                grid_n=384,
                input_pre_radius_mm=float(tel.output_radius_1e_field_mm),
                pinhole_radius_mm=0.50,
                measured_physical_only_fwhm_um=float(args.physical_fwhm_um),
                lens_clear_radius_mm=7.0,
            )
            run = run_selected_order_frame(design, ell=0, config=cfg, sample_z_max_um=850.0, sample_z_points=121)
            row = run.metrics.as_dict()
            row.update({
                "programmed_target_fwhm_um": float(programmed),
                "telescope_f1_mm": tel.f1_mm,
                "telescope_f2_mm": tel.f2_mm,
                "telescope_separation_mm": tel.separation_mm,
                "telescope_magnification": tel.paraxial_magnification,
                "telescope_output_diameter_mm": tel.output_diameter_1e2_mm,
                "telescope_output_wavefront_radius_m": tel.output_wavefront_radius_m,
                "telescope_lens1_capture": tel.lens1_gaussian_capture,
                "telescope_lens2_capture": tel.lens2_gaussian_capture,
            })
            row["score"] = _score(row)
            coarse_rows.append(row)

    _write_csv(out / "coarse_telescope_b0_sweep.csv", coarse_rows)
    _coarse_figure(coarse_rows, out / "01_4mm_telescope_sweep.png")
    best = min(coarse_rows, key=lambda r: float(r["score"]))
    best_f2 = float(best["telescope_f2_mm"])
    best_target = float(best["programmed_target_fwhm_um"])
    tel_best = next(t for t in telescopes if abs(t.f2_mm - best_f2) < 1e-12)
    design_best = design_short_bessel(ShortBesselDesignInput(target_fwhm_um=best_target, target_length_um=500.0))

    # Filter sensitivity with the actual selected-order frame.
    filter_rows: list[dict[str, object]] = []
    filter_runs: dict[float, dict[int, object]] = {}
    for pinhole in (0.35, 0.50, 0.65, 0.80):
        runs: dict[int, object] = {}
        for ell in (0, 1, 3):
            cfg = BenchCandidateConfig(
                grid_n=int(args.final_grid_n),
                input_pre_radius_mm=float(tel_best.output_radius_1e_field_mm),
                pinhole_radius_mm=float(pinhole),
                measured_physical_only_fwhm_um=float(args.physical_fwhm_um),
                lens_clear_radius_mm=7.0,
            )
            runs[ell] = run_selected_order_frame(design_best, ell=ell, config=cfg, sample_z_max_um=850.0, sample_z_points=171)
        b0, v1, v3 = runs[0].metrics, runs[1].metrics, runs[3].metrics
        score = (
            abs(float(b0.central_fwhm_um) - 5.0) / 0.15
            + abs(float(b0.sample_halfmax_zone_length_um) - 500.0) / 20.0
            + abs(abs(float(v1.measured_phase_winding)) - 1.0) / 0.25
            + abs(abs(float(v3.measured_phase_winding)) - 3.0) / 0.25
        )
        filter_rows.append({
            "pinhole_radius_mm": float(pinhole),
            "score": float(score),
            "B0_FWHM_um": float(b0.central_fwhm_um),
            "B0_L50_um": float(b0.sample_halfmax_zone_length_um),
            "V1_ring_um": float(v1.measured_ring_diameter_um),
            "V1_winding": float(v1.measured_phase_winding),
            "V3_ring_um": float(v3.measured_ring_diameter_um),
            "V3_winding": float(v3.measured_phase_winding),
        })
        filter_runs[float(pinhole)] = runs
    _write_csv(out / "filter_sensitivity.csv", filter_rows)
    selected_filter = min(filter_rows, key=lambda r: float(r["score"]))
    selected_pinhole = float(selected_filter["pinhole_radius_mm"])
    final_runs = filter_runs[selected_pinhole]
    _final_figure(final_runs, out / "02_4mm_telescope_final_B0_V1_V3.png")

    # The repository still carries a 0.04 mm diagnostic placeholder for the
    # SLM1->SLM2 separation. Sweep realistic free-space distances to check that
    # the candidate does not depend on that placeholder.
    spacing_rows: list[dict[str, object]] = []
    for spacing_mm in (0.04, 50.0, 100.0, 200.0, 400.0):
        cfg = BenchCandidateConfig(
            grid_n=384,
            input_pre_radius_mm=float(tel_best.output_radius_1e_field_mm),
            slm1_to_slm2_distance_mm=float(spacing_mm),
            pinhole_radius_mm=float(selected_pinhole),
            measured_physical_only_fwhm_um=float(args.physical_fwhm_um),
            lens_clear_radius_mm=7.0,
        )
        run = run_selected_order_frame(design_best, ell=0, config=cfg, sample_z_max_um=850.0, sample_z_points=121)
        row = run.metrics.as_dict()
        row["slm1_to_slm2_distance_mm_tested"] = float(spacing_mm)
        spacing_rows.append(row)
    _write_csv(out / "slm_spacing_sensitivity.csv", spacing_rows)

    report = {
        "input_assumption": {
            "diameter_1e2_intensity_mm": float(args.input_diameter_mm),
            "radius_1e_field_mm": 0.5 * float(args.input_diameter_mm),
            "wavelength_nm": float(args.wavelength_nm),
            "beam_at_lens1": "collimated Gaussian",
        },
        "selected_telescope": tel_best.as_dict(),
        "selected_programmed_target_fwhm_um": best_target,
        "selected_pinhole_radius_mm": selected_pinhole,
        "coarse_best": best,
        "final_high_resolution": {str(ell): final_runs[ell].metrics.as_dict() for ell in (0, 1, 3)},
        "filter_sensitivity": filter_rows,
        "slm_spacing_sensitivity": spacing_rows,
        "known_hardware_modelled": [
            "4.0 mm 1/e^2 input Gaussian assumption",
            "explicit afocal Galilean telescope by Gaussian q propagation",
            "1920x1080, 8 um, 8-bit rectangular SLM apertures",
            "0.93 fill-factor surrogate",
            "20-pixel / 6.25 lp/mm carrier",
            "selected +1-order reference frame",
            "nominal 300 mm scalar 4F candidate",
            "physical axicon represented by measured-physical-only FWHM parameter",
            "fixed repository objective/relay mapping",
            "homogeneous n=1.45 fused-silica BL-ASM propagation",
        ],
        "not_yet_measured_or_calibrated": [
            "actual input 1/e^2 diameter and M2",
            "actual telescope lens focal lengths/AR losses/clear apertures if added",
            "actual SLM1->SLM2 separation and conjugacy",
            "actual 4F distances and lens clear apertures",
            "actual Fourier-stop centre/radius/shape",
            "actual SLM LUT and spatial correction maps",
            "physical-axicon-only B0 FWHM at the relevant plane",
            "objective/sample-interface aberration at write depth",
            "nonlinear material response",
        ],
        "claim_boundary": final_runs[0].metrics.claim_boundary,
    }
    (out / "4mm_telescope_lab_candidate_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
