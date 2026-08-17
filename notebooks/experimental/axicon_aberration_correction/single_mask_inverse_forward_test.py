"""Forward-propagation falsification test for the recovered SLM2 correction.

One ideal q=20 conical Gaussian input field is constructed at the SLM/input
plane.  The recovered 2-D correction map is applied there with both signs and
the three fields are propagated with the same band-limited ASM.  In particular,
``inverse_correction`` is the user's requested experiment: ideal times the
negative of the correction phase, which should recreate the laboratory error if
the recovered map and the assumed input-annulus mapping are physically right.

The blaze is deliberately absent.  This scalar model represents the selected
first diffraction order; retaining the carrier would merely steer that order
off the numerical grid and would not represent an aberration.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from modal_vortex_bessel import load_first_scan, estimate_global_kr, find_dark_core_center


for CODE_ROOT in Path(__file__).resolve().parents:
    if ((CODE_ROOT / "vbb_study").is_dir() or
            (CODE_ROOT / "Publication_Study" / "vbb_study").is_dir()):
        if str(CODE_ROOT) not in sys.path:
            sys.path.insert(0, str(CODE_ROOT))
        break
try:
    from vbb_study.equations.fields import make_xy_grid
    from vbb_study.equations.propagation import make_bl_asm_propagator
except ModuleNotFoundError:
    from Publication_Study.vbb_study.equations.fields import make_xy_grid
    from Publication_Study.vbb_study.equations.propagation import make_bl_asm_propagator


EPS = 1.0e-12


def _normalise(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float)
    return a / max(float(np.max(a)), EPS)


def _corr_rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    av = np.asarray(a)[mask]
    bv = np.asarray(b)[mask]
    return (float(np.corrcoef(av, bv)[0, 1]),
            float(np.sqrt(np.mean((av - bv) ** 2))))


def _sample_measured(images, axis_um, pixel_pitch_m):
    X_um, Y_um = np.meshgrid(axis_um, axis_um, indexing="xy")
    out = []
    centers = []
    for image in images:
        cy, cx, _ = find_dark_core_center(image)
        yy = cy + Y_um * 1e-6 / pixel_pitch_m
        xx = cx + X_um * 1e-6 / pixel_pitch_m
        out.append(_normalise(ndimage.map_coordinates(
            image, [yy, xx], order=1, mode="constant", cval=0.0)))
        centers.append((cy, cx))
    return np.stack(out), centers


def _sample_model_intensity(field, grid, axis_um):
    coord_px = axis_um * 1e-6 / float(grid["dx"]) + (int(grid["N"]) - 1) / 2
    xx, yy = np.meshgrid(coord_px, coord_px, indexing="xy")
    intensity = np.abs(field) ** 2
    return _normalise(ndimage.map_coordinates(
        intensity, [yy, xx], order=1, mode="constant", cval=0.0))


def _map_wrapped_phase_to_grid(correction, grid, beam_radius_m):
    """Interpolate a wrapped phase through its unit phasor, avoiding wrap seams."""
    correction = np.asarray(correction, float)
    valid = np.isfinite(correction)
    phasor = np.zeros(correction.shape, complex)
    phasor[valid] = np.exp(1j * correction[valid])
    ny, nx = correction.shape
    sx = (grid["X"] / beam_radius_m + 1.0) * (nx - 1) / 2.0
    sy = (grid["Y"] / beam_radius_m + 1.0) * (ny - 1) / 2.0
    re = ndimage.map_coordinates(phasor.real, [sy, sx], order=1,
                                 mode="constant", cval=1.0)
    im = ndimage.map_coordinates(phasor.imag, [sy, sx], order=1,
                                 mode="constant", cval=0.0)
    mapped = np.angle(re + 1j * im)
    mapped[grid["R"] > beam_radius_m] = 0.0
    return mapped


def _map_annular_correction_to_grid(phase_rows, z_absolute_mm, grid,
                                     kr_m_inv, wavelength_m):
    """Map retrieved camera-z annuli to their conical-ray input radii.

    For a cone angle beta, r_input = z*tan(beta), with
    tan(beta)=kr/sqrt(k**2-kr**2).  The input array contains the retrieved
    *error* phase, so the returned map is its negative (the correction phase).
    Unmeasured inner and outer radii are deliberately left at zero phase.
    """
    phase_rows = np.asarray(phase_rows, float)
    z_absolute_mm = np.asarray(z_absolute_mm, float)
    k = 2 * np.pi / float(wavelength_m)
    kz = np.sqrt(max(k * k - float(kr_m_inv) ** 2, EPS))
    z_for_radius_mm = grid["R"] * kz / float(kr_m_inv) * 1e3
    row = np.interp(z_for_radius_mm, z_absolute_mm,
                    np.arange(len(z_absolute_mm), dtype=float))
    valid = ((z_for_radius_mm >= z_absolute_mm.min()) &
             (z_for_radius_mm <= z_absolute_mm.max()))
    # Append the first angular sample so interpolation is continuous at 2pi.
    correction_phasor = np.exp(-1j * phase_rows)
    correction_phasor = np.concatenate(
        [correction_phasor, correction_phasor[:, :1]], axis=1)
    theta_coord = np.mod(grid["PHI"], 2 * np.pi) / (2 * np.pi) * phase_rows.shape[1]
    re = ndimage.map_coordinates(correction_phasor.real, [row, theta_coord],
                                 order=1, mode="nearest")
    im = ndimage.map_coordinates(correction_phasor.imag, [row, theta_coord],
                                 order=1, mode="nearest")
    mapped = np.angle(re + 1j * im)
    mapped[~valid] = 0.0
    return mapped, (float(z_absolute_mm.min() * 1e-3 * kr_m_inv / kz),
                    float(z_absolute_mm.max() * 1e-3 * kr_m_inv / kz))


def _azimuth_cv(image, axis_um, ring_radius_um=45.4, half_width_um=8.0):
    X, Y = np.meshgrid(axis_um, axis_um, indexing="xy")
    R = np.hypot(X, Y)
    ann = np.abs(R - ring_radius_um) <= half_width_um
    values = image[ann]
    return float(np.std(values) / max(float(np.mean(values)), EPS))


def run_single_mask_inverse_test(
        data_dir, correction_path, output_dir, *, z_relative_mm=np.arange(-17, 1),
        wavelength_m=1030e-9, pixel_pitch_m=5.5e-6, q=20,
        beam_radius_m=2.0e-3, grid_n=1024, grid_dx_m=5.5e-6,
        view_limit_um=180.0, view_size=181):
    data_dir = Path(data_dir)
    correction_path = Path(correction_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = load_first_scan(data_dir)
    z_relative_mm = np.asarray(z_relative_mm, float)
    if len(images) != len(z_relative_mm):
        raise ValueError("z_relative_mm must contain one position per measured plane")

    kr_m_inv, geometry = estimate_global_kr(images, pixel_pitch_m, q, .55)
    axis_um = np.linspace(-view_limit_um, view_limit_um, view_size)
    measured, centers = _sample_measured(images, axis_um, pixel_pitch_m)
    Xv, Yv = np.meshgrid(axis_um, axis_um, indexing="xy")
    roi = np.hypot(Xv, Yv) <= 160.0

    grid = make_xy_grid(grid_n, grid_dx_m)
    correction = np.load(correction_path)
    correction_slm = _map_wrapped_phase_to_grid(correction, grid, beam_radius_m)
    aperture = np.exp(-(grid["R"] / beam_radius_m) ** 2)
    ideal_input = aperture * np.exp(1j * (q * grid["PHI"] - kr_m_inv * grid["R"]))
    inverse_input = ideal_input * np.exp(-1j * correction_slm)
    correction_input = ideal_input * np.exp(+1j * correction_slm)

    prop_ideal = make_bl_asm_propagator(ideal_input, grid, wavelength_m)
    prop_inverse = make_bl_asm_propagator(inverse_input, grid, wavelength_m)
    prop_correction = make_bl_asm_propagator(correction_input, grid, wavelength_m)

    cache_path = output_dir / "single_mask_forward_stacks.npz"
    cached = np.load(cache_path) if cache_path.exists() else None

    # Absolute camera-to-input distance is not recorded.  Select only this one
    # nuisance parameter using the unaberrated ideal field, then freeze it for
    # both phase-map signs.  Three spaced planes keep this a cheap, transparent
    # registration rather than a fit of the recovered error to the data.
    if cached is not None and "correction_sign" in cached:
        absolute_z_mm = cached["z_absolute_model_mm"].copy()
        absolute_end_mm = float(absolute_z_mm[-1])
        ideal_stack = cached["ideal"].copy()
        inverse_stack = cached["inverse_correction"].copy()
        correction_stack = cached["correction_sign"].copy()
        cached.close()
        search_file = output_dir / "ideal_absolute_z_registration.csv"
        search = pd.read_csv(search_file) if search_file.exists() else pd.DataFrame()
    else:
        coarse_mm = np.arange(18.0, 30.01, 2.0)
        anchor_indices = np.array([0, len(images) - 1])
        search_rows = []
        def evaluate_end(end_mm):
            correlations = []
            for iz in anchor_indices:
                absolute_mm = end_mm + z_relative_mm[iz]
                model = _sample_model_intensity(prop_ideal(absolute_mm * 1e-3), grid, axis_um)
                correlations.append(_corr_rmse(measured[iz], model, roi)[0])
            search_rows.append({"absolute_z_at_relative_zero_mm": end_mm,
                                "mean_anchor_ideal_correlation": float(np.mean(correlations))})
        for end_mm in coarse_mm:
            evaluate_end(float(end_mm))
        coarse = pd.DataFrame(search_rows)
        coarse_best = float(coarse.loc[coarse.mean_anchor_ideal_correlation.idxmax(),
                                        "absolute_z_at_relative_zero_mm"])
        for end_mm in np.arange(max(17.25, coarse_best - 1.5),
                                min(31.0, coarse_best + 1.5) + .01, .5):
            if not np.any(np.isclose(coarse_mm, end_mm)):
                evaluate_end(float(end_mm))
        search = pd.DataFrame(search_rows)
        absolute_end_mm = float(search.loc[search.mean_anchor_ideal_correlation.idxmax(),
                                           "absolute_z_at_relative_zero_mm"])
        absolute_z_mm = absolute_end_mm + z_relative_mm
        ideal_stack = []
        inverse_stack = []
        correction_stack = []
        for z_mm in absolute_z_mm:
            ideal_stack.append(_sample_model_intensity(prop_ideal(z_mm * 1e-3), grid, axis_um))
            inverse_stack.append(_sample_model_intensity(prop_inverse(z_mm * 1e-3), grid, axis_um))
            correction_stack.append(_sample_model_intensity(prop_correction(z_mm * 1e-3), grid, axis_um))
        ideal_stack = np.stack(ideal_stack)
        inverse_stack = np.stack(inverse_stack)
        correction_stack = np.stack(correction_stack)

    phase_rows_path = correction_path.with_name("annular_aberration_phase.npy")
    phase_rows = np.load(phase_rows_path)
    physical_correction_slm, annulus_range_m = _map_annular_correction_to_grid(
        phase_rows, absolute_z_mm, grid, kr_m_inv, wavelength_m)
    physical_inverse_input = ideal_input * np.exp(-1j * physical_correction_slm)
    prop_physical_inverse = make_bl_asm_propagator(
        physical_inverse_input, grid, wavelength_m)
    physical_inverse_stack = np.stack([
        _sample_model_intensity(prop_physical_inverse(z_mm * 1e-3), grid, axis_um)
        for z_mm in absolute_z_mm])

    rows = []
    for iz, z_rel in enumerate(z_relative_mm):
        ci, ei = _corr_rmse(measured[iz], ideal_stack[iz], roi)
        ce, ee = _corr_rmse(measured[iz], inverse_stack[iz], roi)
        cc, ec = _corr_rmse(measured[iz], correction_stack[iz], roi)
        cp, ep = _corr_rmse(measured[iz], physical_inverse_stack[iz], roi)
        rows.append({
            "z_relative_mm": z_rel, "z_absolute_model_mm": absolute_z_mm[iz],
            "measured_vs_ideal_corr": ci, "measured_vs_inverse_correction_corr": ce,
            "measured_vs_correction_corr": cc,
            "measured_vs_physical_annulus_inverse_corr": cp,
            "measured_vs_ideal_rmse": ei, "measured_vs_inverse_correction_rmse": ee,
            "measured_vs_correction_rmse": ec,
            "measured_vs_physical_annulus_inverse_rmse": ep,
            "measured_annular_cv": _azimuth_cv(measured[iz], axis_um),
            "ideal_annular_cv": _azimuth_cv(ideal_stack[iz], axis_um),
            "inverse_correction_annular_cv": _azimuth_cv(inverse_stack[iz], axis_um),
            "correction_annular_cv": _azimuth_cv(correction_stack[iz], axis_um),
            "physical_annulus_inverse_cv": _azimuth_cv(physical_inverse_stack[iz], axis_um),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "single_mask_forward_metrics.csv", index=False)
    search.to_csv(output_dir / "ideal_absolute_z_registration.csv", index=False)
    np.savez_compressed(output_dir / "single_mask_forward_stacks.npz",
                       x_um=axis_um, z_relative_mm=z_relative_mm,
                       z_absolute_model_mm=absolute_z_mm, measured=measured,
                       ideal=ideal_stack, inverse_correction=inverse_stack,
                       correction_sign=correction_stack,
                       physical_annulus_inverse=physical_inverse_stack,
                       mapped_correction_phase_rad=correction_slm.astype(np.float32),
                       physical_mapped_correction_phase_rad=physical_correction_slm.astype(np.float32))

    gain = float(metrics.measured_vs_inverse_correction_corr.median()
                 - metrics.measured_vs_ideal_corr.median())
    rmse_gain = float(metrics.measured_vs_ideal_rmse.median()
                      - metrics.measured_vs_inverse_correction_rmse.median())
    physical_gain = float(metrics.measured_vs_physical_annulus_inverse_corr.median()
                          - metrics.measured_vs_ideal_corr.median())
    physical_rmse_gain = float(metrics.measured_vs_ideal_rmse.median()
                               - metrics.measured_vs_physical_annulus_inverse_rmse.median())
    supports = bool(physical_gain >= 0.03 and physical_rmse_gain > 0)
    summary = {
        "test": "single input-plane mask followed by one common BL-ASM propagation model",
        "requested_inverse_operation": "ideal_input * exp(-i * recovered_correction_phase)",
        "q_on_slm1": q,
        "slm2_role": "correction phase only; blaze interpreted as selected diffraction order",
        "planes": len(images),
        "kr_rad_per_um": float(kr_m_inv * 1e-6),
        "beam_radius_mm_nominal": beam_radius_m * 1e3,
        "absolute_z_at_relative_zero_mm_fitted_from_ideal_only": absolute_end_mm,
        "median_measured_vs_ideal_corr": float(metrics.measured_vs_ideal_corr.median()),
        "median_measured_vs_inverse_correction_corr": float(metrics.measured_vs_inverse_correction_corr.median()),
        "median_measured_vs_correction_corr": float(metrics.measured_vs_correction_corr.median()),
        "inverse_correlation_gain_over_ideal": gain,
        "median_measured_vs_ideal_rmse": float(metrics.measured_vs_ideal_rmse.median()),
        "median_measured_vs_inverse_correction_rmse": float(metrics.measured_vs_inverse_correction_rmse.median()),
        "inverse_rmse_reduction_from_ideal": rmse_gain,
        "physical_input_annulus_radius_range_mm": [annulus_range_m[0] * 1e3,
                                                     annulus_range_m[1] * 1e3],
        "median_measured_vs_physical_annulus_inverse_corr": float(
            metrics.measured_vs_physical_annulus_inverse_corr.median()),
        "physical_annulus_inverse_correlation_gain_over_ideal": physical_gain,
        "median_measured_vs_physical_annulus_inverse_rmse": float(
            metrics.measured_vs_physical_annulus_inverse_rmse.median()),
        "physical_annulus_inverse_rmse_reduction_from_ideal": physical_rmse_gain,
        "supports_phase_map_as_cause_of_measured_distortion": supports,
        "hardware_ready": False,
        "remaining_calibration": "camera z to SLM annulus radius, relay magnification, rotation, parity, phase LUT",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Full signed transverse planes, six rows per page.
    stacks = (measured, ideal_stack, inverse_stack, physical_inverse_stack, correction_stack)
    titles = ("LAB MEASURED", "IDEAL FORWARD MODEL",
              "INVERSE: NORMALIZED MAP", "INVERSE: PHYSICAL ANNULI",
              "IDEAL + CORRECTION SIGN")
    extent = [axis_um[0], axis_um[-1], axis_um[0], axis_um[-1]]
    for page, start in enumerate(range(0, len(images), 6), 1):
        stop = min(start + 6, len(images))
        fig, axes = plt.subplots(stop - start, 5, figsize=(18, 3.25 * (stop - start)),
                                 constrained_layout=True, squeeze=False)
        for row, iz in enumerate(range(start, stop)):
            for col, (stack, title) in enumerate(zip(stacks, titles)):
                im = axes[row, col].imshow(stack[iz], origin="lower", extent=extent,
                                           cmap="inferno", vmin=0, vmax=1,
                                           interpolation="nearest")
                axes[row, col].axhline(0, color="white", lw=.35, alpha=.45)
                axes[row, col].axvline(0, color="white", lw=.35, alpha=.45)
                axes[row, col].set_title(title, fontsize=9)
                axes[row, col].text(.02, .97, f"z={z_relative_mm[iz]:g} mm",
                                    transform=axes[row, col].transAxes, va="top",
                                    color="white", fontsize=8,
                                    bbox=dict(facecolor="black", alpha=.55, pad=2,
                                              edgecolor="none"))
                axes[row, col].set(xlabel="x (um)", ylabel="y (um)")
                axes[row, col].set_aspect("equal")
        fig.colorbar(im, ax=axes, label="plane-normalized intensity", shrink=.7)
        fig.suptitle("Single-mask input-plane forward propagation — full signed x/y")
        fig.savefig(output_dir / f"single_mask_planes_page_{page}.png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)

    # Signed x-z and y-z sections preserve both negative and positive coordinates.
    mid = view_size // 2
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), constrained_layout=True)
    for row, (take, label) in enumerate(((lambda a: a[:, mid, :], "x-z at y=0"),
                                         (lambda a: a[:, :, mid], "y-z at x=0"))):
        for col, (stack, title) in enumerate(zip(stacks, titles)):
            im = axes[row, col].imshow(take(stack), origin="lower", aspect="auto",
                                       cmap="inferno", vmin=0, vmax=1,
                                       extent=[axis_um[0], axis_um[-1],
                                               z_relative_mm[0], z_relative_mm[-1]])
            axes[row, col].set(title=f"{label} | {title}",
                               xlabel="signed transverse coordinate (um)",
                               ylabel="relative z (mm)")
    fig.colorbar(im, ax=axes, label="plane-normalized intensity", shrink=.8)
    fig.suptitle("Measured versus single-input-mask propagation")
    fig.savefig(output_dir / "single_mask_signed_xz_yz.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # Linear cross sections at entrance, middle and final measured planes.
    chosen = [0, len(images) // 2, len(images) - 1]
    fig, axes = plt.subplots(len(chosen), 2, figsize=(13, 10), constrained_layout=True)
    colours = ("black", "#377eb8", "#e41a1c", "#984ea3", "#4daf4a")
    for row, iz in enumerate(chosen):
        for col, (direction, selector) in enumerate((("x cut (y=0)", lambda a: a[mid, :]),
                                                      ("y cut (x=0)", lambda a: a[:, mid]))):
            for stack, title, colour in zip(stacks, titles, colours):
                axes[row, col].plot(axis_um, selector(stack[iz]), lw=1.6,
                                    label=title, color=colour)
            axes[row, col].set(title=f"z={z_relative_mm[iz]:g} mm — {direction}",
                               xlabel="signed coordinate (um)", ylabel="normalized intensity",
                               xlim=(-view_limit_um, view_limit_um), ylim=(-.02, 1.05))
            axes[row, col].grid(alpha=.25)
            axes[row, col].legend(fontsize=7)
    fig.suptitle("Measured / ideal / inverse-map / correction-map linear profiles")
    fig.savefig(output_dir / "single_mask_linear_cross_sections.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].plot(z_relative_mm, metrics.measured_vs_ideal_corr, "o-", label="ideal")
    axes[0].plot(z_relative_mm, metrics.measured_vs_inverse_correction_corr, "o-", label="inverse correction")
    axes[0].plot(z_relative_mm, metrics.measured_vs_correction_corr, "o-", label="correction sign")
    axes[0].plot(z_relative_mm, metrics.measured_vs_physical_annulus_inverse_corr,
                 "o-", label="inverse, physical annuli")
    axes[0].set(ylabel="correlation", ylim=(-.05, 1.05), title="Agreement with lab image")
    axes[1].plot(z_relative_mm, metrics.measured_vs_ideal_rmse, "o-", label="ideal")
    axes[1].plot(z_relative_mm, metrics.measured_vs_inverse_correction_rmse, "o-", label="inverse correction")
    axes[1].plot(z_relative_mm, metrics.measured_vs_correction_rmse, "o-", label="correction sign")
    axes[1].plot(z_relative_mm, metrics.measured_vs_physical_annulus_inverse_rmse,
                 "o-", label="inverse, physical annuli")
    axes[1].set(ylabel="normalized RMSE", title="Pixelwise intensity error")
    axes[2].plot(z_relative_mm, metrics.measured_annular_cv, "o-", label="measured")
    axes[2].plot(z_relative_mm, metrics.ideal_annular_cv, "o-", label="ideal")
    axes[2].plot(z_relative_mm, metrics.inverse_correction_annular_cv, "o-", label="inverse correction")
    axes[2].plot(z_relative_mm, metrics.correction_annular_cv, "o-", label="correction sign")
    axes[2].plot(z_relative_mm, metrics.physical_annulus_inverse_cv,
                 "o-", label="inverse, physical annuli")
    axes[2].set(ylabel="annular coefficient of variation", title="Ring non-uniformity")
    for ax in axes:
        ax.set_xlabel("relative z (mm)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Single-mask forward-propagation verdict")
    fig.savefig(output_dir / "single_mask_metrics_vs_z.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)
    return metrics, summary


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    result_dir = (here / "outputs" / "slm_closed_loop_alignment" / "modal_q20" /
                  "single_mask_inverse_forward_test")
    _, report = run_single_mask_inverse_test(
        here / "z-scan 2 1010",
        here / "outputs" / "slm_closed_loop_alignment" / "modal_q20" /
        "UNCALIBRATED_DO_NOT_APPLY_q20_modal_correction.npy",
        result_dir)
    print(json.dumps(report, indent=2))
