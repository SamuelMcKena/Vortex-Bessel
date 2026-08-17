"""Run the q=20 constrained Bessel-modal diagnostic on a Beamage z scan.

The generated correction is normalized annulus coordinates only.  It is deliberately
blocked from hardware use until z-to-input-radius and SLM coordinate/LUT calibration exist.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

from modal_vortex_bessel import load_first_scan, estimate_global_kr, fit_plane, modal_basis

# Reuse the validated analytic target from either supported repository layout.
for CODE_ROOT in Path(__file__).resolve().parents:
    if ((CODE_ROOT / "vbb_study").is_dir() or
            (CODE_ROOT / "Publication_Study" / "vbb_study").is_dir()):
        if str(CODE_ROOT) not in sys.path:
            sys.path.insert(0, str(CODE_ROOT))
        break
try:
    from vbb_study.equations.scalar_bessel import bessel_gauss_field
except ModuleNotFoundError:
    from Publication_Study.vbb_study.equations.scalar_bessel import bessel_gauss_field


def run_modal_q20(data_dir, output_dir, *, pixel_pitch_m=5.5e-6, q=20,
                  z_positions_mm=None, m_max=8, comparison_data_dir=None,
                  comparison_z_mm=None):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = load_first_scan(data_dir)
    if not images:
        raise FileNotFoundError(f"No z*_*.bmg files found in {data_dir.resolve()}")
    if z_positions_mm is None:
        z_positions_mm = np.arange(len(images), dtype=float)
    z_positions_mm = np.asarray(z_positions_mm, float)
    if len(z_positions_mm) != len(images):
        raise ValueError("z_positions_mm must contain one value per BMG plane")

    kr, geometry = estimate_global_kr(images, pixel_pitch_m, q, .55)
    geometry.insert(1, "z_mm", z_positions_mm)
    geometry.to_csv(output_dir / "inner_ring_geometry.csv", index=False)

    fits, aux = [], []
    for zi, img in enumerate(images):
        fit, arrays = fit_plane(img, zi, pixel_pitch_m, q, kr, m_max=m_max,
                                rmax_um=220, n_r=44, n_theta=96)
        fits.append(fit); aux.append(arrays)

    rows = []
    for z_mm, f in zip(z_positions_mm, fits):
        row = dict(z_index=f.z_index, z_mm=z_mm, core_score=f.core_score,
                   ring_radius_um=f.ring_radius_px*pixel_pitch_m*1e6,
                   fit_corr=f.corr, fit_nrmse=f.nrmse,
                   azcv_model=f.azimuth_cv_before,
                   azcv_phase_corrected=f.azimuth_cv_after,
                   dark_model=f.dark_core_before,
                   dark_phase_corrected=f.dark_core_after)
        c0 = max(abs(f.coeffs[f.m_values == 0][0]), 1e-12)
        for m, c in zip(f.m_values, f.coeffs):
            row[f"cm{m:+d}_abs_rel"] = abs(c)/c0
            row[f"cm{m:+d}_phase"] = np.angle(c)
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "modal_fit_metrics.csv", index=False)

    good = (metrics.core_score > .55) & (metrics.fit_corr > .75)
    before = float(metrics.loc[good, "azcv_model"].mean()) if good.any() else np.nan
    after = float(metrics.loc[good, "azcv_phase_corrected"].mean()) if good.any() else np.nan
    summary = {
        "method": "Miao-style q=20 constrained Bessel modal retrieval",
        "interpretation": "model-internal phase-only prediction",
        "q": int(q), "planes": len(images), "trusted_planes": int(good.sum()),
        "kr_rad_per_um": float(kr*1e-6),
        "median_fit_corr": float(metrics.loc[good, "fit_corr"].median()) if good.any() else None,
        "mean_model_azcv_before": before if np.isfinite(before) else None,
        "mean_model_azcv_phase_only_after": after if np.isfinite(after) else None,
        "azcv_improvement_percent": float(100*(1-after/before)) if before > 0 else None,
        "hardware_ready": False,
        "hardware_blocker": "z-to-input-annulus plus SLM scale/rotation/parity/LUT calibration missing",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axs = plt.subplots(len(images), 3, figsize=(10, 2.45*len(images)),
                            constrained_layout=True, squeeze=False)
    for zi, a in enumerate(aux):
        for col, (arr, title) in enumerate(((a["measured"], "MEASURED"),
                                            (a["pred"], f"q={q} MODAL FIT"),
                                            (a["pred_corr"], "PHASE-ONLY MODEL"))):
            arr = arr/max(float(np.max(arr)), 1e-12)
            axs[zi, col].imshow(arr, origin="lower", aspect="auto", cmap="inferno",
                                vmin=0, vmax=1, extent=[0, 360,
                                a["radii_px"][0]*pixel_pitch_m*1e6,
                                a["radii_px"][-1]*pixel_pitch_m*1e6])
            axs[zi, col].set_title(f"z={z_positions_mm[zi]:g} mm {title}")
            axs[zi, col].set_xlabel("theta (deg)"); axs[zi, col].set_ylabel("r (um)")
    fig.savefig(output_dir / "polar_measured_fit_corrected.png", dpi=300)
    plt.close(fig)

    m = fits[0].m_values
    fig, ax = plt.subplots(figsize=(10, 5))
    for z_mm, f in zip(z_positions_mm, fits):
        c0 = max(abs(f.coeffs[f.m_values == 0][0]), 1e-12)
        ax.plot(m, np.abs(f.coeffs)/c0, "-o", alpha=.7, label=f"{z_mm:g} mm")
    ax.set_yscale("log"); ax.grid(True, alpha=.25)
    ax.set(xlabel="aberration harmonic m=n+q", ylabel="|c_m|/|c_0|",
           title=f"Retrieved q={q} modal aberration spectrum")
    ax.legend(ncol=3, fontsize=7)
    fig.savefig(output_dir / "modal_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    _write_profile_figures(output_dir, images, fits, aux, z_positions_mm, pixel_pitch_m,
                           q, kr,
                           prefix="realigned")
    cart = pd.read_csv(output_dir/"realigned_cartesian_comparison_metrics.csv")
    summary.update({
        "median_measured_vs_ideal_cartesian_corr": float(cart.measured_vs_ideal_corr.median()),
        "median_corrected_model_vs_ideal_cartesian_corr": float(cart.corrected_model_vs_ideal_corr.median()),
        "median_corrected_model_vs_ideal_cartesian_rmse": float(cart.corrected_model_vs_ideal_rmse.median()),
        "cartesian_interpretation": "Per-plane constrained-model result; not a single-mask propagation validation",
    })
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if comparison_data_dir is not None:
        comparison_images = load_first_scan(Path(comparison_data_dir))
        comparison_z_mm = np.asarray(comparison_z_mm, float)
        if len(comparison_images) != len(comparison_z_mm):
            raise ValueError("comparison_z_mm must contain one value per comparison plane")
        old_kr, _ = estimate_global_kr(comparison_images, pixel_pitch_m, q, .55)
        old_outputs = [fit_plane(im, i, pixel_pitch_m, q, old_kr, m_max=m_max,
                                 rmax_um=220, n_r=44, n_theta=96)
                       for i, im in enumerate(comparison_images)]
        old_fits = [item[0] for item in old_outputs]
        old_aux = [item[1] for item in old_outputs]
        _write_profile_figures(output_dir, comparison_images, old_fits, old_aux,
                               comparison_z_mm, pixel_pitch_m, q, old_kr,
                               prefix="pre_realign")
        _write_two_scan_propagation_figure(
            output_dir, fits, aux, z_positions_mm, old_fits, old_aux,
            comparison_z_mm, pixel_pitch_m)

    phases = np.stack([np.angle(a["g_theta"]) for a in aux])
    np.save(output_dir / "annular_aberration_phase.npy", phases)
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(phases, origin="lower", aspect="auto", cmap="twilight",
                   vmin=-np.pi, vmax=np.pi,
                   extent=[0, 360, z_positions_mm[0], z_positions_mm[-1]])
    ax.set(xlabel="theta (deg)", ylabel="physical z (mm)",
           title=f"q={q} constrained retrieved annular phase")
    fig.colorbar(im, ax=ax, label="wrapped phase (rad)")
    fig.savefig(output_dir / "annular_aberration_phase.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Visualization only: row interpolation maps measured z ordering to normalized radius.
    nrho = 512; ntheta = phases.shape[1]
    phase_r = np.vstack([np.interp(np.linspace(0, len(phases)-1, nrho),
                                   np.arange(len(phases)), phases[:, j])
                         for j in range(ntheta)]).T
    yy, xx = np.indices((512, 512)); rho = np.hypot(xx-255.5, yy-255.5)/255.5
    theta = np.mod(np.arctan2(yy-255.5, xx-255.5), 2*np.pi)
    ri = np.clip((rho*(nrho-1)).astype(int), 0, nrho-1)
    ti = np.mod((theta/(2*np.pi)*ntheta).astype(int), ntheta)
    correction = np.mod(-phase_r[ri, ti], 2*np.pi); correction[rho > 1] = np.nan
    np.save(output_dir / "UNCALIBRATED_DO_NOT_APPLY_q20_modal_correction.npy", correction)
    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(correction, cmap="twilight", vmin=0, vmax=2*np.pi)
    ax.set_title("UNCALIBRATED q=20 modal correction\nNORMALIZED COORDINATES — DO NOT APPLY")
    fig.colorbar(im, ax=ax, label="wrapped phase (rad)")
    fig.savefig(output_dir / "UNCALIBRATED_DO_NOT_APPLY_q20_modal_correction.png",
                dpi=400, bbox_inches="tight"); plt.close(fig)
    return metrics, geometry, summary


def _profile_arrays(fits, aux, pixel_pitch_m):
    radius_um = aux[0]["radii_px"]*pixel_pitch_m*1e6
    measured = np.stack([a["measured"].mean(axis=1) for a in aux])
    model = np.stack([a["pred"].mean(axis=1) for a in aux])
    corrected = np.stack([a["pred_corr"].mean(axis=1) for a in aux])
    for array in (measured, model, corrected):
        array /= np.maximum(array.max(axis=1, keepdims=True), 1e-12)
    ring_theta = []
    for f, a in zip(fits, aux):
        ir = int(np.argmin(np.abs(a["radii_px"]-f.ring_radius_px)))
        row = a["measured"][ir].astype(float)
        ring_theta.append(row/max(row.mean(), 1e-12))
    return radius_um, measured, model, corrected, np.stack(ring_theta)


def _write_profile_figures(output_dir, images, fits, aux, z_mm, pixel_pitch_m,
                           q, kr_m_inv, prefix):
    radius_um, measured, model, corrected, ring_theta = _profile_arrays(
        fits, aux, pixel_pitch_m)

    ncols = 3; nrows = int(np.ceil(len(fits)/ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 3*nrows),
                            constrained_layout=True, squeeze=False)
    for ax, z, ym, yf, yc in zip(axs.ravel(), z_mm, measured, model, corrected):
        ax.plot(radius_um, ym, color="black", lw=1.5, label="measured")
        ax.plot(radius_um, yf, color="tab:orange", label="q=20 fit")
        ax.plot(radius_um, yc, color="tab:blue", ls="--", label="phase-only model")
        ax.set(title=f"z={z:g} mm", xlabel="radius (um)", ylabel="normalized intensity",
               xlim=(0, radius_um[-1]), ylim=(0, 1.08))
        ax.grid(alpha=.2)
    for ax in axs.ravel()[len(fits):]: ax.axis("off")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"Radial profiles — {prefix.replace('_', ' ')} dataset", fontsize=15)
    fig.savefig(output_dir/f"{prefix}_radial_profiles_all_planes.png", dpi=300)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    extent = [radius_um[0], radius_um[-1], z_mm[0], z_mm[-1]]
    for ax, arr, title in zip(axs, (measured, model, corrected),
                              ("MEASURED", "q=20 MODAL FIT", "PHASE-ONLY MODEL")):
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap="inferno",
                       vmin=0, vmax=1, extent=extent)
        ax.set(title=title, xlabel="radius (um)", ylabel="z (mm)")
    fig.colorbar(im, ax=axs, label="plane-normalized intensity", shrink=.85)
    fig.suptitle(f"Axial radial propagation — {prefix.replace('_', ' ')}")
    fig.savefig(output_dir/f"{prefix}_axial_radial_profiles.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    im = ax.imshow(ring_theta, origin="lower", aspect="auto", cmap="magma",
                   extent=[0, 360, z_mm[0], z_mm[-1]])
    ax.set(title=f"Measured intensity around principal annulus — {prefix.replace('_', ' ')}",
           xlabel="theta (deg)", ylabel="z (mm)")
    fig.colorbar(im, ax=ax, label="intensity / azimuthal mean")
    fig.savefig(output_dir/f"{prefix}_ring_theta_z_profile.png", dpi=300)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axs[0, 0].plot(z_mm, [f.ring_radius_px*pixel_pitch_m*1e6 for f in fits], "o-")
    axs[0, 0].set(ylabel="principal ring radius (um)", title="Inner-annulus radius")
    axs[0, 1].plot(z_mm, [f.corr for f in fits], "o-")
    axs[0, 1].axhline(.75, color="red", ls="--", lw=1); axs[0, 1].set(ylabel="fit correlation", title="Measured/model fit")
    axs[1, 0].plot(z_mm, [f.azimuth_cv_before for f in fits], "o-", label="fit")
    axs[1, 0].plot(z_mm, [f.azimuth_cv_after for f in fits], "o-", label="phase-only model")
    axs[1, 0].set(ylabel="azimuthal CV", title="Annular non-uniformity"); axs[1, 0].legend()
    axs[1, 1].plot(z_mm, [f.dark_core_before for f in fits], "o-", label="fit")
    axs[1, 1].plot(z_mm, [f.dark_core_after for f in fits], "o-", label="phase-only model")
    axs[1, 1].set(ylabel="core/ring intensity", title="Dark-core preservation"); axs[1, 1].legend()
    for ax in axs.ravel(): ax.set_xlabel("z (mm)"); ax.grid(alpha=.25)
    fig.savefig(output_dir/f"{prefix}_profile_metrics_vs_z.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for order in range(1, 7):
        amplitude = []
        for f in fits:
            c0 = max(abs(f.coeffs[f.m_values == 0][0]), 1e-12)
            cp = abs(f.coeffs[f.m_values == order][0])
            cm = abs(f.coeffs[f.m_values == -order][0])
            amplitude.append((cp+cm)/(2*c0))
        ax.plot(z_mm, amplitude, "o-", label=f"A{order} (|c+|+|c-|)/2|c0|")
    ax.set(title=f"Modal profile versus propagation — {prefix.replace('_', ' ')}",
           xlabel="z (mm)", ylabel="relative modal amplitude")
    ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8)
    fig.savefig(output_dir/f"{prefix}_A1_A6_vs_z.png", dpi=300)
    plt.close(fig)

    _write_cartesian_figures(output_dir, images, fits, aux, z_mm, pixel_pitch_m,
                             q, kr_m_inv, prefix)


def _cartesian_stack(images, fits, aux, pixel_pitch_m, q, kr_m_inv,
                     limit_um=180.0, size=241):
    axis_um = np.linspace(-limit_um, limit_um, size)
    x_m = axis_um*1e-6
    X, Y = np.meshgrid(x_m, x_m); R = np.hypot(X, Y); PHI = np.arctan2(Y, X)
    basis, _ = modal_basis(q, fits[0].m_values, kr_m_inv, R.ravel(), PHI.ravel())
    # The validated project field; wide measured-window envelope avoids inventing a
    # tighter waist than the camera data supports.
    ideal_field = bessel_gauss_field(R, PHI, ell=q, kr_m_inv=kr_m_inv,
                                     waist_m=limit_um*1e-6)
    ideal = np.abs(ideal_field)**2
    ideal /= max(float(ideal.max()), 1e-12)
    measured, model, corrected = [], [], []
    for image, fit, arrays in zip(images, fits, aux):
        yy = fit.center_y_px + Y/pixel_pitch_m
        xx = fit.center_x_px + X/pixel_pitch_m
        m = ndimage.map_coordinates(image, [yy, xx], order=1,
                                    mode="constant", cval=0.0)
        f = np.abs(basis@fit.coeffs).reshape(size, size)**2
        c = np.abs(basis@arrays["coeffs_corr"]).reshape(size, size)**2
        measured.append(m/max(float(m.max()), 1e-12))
        model.append(f/max(float(f.max()), 1e-12))
        corrected.append(c/max(float(c.max()), 1e-12))
    return axis_um, np.stack(measured), np.stack(model), np.stack(corrected), ideal


def _write_cartesian_figures(output_dir, images, fits, aux, z_mm, pixel_pitch_m,
                             q, kr_m_inv, prefix):
    axis, measured, model, corrected, ideal = _cartesian_stack(
        images, fits, aux, pixel_pitch_m, q, kr_m_inv)
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    XX, YY = np.meshgrid(axis, axis); roi = np.hypot(XX, YY) <= 160
    def similarity(a, b):
        av, bv = a[roi], b[roi]
        return float(np.corrcoef(av, bv)[0, 1]), float(np.sqrt(np.mean((av-bv)**2)))
    comparison_rows = []
    for iz, z in enumerate(z_mm):
        measured_ideal_corr, measured_ideal_rmse = similarity(measured[iz], ideal)
        fit_measured_corr, fit_measured_rmse = similarity(model[iz], measured[iz])
        corrected_ideal_corr, corrected_ideal_rmse = similarity(corrected[iz], ideal)
        comparison_rows.append(dict(
            z_mm=z, measured_vs_ideal_corr=measured_ideal_corr,
            measured_vs_ideal_rmse=measured_ideal_rmse,
            modal_fit_vs_measured_corr=fit_measured_corr,
            modal_fit_vs_measured_rmse=fit_measured_rmse,
            corrected_model_vs_ideal_corr=corrected_ideal_corr,
            corrected_model_vs_ideal_rmse=corrected_ideal_rmse))
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir/f"{prefix}_cartesian_comparison_metrics.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axs[0].plot(z_mm, comparison.measured_vs_ideal_corr, "o-", label="measured vs ideal")
    axs[0].plot(z_mm, comparison.modal_fit_vs_measured_corr, "o-", label="modal fit vs measured")
    axs[0].plot(z_mm, comparison.corrected_model_vs_ideal_corr, "o-", label="corrected model vs ideal")
    axs[0].set(ylabel="Pearson correlation", ylim=(-.05, 1.05), title="Cartesian intensity correlation")
    axs[1].plot(z_mm, comparison.measured_vs_ideal_rmse, "o-", label="measured vs ideal")
    axs[1].plot(z_mm, comparison.modal_fit_vs_measured_rmse, "o-", label="modal fit vs measured")
    axs[1].plot(z_mm, comparison.corrected_model_vs_ideal_rmse, "o-", label="corrected model vs ideal")
    axs[1].set(ylabel="normalized RMSE", title="Cartesian intensity error")
    for ax in axs:
        ax.set_xlabel("z (mm)"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle(f"Measured / model / ideal agreement — {prefix.replace('_', ' ')}")
    fig.savefig(output_dir/f"{prefix}_cartesian_similarity_vs_z.png", dpi=300)
    plt.close(fig)

    # Full signed x-y slices. Four columns prevent the model fit being confused with
    # either measured data or the phase-only prediction.
    fig, axs = plt.subplots(len(images), 4, figsize=(13.5, 3.05*len(images)),
                            constrained_layout=True, squeeze=False)
    columns = (measured, model, corrected,
               np.broadcast_to(ideal, measured.shape))
    titles = ("MEASURED", "q=20 MODAL FIT",
              "PHASE-ONLY PER-PLANE MODEL", "IDEAL q=20 BESSEL–GAUSS")
    for iz, z in enumerate(z_mm):
        for col, (stack, title) in enumerate(zip(columns, titles)):
            im = axs[iz, col].imshow(stack[iz], origin="lower", extent=extent,
                                     cmap="inferno", vmin=0, vmax=1,
                                     interpolation="nearest")
            axs[iz, col].axhline(0, color="white", lw=.35, alpha=.45)
            axs[iz, col].axvline(0, color="white", lw=.35, alpha=.45)
            axs[iz, col].set(title=f"z={z:g} mm | {title}", xlabel="x (um)", ylabel="y (um)",
                             xlim=(axis[0], axis[-1]), ylim=(axis[0], axis[-1]))
            axs[iz, col].set_aspect("equal")
    fig.colorbar(im, ax=axs, label="plane-normalized intensity", shrink=.35, pad=.01)
    fig.suptitle(f"Signed Cartesian beam slices — {prefix.replace('_', ' ')}\n"
                 "corrected column is a per-plane modal prediction, not a measured post-SLM field",
                 fontsize=15)
    fig.savefig(output_dir/f"{prefix}_cartesian_xy_measured_fit_corrected_ideal.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Readable six-plane pages for normal viewing and print/export.
    for page, start in enumerate(range(0, len(images), 6), 1):
        stop = min(start+6, len(images)); count = stop-start
        fig, axs = plt.subplots(count, 4, figsize=(14.5, 3.25*count),
                                constrained_layout=True, squeeze=False)
        for local, iz in enumerate(range(start, stop)):
            for col, (stack, title) in enumerate(zip(columns, titles)):
                im = axs[local, col].imshow(stack[iz], origin="lower", extent=extent,
                                             cmap="inferno", vmin=0, vmax=1,
                                             interpolation="nearest")
                axs[local, col].axhline(0, color="white", lw=.35, alpha=.4)
                axs[local, col].axvline(0, color="white", lw=.35, alpha=.4)
                axs[local, col].set_title(title, fontsize=10)
                axs[local, col].text(.02, .97, f"z={z_mm[iz]:g} mm",
                                     transform=axs[local, col].transAxes, va="top",
                                     color="white", fontsize=9,
                                     bbox=dict(facecolor="black", alpha=.55, pad=2, edgecolor="none"))
                axs[local, col].set(xlabel="x (um)", ylabel="y (um)",
                                    xlim=(axis[0], axis[-1]), ylim=(axis[0], axis[-1]))
                axs[local, col].set_aspect("equal")
        fig.colorbar(im, ax=axs, label="plane-normalized intensity", shrink=.65, pad=.01)
        fig.suptitle(f"Signed Cartesian q={q} beam slices — {prefix.replace('_', ' ')} "
                     f"(page {page})\nper-plane model prediction; not post-SLM measurement", fontsize=14)
        fig.savefig(output_dir/f"{prefix}_cartesian_xy_page_{page}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Signed x-z/y-z sections sampled through the fitted dark-core centre.
    mid = len(axis)//2
    sections = ((measured[:, mid, :], corrected[:, mid, :], ideal[mid, :], "x", "y=0"),
                (measured[:, :, mid], corrected[:, :, mid], ideal[:, mid], "y", "x=0"))
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for row, (meas_sec, corr_sec, ideal_cut, coord, fixed) in enumerate(sections):
        ideal_sec = np.broadcast_to(ideal_cut, meas_sec.shape)
        for col, (arr, title) in enumerate(((meas_sec, "MEASURED"),
                                            (corr_sec, "PHASE-ONLY PER-PLANE MODEL"),
                                            (ideal_sec, "IDEAL q=20 BESSEL–GAUSS"))):
            im = axs[row, col].imshow(arr, origin="lower", aspect="auto", cmap="inferno",
                                      vmin=0, vmax=1,
                                      extent=[axis[0], axis[-1], z_mm[0], z_mm[-1]],
                                      interpolation="nearest")
            axs[row, col].axvline(0, color="cyan", lw=.6, alpha=.7)
            axs[row, col].set(title=f"{coord}–z {fixed} | {title}",
                              xlabel=f"{coord} (um)", ylabel="z (mm)")
    fig.colorbar(im, ax=axs, label="plane-normalized intensity", shrink=.8)
    fig.suptitle(f"Signed propagation sections — {prefix.replace('_', ' ')}")
    fig.savefig(output_dir/f"{prefix}_signed_xz_yz_measured_corrected_ideal.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Linear Cartesian centre cuts for every plane, preserving negative coordinates.
    fig, axs = plt.subplots(len(images), 2, figsize=(14, 2.5*len(images)),
                            constrained_layout=True, squeeze=False)
    for iz, z in enumerate(z_mm):
        for col, (mcut, ccut, icut, label) in enumerate((
                (measured[iz, mid, :], corrected[iz, mid, :], ideal[mid, :], "x cut (y=0)"),
                (measured[iz, :, mid], corrected[iz, :, mid], ideal[:, mid], "y cut (x=0)"))):
            ax = axs[iz, col]
            ax.plot(axis, mcut, color="black", lw=1.3, label="measured")
            ax.plot(axis, ccut, color="#0072B2", lw=1.4, label="phase-only per-plane model")
            ax.plot(axis, icut, color="#D55E00", lw=1.2, ls="--", label="ideal q=20 Bessel–Gauss")
            ax.axvline(0, color="0.6", lw=.6)
            ax.set(title=f"z={z:g} mm | {label}", xlabel=f"signed {label[0]} (um)",
                   ylabel="normalized intensity", xlim=(axis[0], axis[-1]), ylim=(0, 1.08))
            ax.grid(alpha=.2)
    axs[0, 0].legend(fontsize=8, ncol=3)
    fig.suptitle(f"Linear measured/corrected/ideal centre cuts — {prefix.replace('_', ' ')}",
                 fontsize=15)
    fig.savefig(output_dir/f"{prefix}_linear_xy_cuts_measured_corrected_ideal.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    for page, start in enumerate(range(0, len(images), 6), 1):
        stop = min(start+6, len(images)); count = stop-start
        fig, axs = plt.subplots(count, 2, figsize=(14, 2.75*count),
                                constrained_layout=True, squeeze=False)
        for local, iz in enumerate(range(start, stop)):
            for col, (mcut, ccut, icut, label) in enumerate((
                    (measured[iz, mid, :], corrected[iz, mid, :], ideal[mid, :], "x cut (y=0)"),
                    (measured[iz, :, mid], corrected[iz, :, mid], ideal[:, mid], "y cut (x=0)"))):
                ax = axs[local, col]
                ax.plot(axis, mcut, color="black", lw=1.5, label="measured")
                ax.plot(axis, ccut, color="#0072B2", lw=1.5, label="phase-only per-plane model")
                ax.plot(axis, icut, color="#D55E00", lw=1.3, ls="--", label="ideal q=20 Bessel–Gauss")
                ax.axvline(0, color="0.6", lw=.6)
                ax.set(title=f"z={z_mm[iz]:g} mm | {label}", xlabel=f"signed {label[0]} (um)",
                       ylabel="normalized intensity", xlim=(axis[0], axis[-1]), ylim=(0, 1.08))
                ax.grid(alpha=.2)
        axs[0, 0].legend(fontsize=8, ncol=3, loc="upper center")
        fig.suptitle(f"Measured / corrected-model / ideal linear cuts — "
                     f"{prefix.replace('_', ' ')} (page {page})", fontsize=14)
        fig.savefig(output_dir/f"{prefix}_linear_xy_cuts_page_{page}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    _write_3d_intensity_volumes(output_dir, axis, z_mm, measured, corrected, prefix)


def _write_3d_intensity_volumes(output_dir, axis_um, z_mm, measured, corrected, prefix):
    """Export matched measured/predicted 3D morphology volumes.

    Inputs are plane-normalized: fitted modal amplitudes have an arbitrary scale at
    each z, so this intentionally does not claim absolute axial power recovery.
    """
    # Moderate interpolation in z removes the blocky 18-slice appearance without
    # pretending that extra measurements were acquired. XY is downsampled for export.
    z_dense = np.linspace(float(z_mm[0]), float(z_mm[-1]), 2*(len(z_mm)-1)+1)
    volumes = []
    for stack in (measured, corrected):
        small = stack[:, ::4, ::4]
        dense = ndimage.zoom(small, (len(z_dense)/len(z_mm), 1, 1), order=1)
        dense = dense[:len(z_dense)]
        volumes.append(np.clip(dense, 0, 1).astype(np.float32))
    axis_small = axis_um[::4]
    np.savez_compressed(output_dir/f"{prefix}_3d_intensity_morphology_stack.npz",
                        x_um=axis_small, y_um=axis_small, z_mm=z_dense,
                        measured=volumes[0], corrected_per_plane_model=volumes[1],
                        normalization="each measured/model plane normalized independently")

    fig = plt.figure(figsize=(15, 7), constrained_layout=True)
    for panel, (volume, title) in enumerate(zip(volumes,
            ("MEASURED 3D INTENSITY MORPHOLOGY",
             "PHASE-ONLY CORRECTED PER-PLANE MODEL")), 1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        iz, iy, ix = np.where(volume >= .30)
        values = volume[iz, iy, ix]
        # Deterministic thinning keeps the 3D morphology responsive and printable.
        keep = np.arange(len(values))[::max(1, len(values)//18000)]
        ax.scatter(axis_small[ix[keep]], axis_small[iy[keep]], z_dense[iz[keep]],
                   c=values[keep], cmap="inferno", vmin=.30, vmax=1,
                   s=2.0, alpha=.30, linewidths=0, rasterized=True)
        ax.set(xlim=(axis_small[0], axis_small[-1]),
               ylim=(axis_small[0], axis_small[-1]), zlim=(z_dense[0], z_dense[-1]),
               xlabel="x (um)", ylabel="y (um)", zlabel="z (mm)", title=title)
        ax.set_box_aspect((1, 1, 2.3))
        ax.view_init(elev=23, azim=-58)
    fig.suptitle(f"3D q=20 intensity morphology — {prefix.replace('_', ' ')}\n"
                 "voxels I>=0.30; each z plane normalized; z visually compressed",
                 fontsize=14)
    fig.savefig(output_dir/f"{prefix}_3d_intensity_isosurfaces.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # Interactive physical-coordinate volume, following the repository's Plotly
    # Vortex_Bessel volume/isosurface convention.
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        x_show = axis_small[::2]; y_show = axis_small[::2]; z_show = z_dense[::2]
        X, Y = np.meshgrid(x_show, y_show, indexing="xy")
        fig = make_subplots(rows=1, cols=2, specs=[[{"type":"scene"},{"type":"scene"}]],
                            subplot_titles=("Measured", "Corrected per-plane model"))
        for col, volume in enumerate(volumes, 1):
            v = volume[::2, ::2, ::2]
            X3 = np.broadcast_to(X, v.shape); Y3 = np.broadcast_to(Y, v.shape)
            Z3 = np.broadcast_to(z_show[:, None, None], v.shape)
            fig.add_trace(go.Isosurface(x=X3.ravel(), y=Y3.ravel(), z=Z3.ravel(),
                value=v.ravel(), isomin=.18, isomax=.85, surface_count=6,
                opacity=.32, colorscale="Inferno", caps=dict(x_show=False,y_show=False,z_show=False),
                showscale=(col == 2), colorbar=dict(title="plane-normalized I")), row=1, col=col)
        scene = dict(xaxis_title="x (um)", yaxis_title="y (um)", zaxis_title="z (mm)",
                     aspectmode="manual", aspectratio=dict(x=1, y=1, z=2.3))
        fig.update_layout(title=f"{prefix.replace('_',' ')} q=20 3D intensity morphology — "
                                "corrected is per-plane model prediction",
                          scene=scene, scene2=scene, margin=dict(l=0,r=0,t=80,b=0))
        fig.write_html(output_dir/f"{prefix}_3d_intensity_interactive.html",
                       include_plotlyjs="cdn")
    except Exception as exc:
        (output_dir/f"{prefix}_3d_interactive_error.txt").write_text(str(exc), encoding="utf-8")


def _write_two_scan_propagation_figure(output_dir, new_fits, new_aux, new_z,
                                       old_fits, old_aux, old_z, pixel_pitch_m):
    radius, new_meas, _, _, _ = _profile_arrays(new_fits, new_aux, pixel_pitch_m)
    _, old_meas, _, _, _ = _profile_arrays(old_fits, old_aux, pixel_pitch_m)
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True, sharex=True)
    for ax, arr, z, title in ((axs[0], new_meas, new_z, "REALIGNED scan (measured)"),
                              (axs[1], old_meas, old_z, "PRE-REALIGN scan (measured)")):
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap="inferno", vmin=0, vmax=1,
                       extent=[radius[0], radius[-1], z[0], z[-1]])
        ax.set(title=title, xlabel="radius (um)", ylabel="z relative to old z0 (mm)")
    fig.colorbar(im, ax=axs, label="plane-normalized radial intensity")
    fig.suptitle("Available propagation regions — datasets kept separate at z=0")
    fig.savefig(output_dir/"two_alignment_available_propagation_regions.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    metrics, geometry, summary = run_modal_q20(here/"z-scan 2 1010",
        here/"outputs"/"modal_q20", z_positions_mm=np.arange(-17, 1),
        comparison_data_dir=here/"new z-scan bessel beam 1010",
        comparison_z_mm=np.arange(0, 16, 2))
    print(json.dumps(summary, indent=2))
