"""Test whether the retrieved phase error recreates the measured q=20 distortion.

This is a model falsification diagnostic: phase-only error is applied to the ideal
q=20 angular field and compared with measured camera intensity at every z plane.
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


def _normalize(array):
    return array/max(float(np.max(array)), 1e-12)


def _corr_rmse(a, b, mask):
    av, bv = a[mask], b[mask]
    return float(np.corrcoef(av, bv)[0, 1]), float(np.sqrt(np.mean((av-bv)**2)))


def run_phase_error_recreation(data_dir, output_dir, *, z_positions_mm,
                               pixel_pitch_m=5.5e-6, q=20, m_max=8,
                               limit_um=180, size=181):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    images = load_first_scan(Path(data_dir))
    z_positions_mm = np.asarray(z_positions_mm, float)
    if len(images) != len(z_positions_mm):
        raise ValueError("z_positions_mm must contain one value per plane")
    kr, _ = estimate_global_kr(images, pixel_pitch_m, q, .55)
    outputs = [fit_plane(im, i, pixel_pitch_m, q, kr, m_max=m_max,
                         rmax_um=220, n_r=44, n_theta=96)
               for i, im in enumerate(images)]
    fits = [item[0] for item in outputs]; aux = [item[1] for item in outputs]

    axis = np.linspace(-limit_um, limit_um, size); xm = axis*1e-6
    X, Y = np.meshgrid(xm, xm); R = np.hypot(X, Y); PHI = np.arctan2(Y, X)
    basis, _ = modal_basis(q, fits[0].m_values, kr, R.ravel(), PHI.ravel())
    ideal = np.abs(bessel_gauss_field(R, PHI, ell=q, kr_m_inv=kr,
                                      waist_m=limit_um*1e-6))**2
    ideal = _normalize(ideal)
    measured, recreated, full_fit = [], [], []
    recreation_coeffs = []
    for image, fit, arrays in zip(images, fits, aux):
        yy = fit.center_y_px+Y/pixel_pitch_m; xx = fit.center_x_px+X/pixel_pitch_m
        measured.append(_normalize(ndimage.map_coordinates(
            image, [yy, xx], order=1, mode="constant", cval=0.0)))
        full_fit.append(_normalize((np.abs(basis@fit.coeffs)**2).reshape(size, size)))
        # g(theta) is the recovered annular complex error. Keep only its phase so
        # this test cannot borrow the fitted amplitude imbalance.
        theta = arrays["theta_dense"]; phase_error = np.exp(1j*np.angle(arrays["g_theta"]))
        coeff = np.asarray([np.mean(phase_error*np.exp(1j*m*theta))
                            for m in fit.m_values], complex)
        recreation_coeffs.append(coeff)
        recreated.append(_normalize((np.abs(basis@coeff)**2).reshape(size, size)))
    measured, recreated, full_fit = map(np.stack, (measured, recreated, full_fit))
    ideal_stack = np.broadcast_to(ideal, measured.shape)
    np.savez_compressed(output_dir/"phase_error_recreation_stack.npz", x_um=axis,
                        z_mm=z_positions_mm, measured=measured,
                        ideal_with_retrieved_phase_error=recreated,
                        full_modal_fit=full_fit, ideal=ideal)

    XX, YY = np.meshgrid(axis, axis); roi = np.hypot(XX, YY) <= 160
    rows = []
    for iz, z in enumerate(z_positions_mm):
        mi_c, mi_e = _corr_rmse(measured[iz], ideal, roi)
        mr_c, mr_e = _corr_rmse(measured[iz], recreated[iz], roi)
        mf_c, mf_e = _corr_rmse(measured[iz], full_fit[iz], roi)
        rf_c, rf_e = _corr_rmse(recreated[iz], full_fit[iz], roi)
        rows.append(dict(z_mm=z, measured_vs_ideal_corr=mi_c,
                         measured_vs_phase_error_recreation_corr=mr_c,
                         measured_vs_full_modal_fit_corr=mf_c,
                         recreation_vs_full_fit_corr=rf_c,
                         measured_vs_ideal_rmse=mi_e,
                         measured_vs_phase_error_recreation_rmse=mr_e,
                         measured_vs_full_modal_fit_rmse=mf_e,
                         recreation_vs_full_fit_rmse=rf_e))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir/"phase_error_recreation_metrics.csv", index=False)
    summary = {
        "test": "ideal q=20 plus retrieved phase error versus measured lab stack",
        "phase_only_error": True, "amplitude_error_excluded": True,
        "planes": len(images), "q": q, "kr_rad_per_um": kr*1e-6,
        "median_measured_vs_ideal_corr": float(metrics.measured_vs_ideal_corr.median()),
        "median_measured_vs_phase_error_recreation_corr": float(metrics.measured_vs_phase_error_recreation_corr.median()),
        "median_measured_vs_full_modal_fit_corr": float(metrics.measured_vs_full_modal_fit_corr.median()),
        "median_recreation_vs_full_fit_corr": float(metrics.recreation_vs_full_fit_corr.median()),
        "correlation_gain_over_ideal": float(metrics.measured_vs_phase_error_recreation_corr.median()-metrics.measured_vs_ideal_corr.median()),
        "interpretation": "Support is strong only if phase-error recreation improves measured agreement materially and approaches the full phase+amplitude modal fit.",
    }
    (output_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axs[0].plot(z_positions_mm, metrics.measured_vs_ideal_corr, "o-", label="measured vs ideal")
    axs[0].plot(z_positions_mm, metrics.measured_vs_phase_error_recreation_corr, "o-", label="measured vs ideal+retrieved phase error")
    axs[0].plot(z_positions_mm, metrics.measured_vs_full_modal_fit_corr, "o-", label="measured vs full phase+amplitude fit")
    axs[0].set(title="Does recovered phase recreate the lab error?", ylabel="Cartesian correlation", ylim=(-.05,1.05))
    axs[1].plot(z_positions_mm, metrics.measured_vs_ideal_rmse, "o-", label="measured vs ideal")
    axs[1].plot(z_positions_mm, metrics.measured_vs_phase_error_recreation_rmse, "o-", label="measured vs ideal+retrieved phase error")
    axs[1].plot(z_positions_mm, metrics.measured_vs_full_modal_fit_rmse, "o-", label="measured vs full phase+amplitude fit")
    axs[1].set(title="Normalized intensity error", ylabel="normalized RMSE")
    for ax in axs: ax.set_xlabel("z (mm)"); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("Retrieved phase-error recreation test")
    fig.savefig(output_dir/"phase_error_recreation_agreement_vs_z.png", dpi=300)
    plt.close(fig)

    stacks = (measured, recreated, full_fit, ideal_stack)
    titles = ("LAB MEASURED", "IDEAL + RETRIEVED PHASE ERROR",
              "FULL MODAL FIT", "IDEAL q=20")
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    for page, start in enumerate(range(0, len(images), 6), 1):
        stop = min(start+6, len(images)); count=stop-start
        fig, axs = plt.subplots(count, 4, figsize=(14.5, 3.25*count),
                                constrained_layout=True, squeeze=False)
        for local, iz in enumerate(range(start, stop)):
            for col, (stack, title) in enumerate(zip(stacks, titles)):
                im=axs[local,col].imshow(stack[iz], origin="lower", extent=extent,
                    cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
                axs[local,col].axhline(0,color="white",lw=.35,alpha=.4)
                axs[local,col].axvline(0,color="white",lw=.35,alpha=.4)
                axs[local,col].set_title(title,fontsize=10)
                axs[local,col].text(.02,.97,f"z={z_positions_mm[iz]:g} mm",
                    transform=axs[local,col].transAxes,va="top",color="white",fontsize=9,
                    bbox=dict(facecolor="black",alpha=.55,pad=2,edgecolor="none"))
                axs[local,col].set(xlabel="x (um)",ylabel="y (um)")
                axs[local,col].set_aspect("equal")
        fig.colorbar(im,ax=axs,label="plane-normalized intensity",shrink=.65,pad=.01)
        fig.suptitle(f"Phase-error recreation versus laboratory data — page {page}\n"
                     "recreation uses retrieved phase only; no fitted amplitude imbalance",fontsize=14)
        fig.savefig(output_dir/f"phase_error_recreation_page_{page}.png",dpi=300,bbox_inches="tight")
        plt.close(fig)

    mid=len(axis)//2
    fig, axs=plt.subplots(2,4,figsize=(18,9),constrained_layout=True)
    for row, (selector, coordinate) in enumerate(((lambda a:a[:,mid,:],"x-z, y=0"),
                                                   (lambda a:a[:,:,mid],"y-z, x=0"))):
        for col,(stack,title) in enumerate(zip(stacks,titles)):
            arr=selector(stack)
            im=axs[row,col].imshow(arr,origin="lower",aspect="auto",cmap="inferno",vmin=0,vmax=1,
                extent=[axis[0],axis[-1],z_positions_mm[0],z_positions_mm[-1]])
            axs[row,col].set(title=f"{coordinate} | {title}",xlabel="signed transverse coordinate (um)",ylabel="z (mm)")
    fig.colorbar(im,ax=axs,label="plane-normalized intensity",shrink=.8)
    fig.suptitle("Signed propagation: does inverse correction recreate the measured distortion?")
    fig.savefig(output_dir/"phase_error_recreation_signed_xz_yz.png",dpi=300,bbox_inches="tight")
    plt.close(fig)
    return metrics, summary


if __name__ == "__main__":
    here=Path(__file__).resolve().parent
    metrics,summary=run_phase_error_recreation(here/"z-scan 2 1010",
        here/"outputs"/"slm_closed_loop_alignment"/"modal_q20"/"phase_error_recreation",
        z_positions_mm=np.arange(-17,1))
    print(json.dumps(summary,indent=2))
