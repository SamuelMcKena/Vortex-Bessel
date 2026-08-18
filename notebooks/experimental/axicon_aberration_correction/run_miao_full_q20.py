"""Run the full radial+angular q=20 Bessel phase retrieval on the measured z scan.

This is the canonical experimental inference path for the clean repository.

It follows the Miao et al. Bessel-specific retrieval structure:
  1. fit k_perp_opt and angular modal coefficients independently at every z;
  2. use k_perp_opt(z) to recover radial wavefront gradient and sampled annulus;
  3. integrate the radial gradient and combine it with angular coefficients to
     reconstruct ONE transverse residual complex field;
  4. propagate that one field through all z to test whether it explains the
     measured morphology;
  5. apply the transverse conjugate only as a model counterfactual.

There is no fitted longitudinal correction mask.  A real SLM correction remains
blocked until the camera/input-to-SLM transform and phase LUT are calibrated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

HERE = Path(__file__).resolve().parent
for ROOT in Path(__file__).resolve().parents:
    if (ROOT / "vbb_study").is_dir():
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        break

from miao_full_phase_retrieval import (
    EPS,
    _interp_complex_rows,
    assemble_cartesian_residual,
    plane_fits_to_table,
    reconstruct_transverse_wavefront,
    retrieve_variable_kperp_stack,
)
from modal_vortex_bessel import find_dark_core_center, load_first_scan
from vbb_study.equations.fields import make_xy_grid
from vbb_study.equations.propagation import make_bl_asm_propagator


THERMAL = "inferno"
PHASE = "twilight_shifted"


def _normalise(image: np.ndarray) -> np.ndarray:
    image = np.clip(np.asarray(image, float), 0.0, None)
    return image / max(float(image.max()), EPS)


def _sample_measured_morphology(images, axis_um, pixel_pitch_m):
    """Sample every plane about its measured dark core for shape-only comparison.

    This deliberately does not claim absolute pointing/coma information.  The
    full-sensor trajectory analysis is the appropriate source for beam-axis drift.
    """

    axis_um = np.asarray(axis_um, float)
    X_um, Y_um = np.meshgrid(axis_um, axis_um, indexing="xy")
    out = []
    for image in images:
        cy, cx, _ = find_dark_core_center(image)
        yy = cy + Y_um * 1e-6 / float(pixel_pitch_m)
        xx = cx + X_um * 1e-6 / float(pixel_pitch_m)
        sampled = ndimage.map_coordinates(
            np.asarray(image, float), [yy, xx], order=1, mode="constant", cval=0.0
        )
        out.append(_normalise(sampled))
    return np.stack(out)


def _sample_model(field, grid, axis_um):
    coord = np.asarray(axis_um, float) * 1e-6 / float(grid["dx"]) + (int(grid["N"]) - 1) / 2
    xx, yy = np.meshgrid(coord, coord, indexing="xy")
    sampled = ndimage.map_coordinates(
        np.abs(field) ** 2, [yy, xx], order=1, mode="constant", cval=0.0
    )
    return _normalise(sampled)


def _corr_rmse(a, b, mask):
    av = np.asarray(a, float)[mask]
    bv = np.asarray(b, float)[mask]
    corr = 0.0 if np.std(av) <= EPS or np.std(bv) <= EPS else float(np.corrcoef(av, bv)[0, 1])
    rmse = float(np.sqrt(np.mean((av-bv)**2)))
    return corr, rmse


def _fit_absolute_z_end_from_ideal(
    measured,
    z_relative_mm,
    prop_ideal,
    grid,
    axis_um,
    roi,
    *,
    search_min_mm=18.0,
    search_max_mm=35.0,
):
    """Fit one nuisance axial offset with the ideal model only.

    No retrieved aberration enters this registration, so the error model cannot
    tune its own axial coordinate to improve agreement.
    """

    anchors = np.unique(np.asarray([0, len(measured)//2, len(measured)-1], int))
    rows = []

    def evaluate(end_mm):
        cors = []
        rmses = []
        for idx in anchors:
            z_abs = float(end_mm) + float(z_relative_mm[idx])
            if z_abs <= 0:
                return
            model = _sample_model(prop_ideal(z_abs*1e-3), grid, axis_um)
            corr, rmse = _corr_rmse(measured[idx], model, roi)
            cors.append(corr); rmses.append(rmse)
        rows.append({
            "absolute_z_at_relative_zero_mm": float(end_mm),
            "mean_anchor_corr": float(np.mean(cors)),
            "mean_anchor_rmse": float(np.mean(rmses)),
        })

    lower = max(float(search_min_mm), -float(np.min(z_relative_mm)) + 0.5)
    for value in np.arange(lower, float(search_max_mm)+1e-9, 1.0):
        evaluate(value)
    coarse = pd.DataFrame(rows)
    if coarse.empty:
        raise RuntimeError("absolute-z registration search has no valid candidate")
    best = float(coarse.loc[coarse.mean_anchor_corr.idxmax(), "absolute_z_at_relative_zero_mm"])
    for value in np.arange(max(lower, best-1.0), min(float(search_max_mm), best+1.0)+1e-9, 0.1):
        if not np.any(np.isclose(coarse.absolute_z_at_relative_zero_mm, value)):
            evaluate(value)
    table = pd.DataFrame(rows).sort_values("absolute_z_at_relative_zero_mm")
    best = float(table.loc[table.mean_anchor_corr.idxmax(), "absolute_z_at_relative_zero_mm"])
    return best, table


def _make_model_inputs(wavefront, grid, ideal_input):
    complex_residual = _interp_complex_rows(
        wavefront.residual_complex_rows,
        wavefront.rho_m,
        grid["R"],
        grid["PHI"],
    )
    valid = (grid["R"] >= wavefront.rho_m[0]) & (grid["R"] <= wavefront.rho_m[-1])
    phase = np.zeros_like(grid["R"], float)
    amplitude = np.ones_like(grid["R"], float)
    phase[valid] = np.angle(complex_residual[valid])
    amplitude[valid] = np.abs(complex_residual[valid])
    if np.any(valid):
        amplitude[valid] /= max(float(np.median(amplitude[valid])), EPS)
    # Do not permit isolated near-zero modal amplitudes to explode a plotting/model
    # normalization.  This clipping is diagnostic and is recorded in provenance.
    amplitude = np.clip(amplitude, 0.0, 3.0)

    phase_error_input = ideal_input * np.exp(1j*phase)
    complex_error_input = ideal_input * amplitude * np.exp(1j*phase)
    phase_corrected_complex_input = ideal_input * amplitude
    return phase, amplitude, valid, phase_error_input, complex_error_input, phase_corrected_complex_input


def _propagate_model_stack(field, grid, wavelength_m, z_absolute_mm, axis_um):
    prop = make_bl_asm_propagator(
        field, grid, float(wavelength_m), bandlimit=True, include_evanescent=False
    )
    return np.stack([
        _sample_model(prop(float(z)*1e-3), grid, axis_um)
        for z in z_absolute_mm
    ])


def _metrics(measured, ideal, phase_error, complex_error, corrected, z_mm, holdout, roi):
    holdout = set(map(int, holdout))
    rows = []
    for i, z in enumerate(z_mm):
        values = {}
        for name, stack in (
            ("ideal", ideal),
            ("phase_error", phase_error),
            ("complex_error", complex_error),
            ("phase_conjugated_model", corrected),
        ):
            corr, rmse = _corr_rmse(measured[i], stack[i], roi)
            values[f"measured_vs_{name}_corr"] = corr
            values[f"measured_vs_{name}_rmse"] = rmse
        rows.append({
            "plane_index": i,
            "z_relative_mm": float(z),
            "split": "holdout" if i in holdout else "train",
            **values,
        })
    return pd.DataFrame(rows)


def _gate(metrics):
    hold = metrics[metrics.split == "holdout"]
    if hold.empty:
        hold = metrics
    phase_corr_gain = float(
        hold.measured_vs_phase_error_corr.median() - hold.measured_vs_ideal_corr.median()
    )
    phase_rmse_gain = float(
        hold.measured_vs_ideal_rmse.median() - hold.measured_vs_phase_error_rmse.median()
    )
    complex_corr_gain = float(
        hold.measured_vs_complex_error_corr.median() - hold.measured_vs_ideal_corr.median()
    )
    complex_rmse_gain = float(
        hold.measured_vs_ideal_rmse.median() - hold.measured_vs_complex_error_rmse.median()
    )
    return {
        "heldout_phase_error_corr_gain_over_ideal": phase_corr_gain,
        "heldout_phase_error_rmse_reduction": phase_rmse_gain,
        "heldout_complex_error_corr_gain_over_ideal": complex_corr_gain,
        "heldout_complex_error_rmse_reduction": complex_rmse_gain,
        "supports_phase_residual_as_cause": bool(phase_corr_gain >= 0.03 and phase_rmse_gain > 0),
        "supports_full_complex_residual_as_cause": bool(complex_corr_gain >= 0.03 and complex_rmse_gain > 0),
    }


def _sections(stack, half_width=2):
    cy = stack.shape[1]//2; cx = stack.shape[2]//2; h = int(half_width)
    xz = stack[:, max(0,cy-h):cy+h+1, :].mean(axis=1)
    yz = stack[:, :, max(0,cx-h):cx+h+1].mean(axis=2)
    return xz, yz


def _save_retrieval_figures(output_dir, fits_table, wavefront, cart):
    output_dir = Path(output_dir)
    rho_mm = wavefront.rho_m*1e3
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    axes[0,0].plot(fits_table.z_relative_mm, fits_table.k_perp_opt_rad_per_um, "o-")
    axes[0,0].set(title="Per-plane optimum transverse wavenumber",
                  xlabel="relative z (mm)", ylabel=r"$k_{\perp}^{opt}$ (rad/µm)")
    axes[0,1].plot(rho_mm, wavefront.radial_gradient_rad_per_m*1e-3, "o-")
    axes[0,1].axhline(0, color="0.5", lw=.8)
    axes[0,1].set(title="Recovered radial phase gradient",
                  xlabel="sampled input radius ρ (mm)", ylabel=r"$d\psi_\rho/d\rho$ (rad/mm)")
    axes[1,0].plot(rho_mm, wavefront.radial_phase_rad, "o-")
    axes[1,0].set(title="Integrated radial residual phase",
                  xlabel="sampled input radius ρ (mm)", ylabel=r"$\psi_\rho$ (rad)")
    axes[1,1].plot(fits_table.z_relative_mm, fits_table.correlation, "o-", label="fit correlation")
    axes[1,1].plot(fits_table.z_relative_mm, fits_table.nrmse, "o-", label="NRMSE")
    axes[1,1].set(title="Per-plane modal retrieval quality", xlabel="relative z (mm)")
    axes[1,1].legend()
    for ax in axes.ravel(): ax.grid(alpha=.2)
    fig.suptitle(
        "Miao-style q=20 retrieval: z-dependent k⊥ recovers the radial part of ONE transverse wavefront\n"
        "m=±1 excluded unless the optical axis is calibrated",
        fontsize=14,
    )
    p1 = output_dir / "01_miao_kperp_and_radial_phase.png"
    fig.savefig(p1, dpi=350, bbox_inches="tight"); plt.close(fig)

    x_mm = cart.x_m*1e3
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), constrained_layout=True)
    rows_phase = wavefront.residual_phase_rows_rad
    im = axes[0].imshow(rows_phase, origin="lower", aspect="auto", cmap=PHASE,
                        vmin=-np.pi, vmax=np.pi,
                        extent=[0,360,rho_mm[0],rho_mm[-1]])
    axes[0].set(title="Residual phase on retrieved annuli", xlabel="azimuth θ (deg)", ylabel="ρ (mm)")
    fig.colorbar(im, ax=axes[0], shrink=.78)
    for ax, arr, title, cmap, lo, hi in (
        (axes[1], cart.residual_phase_rad, "Full transverse residual phase", PHASE, -np.pi, np.pi),
        (axes[2], cart.residual_amplitude, "Retrieved angular amplitude modulation", "viridis", 0, 2),
        (axes[3], cart.correction_phase_rad, "Phase-only conjugate candidate", PHASE, -np.pi, np.pi),
    ):
        im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=lo, vmax=hi,
                       extent=[x_mm[0],x_mm[-1],x_mm[0],x_mm[-1]])
        ax.set_aspect("equal")
        ax.set(title=title, xlabel="input-plane x (mm)", ylabel="input-plane y (mm)")
        fig.colorbar(im, ax=ax, shrink=.78)
    fig.suptitle(
        "Retrieved q=20 residual wavefront — programmed qθ removed; radial phase comes from k⊥(z)\n"
        "model inference only; global piston gauge-fixed; hardware mapping not calibrated",
        fontsize=14,
    )
    p2 = output_dir / "02_miao_full_transverse_residual.png"
    fig.savefig(p2, dpi=350, bbox_inches="tight"); plt.close(fig)
    return p1, p2


def _save_forward_figure(output_dir, measured, ideal, phase_error, complex_error, corrected,
                         z_rel_mm, axis_um):
    stacks = (measured, ideal, phase_error, complex_error, corrected)
    titles = (
        "LAB MEASURED\n(core-centred morphology)",
        "NOMINAL MODEL",
        "NOMINAL + RETRIEVED PHASE",
        "NOMINAL + RETRIEVED COMPLEX MODULATION",
        "MODEL AFTER PHASE CONJUGATE\n(amplitude residual remains)",
    )
    fig, axes = plt.subplots(2, 5, figsize=(23, 8.8), constrained_layout=True, sharey=True)
    for col, (stack, title) in enumerate(zip(stacks, titles)):
        xz, yz = _sections(stack)
        for row, (section, label) in enumerate(((xz,"x-z"),(yz,"y-z"))):
            im = axes[row,col].imshow(
                section, origin="lower", aspect="auto", cmap=THERMAL, vmin=0, vmax=1,
                extent=[axis_um[0],axis_um[-1],z_rel_mm[0],z_rel_mm[-1]],
                interpolation="nearest",
            )
            axes[row,col].axvline(0,color="cyan",lw=.5,alpha=.55)
            axes[row,col].set(title=f"{label} | {title}", xlabel="signed transverse coordinate (µm)",
                              ylabel="relative z (mm)")
    fig.colorbar(im, ax=axes, label="plane-normalized intensity", shrink=.78)
    fig.suptitle(
        "One transverse residual field -> all z by propagation\n"
        "no independently fitted longitudinal correction; rightmost column is MODEL closure, not post-SLM camera data",
        fontsize=14,
    )
    path = Path(output_dir) / "03_miao_single_transverse_field_xz_yz.png"
    fig.savefig(path, dpi=350, bbox_inches="tight"); plt.close(fig)
    return path


def _save_metrics(output_dir, metrics, holdout):
    fig, axes = plt.subplots(1,2,figsize=(13,4.8),constrained_layout=True)
    z = metrics.z_relative_mm
    for name,label in (("ideal","nominal"),("phase_error","+ retrieved phase"),("complex_error","+ retrieved complex residual")):
        axes[0].plot(z,metrics[f"measured_vs_{name}_corr"],"o-",label=label)
        axes[1].plot(z,metrics[f"measured_vs_{name}_rmse"],"o-",label=label)
    for ax in axes:
        for idx in holdout:
            ax.axvline(float(z.iloc[int(idx)]),color="0.7",lw=.7,ls=":")
        ax.grid(alpha=.2); ax.set_xlabel("relative z (mm)"); ax.legend(fontsize=8)
    axes[0].set(title="Measured morphology agreement",ylabel="correlation",ylim=(-.05,1.05))
    axes[1].set(title="Measured morphology error",ylabel="normalized RMSE")
    fig.suptitle("Held-out z planes are dotted; their annular fits are excluded from the validation wavefront")
    path=Path(output_dir)/"04_miao_heldout_forward_metrics.png"
    fig.savefig(path,dpi=350,bbox_inches="tight"); plt.close(fig)
    return path


def run_miao_full_q20(
    data_dir,
    output_dir,
    *,
    z_relative_mm=None,
    wavelength_m=1030e-9,
    pixel_pitch_m=5.5e-6,
    q=20,
    nominal_k_tan_alpha_m_inv=None,
    absolute_z_at_relative_zero_mm=None,
    search_fraction=0.12,
    m_max_search=4,
    m_max_final=8,
    include_first_order=False,
    model_grid_n=1024,
    model_dx_m=5.5e-6,
    beam_radius_m=2.0e-3,
    view_limit_um=180.0,
    view_size=181,
):
    data_dir=Path(data_dir); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    images=load_first_scan(data_dir)
    if not images: raise FileNotFoundError(f"No BMG z scan found in {data_dir}")
    if z_relative_mm is None: z_relative_mm=np.arange(-(len(images)-1),1,dtype=float)
    z_relative_mm=np.asarray(z_relative_mm,float)
    if len(z_relative_mm)!=len(images): raise ValueError("z_relative_mm must contain one value per measured plane")

    fits=retrieve_variable_kperp_stack(
        images,z_relative_mm*1e-3,pixel_pitch_m=pixel_pitch_m,q=q,
        search_fraction=search_fraction,m_max_search=m_max_search,m_max_final=m_max_final,
        include_first_order=include_first_order,
    )
    fit_table=plane_fits_to_table(fits); fit_table.to_csv(output_dir/"miao_per_plane_variable_kperp.csv",index=False)

    holdout=np.arange(2,len(images),4,dtype=int)
    train=np.asarray([i for i in range(len(images)) if i not in set(holdout)],int)
    nominal_k=(float(nominal_k_tan_alpha_m_inv) if nominal_k_tan_alpha_m_inv is not None
               else float(np.median([fits[i].k_perp_opt_m_inv for i in train])))

    axis_um=np.linspace(-view_limit_um,view_limit_um,view_size)
    measured=_sample_measured_morphology(images,axis_um,pixel_pitch_m)
    Xv,Yv=np.meshgrid(axis_um,axis_um,indexing="xy"); roi=np.hypot(Xv,Yv)<=160.0
    grid=make_xy_grid(model_grid_n,model_dx_m)
    aperture=np.exp(-(grid["R"]/beam_radius_m)**2)
    ideal_input=aperture*np.exp(1j*(q*grid["PHI"]-nominal_k*grid["R"]))
    prop_ideal=make_bl_asm_propagator(ideal_input,grid,wavelength_m,bandlimit=True,include_evanescent=False)

    if absolute_z_at_relative_zero_mm is None:
        absolute_end_mm,z_registration=_fit_absolute_z_end_from_ideal(
            measured,z_relative_mm,prop_ideal,grid,axis_um,roi)
        z_mapping_source="ideal-only morphology registration; model coordinate, not hardware calibration"
    else:
        absolute_end_mm=float(absolute_z_at_relative_zero_mm)
        z_registration=pd.DataFrame([{"absolute_z_at_relative_zero_mm":absolute_end_mm,
                                     "mean_anchor_corr":np.nan,"mean_anchor_rmse":np.nan}])
        z_mapping_source="externally supplied absolute distance"
    z_registration.to_csv(output_dir/"ideal_only_absolute_z_registration.csv",index=False)
    z_absolute_mm=absolute_end_mm+z_relative_mm

    # Validation wavefront excludes interleaved hold-out annuli.
    train_fits=[fits[i] for i in train]
    wf_train=reconstruct_transverse_wavefront(
        train_fits,q=q,z_at_relative_zero_from_reference_m=absolute_end_mm*1e-3,
        nominal_k_tan_alpha_m_inv=nominal_k,n_theta=720,
        include_first_order=include_first_order,twin_ambiguity_resolved=False)
    phase,amp,valid,phase_error_input,complex_error_input,corrected_input=_make_model_inputs(wf_train,grid,ideal_input)

    ideal_stack=_propagate_model_stack(ideal_input,grid,wavelength_m,z_absolute_mm,axis_um)
    phase_error_stack=_propagate_model_stack(phase_error_input,grid,wavelength_m,z_absolute_mm,axis_um)
    complex_error_stack=_propagate_model_stack(complex_error_input,grid,wavelength_m,z_absolute_mm,axis_um)
    corrected_stack=_propagate_model_stack(corrected_input,grid,wavelength_m,z_absolute_mm,axis_um)
    metrics=_metrics(measured,ideal_stack,phase_error_stack,complex_error_stack,corrected_stack,
                     z_relative_mm,holdout,roi)
    metrics.to_csv(output_dir/"miao_forward_metrics.csv",index=False)
    gate=_gate(metrics)

    # Final presentation/diagnostic map uses all measured annuli, after the held-out
    # validation has been evaluated with the training-only wavefront.
    wf_full=reconstruct_transverse_wavefront(
        fits,q=q,z_at_relative_zero_from_reference_m=absolute_end_mm*1e-3,
        nominal_k_tan_alpha_m_inv=nominal_k,n_theta=720,
        include_first_order=include_first_order,twin_ambiguity_resolved=False)
    cart=assemble_cartesian_residual(wf_full,grid_size=700)
    p1,p2=_save_retrieval_figures(output_dir,fit_table,wf_full,cart)
    p3=_save_forward_figure(output_dir,measured,ideal_stack,phase_error_stack,complex_error_stack,
                            corrected_stack,z_relative_mm,axis_um)
    p4=_save_metrics(output_dir,metrics,holdout)

    np.savez_compressed(
        output_dir/"miao_full_q20_retrieval.npz",
        z_relative_mm=z_relative_mm,z_absolute_model_mm=z_absolute_mm,
        k_perp_opt_m_inv=np.asarray([f.k_perp_opt_m_inv for f in fits]),
        rho_m=wf_full.rho_m,radial_gradient_rad_per_m=wf_full.radial_gradient_rad_per_m,
        radial_phase_rad=wf_full.radial_phase_rad,theta_rad=wf_full.theta_rad,
        residual_complex_rows=wf_full.residual_complex_rows.astype(np.complex64),
        cartesian_x_m=cart.x_m,cartesian_y_m=cart.y_m,
        cartesian_residual_phase_rad=cart.residual_phase_rad.astype(np.float32),
        cartesian_residual_amplitude=cart.residual_amplitude.astype(np.float32),
        measured=measured.astype(np.float32),ideal=ideal_stack.astype(np.float32),
        phase_error_model=phase_error_stack.astype(np.float32),
        complex_error_model=complex_error_stack.astype(np.float32),
        phase_conjugated_model=corrected_stack.astype(np.float32),
        heldout_indices=holdout,
    )

    summary={
        "method":"full Miao-style q20 Bessel phase-front retrieval",
        "reference":"Miao et al., Optics Express 30, 11360-11371 (2022), DOI 10.1364/OE.454796",
        "planes":len(images),"q":int(q),"heldout_indices":holdout.tolist(),
        "retrieves_k_perp_per_plane":True,
        "radial_phase_from_kperp_gradient":True,
        "angular_phase_from_modal_coefficients":True,
        "single_transverse_correction_phase":True,
        "independent_longitudinal_correction_phase":False,
        "programmed_qtheta_in_residual":False,
        "m_plus_minus_1_retrieved":bool(include_first_order),
        "pointing_coma_interpretation":("enabled only because caller explicitly requested m=+-1" if include_first_order
                                        else "blocked: morphology-centred scan does not calibrate optical-axis position"),
        "nominal_k_tan_alpha_rad_per_um":nominal_k*1e-6,
        "nominal_cone_source":("external calibration" if nominal_k_tan_alpha_m_inv is not None
                               else "median training-plane k_perp; absolute cone/radial slope not calibrated"),
        "absolute_z_at_relative_zero_model_mm":absolute_end_mm,
        "z_mapping_source":z_mapping_source,
        "twin_ambiguity_resolved":False,
        "global_piston_observable":False,
        **gate,
        "phase_conjugated_model_interpretation":"model-only phase cancellation; residual amplitude remains",
        "hardware_ready":False,
        "hardware_blocker":"camera/input-to-SLM magnification, rotation/parity, footprint and 1030-nm phase LUT; then new post-correction z scan",
        "figures":[str(p1),str(p2),str(p3),str(p4)],
    }
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return metrics,summary


def _parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,default=Path(os.environ.get("BESSEL_ZSCAN_DATA_DIR",HERE/"z-scan 2 1010")))
    p.add_argument("--output-dir",type=Path,default=HERE/"outputs"/"miao_full_q20")
    p.add_argument("--nominal-kperp-rad-per-um",type=float,default=None,
                   help="calibrated nominal k*tan(alpha) in rad/um; omit for relative median reference")
    p.add_argument("--absolute-z-at-relative-zero-mm",type=float,default=None,
                   help="distance from input/axicon reference to z_rel=0; omit for ideal-only model registration")
    p.add_argument("--include-first-order",action="store_true",
                   help="enable m=+-1 only if optical-axis positions are calibrated")
    return p


def main(argv=None):
    a=_parser().parse_args(argv)
    nominal=None if a.nominal_kperp_rad_per_um is None else a.nominal_kperp_rad_per_um*1e6
    _,summary=run_miao_full_q20(
        a.data_dir,a.output_dir,nominal_k_tan_alpha_m_inv=nominal,
        absolute_z_at_relative_zero_mm=a.absolute_z_at_relative_zero_mm,
        include_first_order=a.include_first_order)
    print(json.dumps(summary,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
