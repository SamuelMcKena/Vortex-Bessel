"""All-z metrics and profiles for correction and inverse-error closure tests.

This report deliberately keeps one common analytic q=20 vortex--Bessel ideal.
It separates broad radial agreement from the non-axisymmetric/angular structure
that actually determines whether the recovered error was recreated correctly.
"""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from modal_vortex_bessel import modal_basis, corrected_coefficients_phase_only


EPS = 1e-12


def _normalise(a):
    a = np.asarray(a, float)
    return a / max(float(np.max(a)), EPS)


def _corr_rmse(a, b, mask):
    av, bv = np.asarray(a)[mask], np.asarray(b)[mask]
    return (float(np.corrcoef(av, bv)[0, 1]),
            float(np.sqrt(np.mean((av - bv) ** 2))))


def _coefficients_from_row(row, m_values):
    return np.asarray([
        float(row[f"cm{m:+d}_abs_rel"]) * np.exp(1j * float(row[f"cm{m:+d}_phase"]))
        for m in m_values
    ], complex)


def _radial_residual(image, radius_um, bin_width_um=2.0):
    bins = np.floor(radius_um / bin_width_um).astype(int)
    means = (np.bincount(bins.ravel(), weights=image.ravel()) /
             np.maximum(np.bincount(bins.ravel()), 1))
    return image - means[bins]


def _radial_profile(image, radius_um, max_radius_um=160.0, bin_width_um=2.0):
    bins = np.floor(radius_um / bin_width_um).astype(int)
    count = np.bincount(bins.ravel())
    means = np.bincount(bins.ravel(), weights=image.ravel()) / np.maximum(count, 1)
    n = int(max_radius_um / bin_width_um)
    return (np.arange(n) + .5) * bin_width_um, means[:n]


def _sample_ring(image, axis_um, radii_um=np.linspace(36, 55, 12), n_theta=720):
    theta = np.linspace(-180.0, 180.0, n_theta, endpoint=False)
    theta_rad = np.deg2rad(theta)
    rr, tt = np.meshgrid(radii_um, theta_rad, indexing="ij")
    pixel_um = float(axis_um[1] - axis_um[0])
    centre = (len(axis_um) - 1) / 2
    yy = centre + rr * np.sin(tt) / pixel_um
    xx = centre + rr * np.cos(tt) / pixel_um
    profile = ndimage.map_coordinates(image, [yy, xx], order=1).mean(axis=0)
    return theta, profile


def _dark_ratio(image, radius_um):
    core = radius_um < 20.0
    ring = (radius_um >= 36.0) & (radius_um <= 55.0)
    return float(np.mean(image[core]) / max(float(np.mean(image[ring])), EPS))


def _dense_morphology_volume(stack, axis_um, z_mm, xy_stride=3):
    """Return a smooth display volume without inventing extra measured planes."""
    axis_small = np.asarray(axis_um)[::xy_stride]
    small = np.asarray(stack)[:, ::xy_stride, ::xy_stride]
    z_dense = np.linspace(float(z_mm[0]), float(z_mm[-1]), 2*(len(z_mm)-1)+1)
    dense = ndimage.zoom(small, (len(z_dense)/len(z_mm), 1, 1), order=1)
    return axis_small, z_dense, np.clip(dense[:len(z_dense)], 0, 1)


def _mesh_coordinates(volume, axis_um, z_mm, level):
    vertices, faces, _, _ = marching_cubes(volume, level=level)
    x = np.interp(vertices[:, 2], np.arange(volume.shape[2]), axis_um)
    y = np.interp(vertices[:, 1], np.arange(volume.shape[1]), axis_um)
    z = np.interp(vertices[:, 0], np.arange(volume.shape[0]), z_mm)
    return np.column_stack((x, y, z)), faces


def _write_measured_corrected_3d_mesh(output_dir, axis_um, z_mm,
                                      measured, corrected):
    """Export true measured/corrected iso-intensity meshes and rotatable HTML."""
    volumes = []
    for stack in (measured, corrected):
        axis_small, z_dense, volume = _dense_morphology_volume(
            stack, axis_um, z_mm)
        volumes.append(volume)
    levels = (.22, .48)
    colours = (plt.cm.inferno(.42), plt.cm.inferno(.82))
    fig = plt.figure(figsize=(16, 7.5), constrained_layout=True)
    for panel, (volume, title) in enumerate(zip(volumes,
            ("LAB MEASURED BEAM PATH",
             "PHASE-CORRECTED PER-PLANE MODEL")), 1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        for level, colour, alpha in zip(levels, colours, (.24, .52)):
            vertices, faces = _mesh_coordinates(volume, axis_small, z_dense, level)
            # Thin only polygon display density; the extracted surface coordinates
            # remain tied to the full 61x61x35 display volume.
            face_step = max(1, len(faces)//55000)
            mesh = Poly3DCollection(vertices[faces[::face_step]], linewidths=.03,
                                    edgecolor=(0, 0, 0, .10), alpha=alpha)
            mesh.set_facecolor(colour)
            ax.add_collection3d(mesh)
        ax.set(xlim=(axis_small[0], axis_small[-1]),
               ylim=(axis_small[0], axis_small[-1]),
               zlim=(z_dense[0], z_dense[-1]),
               xlabel="x (um)", ylabel="y (um)", zlabel="relative z (mm)",
               title=title)
        ax.set_box_aspect((1, 1, 2.35))
        ax.view_init(elev=22, azim=-58)
    fig.suptitle("Measured versus predicted corrected q=20 beam path — true iso-intensity meshes\n"
                 "surfaces I=0.22 and 0.48; each z plane normalized independently",
                 fontsize=14)
    fig.savefig(output_dir/"measured_vs_corrected_3d_mesh.png", dpi=400,
                bbox_inches="tight")
    plt.close(fig)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        interactive = make_subplots(rows=1, cols=2,
            specs=[[{"type":"scene"}, {"type":"scene"}]],
            subplot_titles=("Lab measured", "Corrected per-plane model"))
        for col, volume in enumerate(volumes, 1):
            for level, colour, opacity in zip(levels,
                    ("rgb(163,54,85)", "rgb(252,191,54)"), (.20, .48)):
                vertices, faces = _mesh_coordinates(volume, axis_small, z_dense, level)
                interactive.add_trace(go.Mesh3d(
                    x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color=colour, opacity=opacity, flatshading=False,
                    name=f"I={level:.2f}", showlegend=(col == 1)), row=1, col=col)
        scene = dict(xaxis_title="x (um)", yaxis_title="y (um)",
                     zaxis_title="relative z (mm)", aspectmode="manual",
                     aspectratio=dict(x=1, y=1, z=2.35))
        interactive.update_layout(
            title="Measured versus corrected q=20 3D beam path — rotatable mesh",
            scene=scene, scene2=scene, margin=dict(l=0, r=0, t=80, b=0))
        interactive.write_html(output_dir/"measured_vs_corrected_3d_mesh_interactive.html",
                               include_plotlyjs="cdn")
    except Exception as exc:
        (output_dir/"measured_vs_corrected_3d_mesh_interactive_error.txt").write_text(
            str(exc), encoding="utf-8")


def run_comprehensive_error_validation(modal_dir, output_dir, *, q=20,
                                       kr_m_inv=489678.1594027835):
    modal_dir, output_dir = Path(modal_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recreation = np.load(modal_dir / "phase_error_recreation" /
                         "phase_error_recreation_stack.npz")
    axis = recreation["x_um"]
    z_mm = recreation["z_mm"]
    measured = np.stack([_normalise(a) for a in recreation["measured"]])
    full_fit = np.stack([_normalise(a) for a in recreation["full_modal_fit"]])
    error_sim = np.stack([_normalise(a) for a in
                          recreation["ideal_with_retrieved_phase_error"]])
    ideal = _normalise(recreation["ideal"])

    metric_source = pd.read_csv(modal_dir / "modal_fit_metrics.csv")
    m_values = np.arange(-8, 9, dtype=int)
    x_m = axis * 1e-6
    X_m, Y_m = np.meshgrid(x_m, x_m, indexing="xy")
    R_m = np.hypot(X_m, Y_m)
    PHI = np.arctan2(Y_m, X_m)
    R_um = R_m * 1e6
    basis, _ = modal_basis(q, m_values, kr_m_inv, R_m.ravel(), PHI.ravel())
    corrected = []
    for z in z_mm:
        row = metric_source.iloc[int(np.argmin(np.abs(metric_source.z_mm - z)))]
        coeffs = _coefficients_from_row(row, m_values)
        corrected_coeffs, _, _ = corrected_coefficients_phase_only(coeffs, m_values)
        field = (np.abs(basis @ corrected_coeffs) ** 2).reshape(len(axis), len(axis))
        corrected.append(_normalise(field))
    corrected = np.stack(corrected)
    ideal_stack = np.broadcast_to(ideal, measured.shape)

    stacks = {
        "measured": measured,
        "full_modal_fit": full_fit,
        "corrected_model": corrected,
        "ideal": ideal_stack,
        "ideal_plus_inverse_error": error_sim,
    }
    roi = R_um <= 160.0
    annulus = (R_um >= 32.0) & (R_um <= 62.0)
    structure_roi = (R_um >= 25.0) & (R_um <= 160.0)
    rows = []
    ring_profiles = {key: [] for key in stacks}
    radial_profiles = {key: [] for key in stacks}
    for iz, z in enumerate(z_mm):
        for key, stack in stacks.items():
            _, ring = _sample_ring(stack[iz], axis)
            ring_profiles[key].append(ring)
            _, radial = _radial_profile(stack[iz], R_um)
            radial_profiles[key].append(radial)
        mi_c, mi_e = _corr_rmse(measured[iz], ideal, roi)
        fm_c, fm_e = _corr_rmse(full_fit[iz], measured[iz], roi)
        ci_c, ci_e = _corr_rmse(corrected[iz], ideal, roi)
        em_c, em_e = _corr_rmse(error_sim[iz], measured[iz], roi)
        ef_c, ef_e = _corr_rmse(error_sim[iz], full_fit[iz], roi)
        em_ann_c, em_ann_e = _corr_rmse(error_sim[iz], measured[iz], annulus)
        measured_residual = _radial_residual(measured[iz], R_um)
        error_residual = _radial_residual(error_sim[iz], R_um)
        structure_c, structure_e = _corr_rmse(
            error_residual, measured_residual, structure_roi)
        ring_meas = ring_profiles["measured"][-1]
        ring_error = ring_profiles["ideal_plus_inverse_error"][-1]
        ring_corr = float(np.corrcoef(ring_meas, ring_error)[0, 1])
        ring_rmse = float(np.sqrt(np.mean(
            (_normalise(ring_meas) - _normalise(ring_error)) ** 2)))
        radial_meas = radial_profiles["measured"][-1]
        radial_error = radial_profiles["ideal_plus_inverse_error"][-1]
        radial_corr = float(np.corrcoef(radial_meas, radial_error)[0, 1])
        row = {
            "z_mm": float(z),
            "measured_vs_ideal_corr": mi_c, "measured_vs_ideal_rmse": mi_e,
            "full_fit_vs_measured_corr": fm_c, "full_fit_vs_measured_rmse": fm_e,
            "corrected_vs_ideal_corr": ci_c, "corrected_vs_ideal_rmse": ci_e,
            "error_sim_vs_measured_corr": em_c, "error_sim_vs_measured_rmse": em_e,
            "error_sim_vs_full_fit_corr": ef_c, "error_sim_vs_full_fit_rmse": ef_e,
            "error_sim_vs_measured_annulus_corr": em_ann_c,
            "error_sim_vs_measured_annulus_rmse": em_ann_e,
            "error_sim_vs_measured_nonaxisymmetric_corr": structure_c,
            "error_sim_vs_measured_nonaxisymmetric_rmse": structure_e,
            "error_sim_vs_measured_ring_angular_corr": ring_corr,
            "error_sim_vs_measured_ring_angular_rmse": ring_rmse,
            "error_sim_vs_measured_radial_profile_corr": radial_corr,
        }
        for key, stack in stacks.items():
            ring = ring_profiles[key][-1]
            row[f"{key}_ring_cv"] = float(np.std(ring) / max(float(np.mean(ring)), EPS))
            row[f"{key}_dark_core_ratio"] = _dark_ratio(stack[iz], R_um)
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "all_z_correction_error_metrics.csv", index=False)
    ring_profiles = {key: np.stack(values) for key, values in ring_profiles.items()}
    radial_profiles = {key: np.stack(values) for key, values in radial_profiles.items()}
    np.savez_compressed(output_dir / "all_z_correction_error_stacks_profiles.npz",
                       x_um=axis, z_mm=z_mm, **stacks,
                       theta_deg=_sample_ring(ideal, axis)[0],
                       **{f"ring_{k}": v for k, v in ring_profiles.items()},
                       radial_radius_um=_radial_profile(ideal, R_um)[0],
                       **{f"radial_{k}": v for k, v in radial_profiles.items()})

    # Metrics dashboard: broad image metrics and the error-sensitive diagnostics.
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    axes[0, 0].plot(z_mm, metrics.measured_vs_ideal_corr, "o-", label="measured vs ideal")
    axes[0, 0].plot(z_mm, metrics.full_fit_vs_measured_corr, "o-", label="modal fit vs measured")
    axes[0, 0].plot(z_mm, metrics.corrected_vs_ideal_corr, "o-", label="corrected vs ideal")
    axes[0, 0].plot(z_mm, metrics.error_sim_vs_measured_corr, "o-", label="error sim vs measured")
    axes[0, 0].plot(z_mm, metrics.error_sim_vs_full_fit_corr, "o-", label="error sim vs modal fit")
    axes[0, 0].set(title="Full Cartesian intensity", ylabel="correlation", ylim=(-1.05, 1.05))
    axes[0, 1].plot(z_mm, metrics.measured_vs_ideal_rmse, "o-", label="measured vs ideal")
    axes[0, 1].plot(z_mm, metrics.full_fit_vs_measured_rmse, "o-", label="modal fit vs measured")
    axes[0, 1].plot(z_mm, metrics.corrected_vs_ideal_rmse, "o-", label="corrected vs ideal")
    axes[0, 1].plot(z_mm, metrics.error_sim_vs_measured_rmse, "o-", label="error sim vs measured")
    axes[0, 1].set(title="Full Cartesian intensity", ylabel="normalized RMSE")
    axes[0, 2].plot(z_mm, metrics.error_sim_vs_measured_annulus_corr, "o-", label="main annulus")
    axes[0, 2].plot(z_mm, metrics.error_sim_vs_measured_nonaxisymmetric_corr, "o-", label="non-axisymmetric residual")
    axes[0, 2].plot(z_mm, metrics.error_sim_vs_measured_ring_angular_corr, "o-", label="ring angular profile")
    axes[0, 2].plot(z_mm, metrics.error_sim_vs_measured_radial_profile_corr, "o-", label="radial profile")
    axes[0, 2].axhline(0, color="black", lw=.7)
    axes[0, 2].set(title="Does inverse phase recreate the lab error?", ylabel="correlation", ylim=(-1.05, 1.05))
    for key, label in (("measured", "measured"), ("full_modal_fit", "modal fit"),
                       ("corrected_model", "corrected"), ("ideal", "ideal"),
                       ("ideal_plus_inverse_error", "error simulation")):
        axes[1, 0].plot(z_mm, metrics[f"{key}_ring_cv"], "o-", label=label)
        axes[1, 1].plot(z_mm, metrics[f"{key}_dark_core_ratio"], "o-", label=label)
    axes[1, 0].set(title="Principal-ring azimuthal non-uniformity", ylabel="ring CV")
    axes[1, 1].set(title="Dark vortex-core leakage", ylabel="core mean / ring mean")
    axes[1, 2].plot(z_mm, metrics.error_sim_vs_measured_ring_angular_rmse, "o-",
                    label="ring angular error")
    axes[1, 2].plot(z_mm, metrics.error_sim_vs_measured_nonaxisymmetric_rmse, "o-",
                    label="non-axisymmetric image error")
    axes[1, 2].set(title="Error-sensitive disagreement", ylabel="RMSE")
    for ax in axes.ravel():
        ax.set_xlabel("relative z (mm)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.suptitle("All-z correction and inverse-error validation — normal analytic q=20 ideal")
    fig.savefig(output_dir / "all_z_metrics_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Five-column image pages retain the distinction between data and predictions.
    column_keys = ("measured", "full_modal_fit", "corrected_model", "ideal",
                   "ideal_plus_inverse_error")
    column_titles = ("LAB MEASURED", "FULL MODAL FIT", "MEASURED + CORRECTION\nMODEL",
                     "NORMAL IDEAL q=20", "IDEAL + INVERSE\nERROR PHASE")
    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    for page, start in enumerate(range(0, len(z_mm), 6), 1):
        stop = min(start + 6, len(z_mm))
        fig, axes = plt.subplots(stop-start, 5, figsize=(18, 3.15*(stop-start)),
                                 constrained_layout=True, squeeze=False)
        for row_i, iz in enumerate(range(start, stop)):
            for col, (key, title) in enumerate(zip(column_keys, column_titles)):
                shown = axes[row_i, col].imshow(stacks[key][iz], origin="lower",
                    extent=extent, cmap="inferno", vmin=0, vmax=1,
                    interpolation="nearest")
                axes[row_i, col].axhline(0, color="white", lw=.3, alpha=.45)
                axes[row_i, col].axvline(0, color="white", lw=.3, alpha=.45)
                axes[row_i, col].set(title=title, xlabel="x (um)", ylabel="y (um)")
                axes[row_i, col].text(.02, .97, f"z={z_mm[iz]:g} mm",
                    transform=axes[row_i, col].transAxes, va="top", color="white",
                    fontsize=8, bbox=dict(facecolor="black", alpha=.55,
                                          edgecolor="none", pad=2))
                axes[row_i, col].set_aspect("equal")
        fig.colorbar(shown, ax=axes, label="plane-normalized intensity", shrink=.7)
        fig.suptitle(f"Correction and error-recreation slices — page {page}")
        fig.savefig(output_dir / f"all_z_images_page_{page}.png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)

    # Signed x/y centre-line comparisons for every measured plane.
    mid = len(axis) // 2
    colours = ("black", "#999999", "#0072B2", "#D55E00", "#CC79A7")
    line_labels = ("measured", "full modal fit", "measured + correction model",
                   "normal ideal", "ideal + inverse error")
    for page, start in enumerate(range(0, len(z_mm), 6), 1):
        stop = min(start + 6, len(z_mm))
        fig, axes = plt.subplots(stop-start, 2, figsize=(14.5, 2.8*(stop-start)),
                                 constrained_layout=True, squeeze=False)
        for row_i, iz in enumerate(range(start, stop)):
            for col, (selector, direction) in enumerate((
                    (lambda a: a[mid, :], "x cut at y=0"),
                    (lambda a: a[:, mid], "y cut at x=0"))):
                ax = axes[row_i, col]
                for key, label, colour in zip(column_keys, line_labels, colours):
                    style = "--" if key == "ideal" else "-"
                    ax.plot(axis, selector(stacks[key][iz]), style, color=colour,
                            lw=1.35, label=label)
                ax.axvline(0, color="0.6", lw=.6)
                ax.set(title=f"z={z_mm[iz]:g} mm | {direction}",
                       xlabel="signed coordinate (um)", ylabel="normalized intensity",
                       xlim=(axis[0], axis[-1]), ylim=(-.02, 1.05))
                ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"All correction/error 1D Cartesian cross-sections — page {page}")
        fig.savefig(output_dir / f"all_z_signed_xy_cross_sections_page_{page}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Angular profiles expose error orientation hidden by Cartesian/radial metrics.
    theta = _sample_ring(ideal, axis)[0]
    for page, start in enumerate(range(0, len(z_mm), 6), 1):
        stop = min(start + 6, len(z_mm))
        fig, axes = plt.subplots(3, 2, figsize=(14, 10.5), constrained_layout=True)
        for ax, iz in zip(axes.ravel(), range(start, stop)):
            for key, label, colour in zip(column_keys, line_labels, colours):
                ax.plot(theta, _normalise(ring_profiles[key][iz]), color=colour,
                        lw=1.25, ls="--" if key == "ideal" else "-", label=label)
            ax.set(title=f"z={z_mm[iz]:g} mm", xlabel="ring angle (deg)",
                   ylabel="normalized annular intensity", xlim=(-180, 180), ylim=(-.02, 1.05))
            ax.grid(alpha=.2)
        for ax in axes.ravel()[stop-start:]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"Principal-ring angular profiles — page {page}")
        fig.savefig(output_dir / f"all_z_ring_angular_profiles_page_{page}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Full signed propagation maps: neither negative transverse half is cropped.
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), constrained_layout=True)
    for row, (selector, section_name) in enumerate((
            (lambda a: a[:, mid, :], "x-z at y=0"),
            (lambda a: a[:, :, mid], "y-z at x=0"))):
        for col, (key, title) in enumerate(zip(column_keys, column_titles)):
            shown = axes[row, col].imshow(selector(stacks[key]), origin="lower",
                aspect="auto", cmap="inferno", vmin=0, vmax=1,
                extent=[axis[0], axis[-1], z_mm[0], z_mm[-1]],
                interpolation="nearest")
            axes[row, col].axvline(0, color="cyan", lw=.55, alpha=.65)
            axes[row, col].set(title=f"{section_name} | {title}",
                               xlabel="signed transverse coordinate (um)",
                               ylabel="relative z (mm)")
    fig.colorbar(shown, ax=axes, label="plane-normalized intensity", shrink=.82)
    fig.suptitle("Full signed measured/corrected/ideal/error-simulation propagation maps")
    fig.savefig(output_dir/"all_z_full_signed_xz_yz_maps.png", dpi=400,
                bbox_inches="tight")
    plt.close(fig)

    # Radial propagation maps and readable per-plane radial line profiles.
    radius_axis = _radial_profile(ideal, R_um)[0]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8), constrained_layout=True)
    for ax, key, title in zip(axes, column_keys, column_titles):
        shown = ax.imshow(radial_profiles[key], origin="lower", aspect="auto",
                          cmap="inferno", vmin=0, vmax=1,
                          extent=[radius_axis[0], radius_axis[-1],
                                  z_mm[0], z_mm[-1]], interpolation="nearest")
        ax.set(title=title, xlabel="radius (um)", ylabel="relative z (mm)")
    fig.colorbar(shown, ax=axes, label="azimuthally averaged normalized intensity",
                 shrink=.8)
    fig.suptitle("All-z radial intensity profiles")
    fig.savefig(output_dir/"all_z_radial_profile_maps.png", dpi=400,
                bbox_inches="tight")
    plt.close(fig)

    for page, start in enumerate(range(0, len(z_mm), 6), 1):
        stop = min(start + 6, len(z_mm))
        fig, axes = plt.subplots(3, 2, figsize=(14, 10.5), constrained_layout=True)
        for ax, iz in zip(axes.ravel(), range(start, stop)):
            for key, label, colour in zip(column_keys, line_labels, colours):
                ax.plot(radius_axis, _normalise(radial_profiles[key][iz]),
                        color=colour, lw=1.35,
                        ls="--" if key == "ideal" else "-", label=label)
            ax.set(title=f"z={z_mm[iz]:g} mm", xlabel="radius (um)",
                   ylabel="normalized radial intensity",
                   xlim=(radius_axis[0], radius_axis[-1]), ylim=(-.02, 1.05))
            ax.grid(alpha=.2)
        for ax in axes.ravel()[stop-start:]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"All correction/error radial profiles — page {page}")
        fig.savefig(output_dir/f"all_z_radial_profiles_page_{page}.png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)

    _write_measured_corrected_3d_mesh(
        output_dir, axis, z_mm, measured, corrected)

    summary = {
        "planes": int(len(z_mm)), "q": int(q), "kr_rad_per_um": kr_m_inv * 1e-6,
        "median_corrected_vs_ideal_corr": float(metrics.corrected_vs_ideal_corr.median()),
        "median_error_sim_vs_measured_global_corr": float(metrics.error_sim_vs_measured_corr.median()),
        "median_error_sim_vs_full_fit_corr": float(metrics.error_sim_vs_full_fit_corr.median()),
        "median_error_sim_vs_measured_nonaxisymmetric_corr": float(
            metrics.error_sim_vs_measured_nonaxisymmetric_corr.median()),
        "median_error_sim_vs_measured_ring_angular_corr": float(
            metrics.error_sim_vs_measured_ring_angular_corr.median()),
        "median_measured_ring_cv": float(metrics.measured_ring_cv.median()),
        "median_full_modal_fit_ring_cv": float(metrics.full_modal_fit_ring_cv.median()),
        "median_corrected_model_ring_cv": float(metrics.corrected_model_ring_cv.median()),
        "median_ideal_ring_cv": float(metrics.ideal_ring_cv.median()),
        "median_error_simulation_ring_cv": float(
            metrics.ideal_plus_inverse_error_ring_cv.median()),
        "three_dimensional_normalization": "Each measured/model z plane normalized independently; morphology only, not absolute axial power.",
        "interpretation": "Global/radial agreement must not be treated as error recreation unless non-axisymmetric and ring-angular metrics also agree.",
        "correction_scope": "Per-plane modal prediction, not post-SLM camera data.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics, summary


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    modal = here / "outputs" / "slm_closed_loop_alignment" / "modal_q20"
    _, result = run_comprehensive_error_validation(
        modal, modal / "comprehensive_error_validation")
    print(json.dumps(result, indent=2))
