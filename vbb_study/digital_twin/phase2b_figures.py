"""Publication figures for PHASE 2B native and beam-volume diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.phase2b_visual_cases import (
    PHASE2B_HEX_EARLY_M,
    PHASE2B_HEX_LATE_M,
    PHASE2B_HEX_REFERENCE_M,
    Phase2BCaseResult,
    Phase2BHexPackage,
)


EPS = np.finfo(float).eps


def _mpl() -> tuple[Any, Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.collections import PolyCollection

    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.2,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 8.4,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt, colors, PolyCollection


def _save(
    fig: Any,
    stem: Path,
    *,
    dpi: int = 360,
) -> tuple[Path, Path]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf, dpi=dpi)
    return png, pdf


def _record(
    figure_id: str,
    paths: tuple[Path, Path],
    *,
    cases: Sequence[str],
    source: str,
    native_grid_n: int | str,
    native_dx_m: float | str,
    render_method: str,
    normalisation: str,
    crop_rule: str,
    interpolation: str,
    paired_rule_id: str = "",
    render_downsampling: str = "none",
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "png_path": str(paths[0]),
        "pdf_path": str(paths[1]),
        "case_ids": ";".join(cases),
        "source": source,
        "native_grid_n": native_grid_n,
        "native_dx_m": native_dx_m,
        "metrics_computed_on_native_arrays": True,
        "render_method": render_method,
        "normalisation_policy": normalisation,
        "crop_rule": crop_rule,
        "display_interpolation": interpolation,
        "display_interpolation_used_for_metrics": False,
        "paired_colour_crop_rule_id": paired_rule_id,
        "render_downsampling": render_downsampling,
    }


def _normalise_peak(array: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(array, dtype=float), 0.0)
    return arr / max(float(np.max(arr)), EPS)


def _equal_power(array: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(array, dtype=float), 0.0)
    return arr / max(float(np.sum(arr)), EPS)


def _native_axis(result: Phase2BCaseResult) -> np.ndarray:
    n = int(result.native_grid_n)
    return (np.arange(n, dtype=float) - n // 2) * float(result.native_dx_m)


def _crop(array: np.ndarray, x_m: np.ndarray, halfwidth_m: float) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(np.abs(x_m) <= float(halfwidth_m))
    if indices.size < 4:
        raise ValueError("plot crop contains too few samples")
    sl = slice(int(indices[0]), int(indices[-1]) + 1)
    return np.asarray(array)[sl, sl], x_m[sl]


def _sas_crop(sas: Mapping[str, Any], halfwidth_m: float) -> tuple[np.ndarray, np.ndarray]:
    grid = sas["grid"]
    x = np.asarray(grid["x"], dtype=float)
    return _crop(np.asarray(sas["intensity"], dtype=float), x, halfwidth_m)


def plot_case_xy_bundle(result: Phase2BCaseResult, root: Path) -> dict[str, Any]:
    """Plot readable native landmarks plus a physical SAS z60 close-up."""

    plt, _, _ = _mpl()
    available = sorted(float(value) for value in result.selected_planes)
    preferred = (0.0, 20e-3, 60e-3, 120e-3, 200e-3)
    targets = tuple(
        available[int(np.argmin(np.abs(np.asarray(available, dtype=float) - value)))]
        for value in preferred
    )
    targets = tuple(dict.fromkeys(targets))
    if len(targets) < 5:
        targets = tuple(available[:5])
    x = _native_axis(result)
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 8.0), constrained_layout=True)
    image = None
    for ax, z_m in zip(axes.ravel()[:5], targets):
        plane, xc = _crop(result.selected_planes[z_m], x, result.focus_halfwidth_m)
        extent = [xc[0] / 1e-3, xc[-1] / 1e-3, xc[0] / 1e-3, xc[-1] / 1e-3]
        image = ax.imshow(
            _normalise_peak(plane),
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="bicubic",
        )
        ax.set_title(f"native z = {z_m / 1e-3:.0f} mm")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
    hero_ax = axes.ravel()[5]
    if result.sas_hero is not None:
        half = min(result.focus_halfwidth_m, 0.76e-3)
        plane, xc = _sas_crop(result.sas_hero, half)
        extent = [xc[0] / 1e-3, xc[-1] / 1e-3, xc[0] / 1e-3, xc[-1] / 1e-3]
        hero_ax.imshow(
            _normalise_peak(plane),
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="lanczos",
        )
        hero_ax.set_title(f"SAS focus z = 60 mm\n{float(result.sas_hero['output_dx_m']) / 1e-6:.2f} um output sampling")
        hero_ax.set_xlabel("x (mm)")
        hero_ax.set_ylabel("y (mm)")
        hero_ax.set_aspect("equal")
    else:
        hero_ax.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes[:, :2], label="plane-peak-normalised intensity", shrink=0.78)
    fig.suptitle(f"{result.case_id}: {result.family} | transverse evolution", fontsize=15)
    paths = _save(fig, root / "02_xy_planes" / f"{result.case_id.lower()}_xy_landmarks_and_sas_focus")
    plt.close(fig)
    return _record(
        f"{result.case_id}_xy_bundle",
        paths,
        cases=(result.case_id,),
        source=str(result.metadata["source_contract"]),
        native_grid_n=result.native_grid_n,
        native_dx_m=result.native_dx_m,
        render_method="native fixed-grid landmarks plus SAS z60 physical resampling",
        normalisation="independent plane peak for morphology; fixed 0..1 colour scale",
        crop_rule=f"native focus halfwidth {result.focus_halfwidth_m / 1e-3:.4g} mm; SAS <=0.76 mm",
        interpolation="bicubic native; lanczos SAS display only",
    )


def plot_case_slices(result: Phase2BCaseResult, root: Path) -> dict[str, Any]:
    """Plot native x-z/y-z centre slices and conserved plane power."""

    plt, colors, _ = _mpl()
    x = _native_axis(result)
    indices = np.flatnonzero(np.abs(x) <= result.focus_halfwidth_m)
    sl = slice(int(indices[0]), int(indices[-1]) + 1)
    extent = [
        x[sl][0] / 1e-3,
        x[sl][-1] / 1e-3,
        result.z_values_m[-1] / 1e-3,
        result.z_values_m[0] / 1e-3,
    ]
    common = max(float(np.max(result.xz_map[:, sl])), float(np.max(result.yz_map[:, sl])), EPS)
    norm = colors.PowerNorm(gamma=0.45, vmin=0.0, vmax=1.0)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), constrained_layout=True)
    image = axes[0].imshow(
        np.asarray(result.xz_map[:, sl], dtype=float) / common,
        origin="upper",
        aspect="auto",
        extent=extent,
        cmap="magma",
        norm=norm,
        interpolation="bicubic",
    )
    axes[1].imshow(
        np.asarray(result.yz_map[:, sl], dtype=float) / common,
        origin="upper",
        aspect="auto",
        extent=extent,
        cmap="magma",
        norm=norm,
        interpolation="bicubic",
    )
    axes[0].set(title="x-z centre slice", xlabel="x (mm)", ylabel="z (mm)")
    axes[1].set(title="y-z centre slice", xlabel="y (mm)", ylabel="z (mm)")
    power = np.asarray(result.power_by_z, dtype=float)
    power_deviation_ppb = (power / max(float(power[0]), EPS) - 1.0) * 1.0e9
    axes[2].plot(result.z_values_m / 1e-3, power_deviation_ppb, color="#0072B2", lw=1.8)
    axes[2].axhline(0.0, color="0.45", lw=0.8, ls="--")
    axes[2].set(title="native integrated plane power", xlabel="z (mm)", ylabel="fractional deviation (ppb)")
    axes[2].grid(alpha=0.24)
    fig.colorbar(image, ax=axes[:2], label="common-global-peak normalised intensity", shrink=0.82)
    fig.suptitle(f"{result.case_id}: propagation volume centre slices", fontsize=14.5)
    paths = _save(fig, root / "03_xz_yz_slices" / f"{result.case_id.lower()}_xz_yz_and_power")
    plt.close(fig)
    return _record(
        f"{result.case_id}_xz_yz",
        paths,
        cases=(result.case_id,),
        source=str(result.metadata["source_contract"]),
        native_grid_n=result.native_grid_n,
        native_dx_m=result.native_dx_m,
        render_method="native centre samples over z",
        normalisation="one common global maximum across x-z and y-z; native integrated power",
        crop_rule=f"matched +/-{result.focus_halfwidth_m / 1e-3:.4g} mm",
        interpolation="bicubic display only",
        paired_rule_id=f"{result.case_id}_xz_yz_common",
    )


def plot_case_profiles(result: Phase2BCaseResult, root: Path) -> dict[str, Any]:
    """Plot native radial, angular, centre-line, and axial profiles."""

    plt, _, _ = _mpl()
    plane = np.asarray(result.selected_planes[PHASE2B_HEX_REFERENCE_M], dtype=float)
    n = result.native_grid_n
    x = _native_axis(result)
    mid = n // 2
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    radial = np.asarray(result.radial_intensity, dtype=float)
    axes[0, 0].plot(result.radial_radius_m / 1e-3, radial / max(float(np.max(radial)), EPS), color="#0072B2")
    axes[0, 0].axvline(result.ring_radius_m / 1e-3, color="#D55E00", ls="--", lw=1.0)
    axes[0, 0].set(title="native radial profile at z=60 mm", xlabel="radius (mm)", ylabel="normalised intensity")
    angular = np.asarray(result.angular_intensity, dtype=float)
    if np.any(np.isfinite(angular)):
        axes[0, 1].plot(np.rad2deg(result.angular_theta_rad), angular / max(float(np.nanmax(angular)), EPS), color="#009E73")
        axes[0, 1].set(title="native angular profile on dominant ring", xlabel="angle (deg)", ylabel="normalised intensity")
    else:
        axes[0, 1].text(0.5, 0.5, "not applicable to Gaussian baseline", ha="center", va="center", transform=axes[0, 1].transAxes)
        axes[0, 1].set_title("angular profile")
    common = max(float(np.max(plane[mid, :])), float(np.max(plane[:, mid])), EPS)
    axes[1, 0].plot(x / 1e-3, plane[mid, :] / common, label="horizontal", color="#0072B2")
    axes[1, 0].plot(x / 1e-3, plane[:, mid] / common, label="vertical", color="#CC79A7", ls="--")
    axes[1, 0].set_xlim(-result.focus_halfwidth_m / 1e-3, result.focus_halfwidth_m / 1e-3)
    axes[1, 0].set(title="matched centre lines at z=60 mm", xlabel="coordinate (mm)", ylabel="common-normalised intensity")
    axes[1, 0].legend(frameon=False)
    peak_z = np.maximum(np.max(result.xz_map, axis=1), np.max(result.yz_map, axis=1))
    axes[1, 1].plot(result.z_values_m / 1e-3, peak_z / max(float(np.max(peak_z)), EPS), color="#D55E00")
    axes[1, 1].set(title="centre-slice peak versus z", xlabel="z (mm)", ylabel="normalised peak")
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.suptitle(f"{result.case_id}: profiles derived from native arrays", fontsize=14.5)
    paths = _save(fig, root / "04_profiles" / f"{result.case_id.lower()}_native_profiles")
    plt.close(fig)
    return _record(
        f"{result.case_id}_profiles",
        paths,
        cases=(result.case_id,),
        source=str(result.metadata["source_contract"]),
        native_grid_n=result.native_grid_n,
        native_dx_m=result.native_dx_m,
        render_method="native one-dimensional reductions",
        normalisation="profile maxima; horizontal and vertical lines share one maximum",
        crop_rule=f"line plot +/-{result.focus_halfwidth_m / 1e-3:.4g} mm",
        interpolation="none",
    )


def plot_3d_volume(
    result: Phase2BCaseResult,
    root: Path,
    *,
    display_intensity: np.ndarray | None = None,
    display_grid: Mapping[str, Any] | None = None,
    display_source: str | None = None,
    display_source_grid_n: int | None = None,
    display_source_dx_m: float | None = None,
) -> dict[str, Any]:
    """Plot a pure transverse 3D intensity surface at the canonical z60 plane."""

    plt, colors, _ = _mpl()
    if display_intensity is None:
        if result.sas_hero is not None:
            source = np.asarray(result.sas_hero["intensity"], dtype=float)
            grid = result.sas_hero["grid"]
            render_method = "scalable angular spectrum z60 intensity surface"
            source_n = result.native_grid_n
            source_dx = result.native_dx_m
        else:
            source = np.asarray(result.selected_planes[PHASE2B_HEX_REFERENCE_M], dtype=float)
            axis = _native_axis(result)
            grid = {"x": axis, "dx": result.native_dx_m, "N": result.native_grid_n}
            render_method = "native fixed-grid z60 intensity surface"
            source_n = result.native_grid_n
            source_dx = result.native_dx_m
    else:
        if display_grid is None:
            raise ValueError("display_grid is required with display_intensity")
        source = np.asarray(display_intensity, dtype=float)
        grid = display_grid
        render_method = str(display_source or "validated z60 intensity surface")
        source_n = int(display_source_grid_n or result.native_grid_n)
        source_dx = float(display_source_dx_m or result.native_dx_m)
    halfwidth = float(np.clip(2.25 * result.ring_radius_m, 0.17e-3, 0.27e-3))
    x = np.asarray(grid["x"], dtype=float)
    plane, xc = _crop(source, x, halfwidth)
    intensity = _normalise_peak(plane)
    surface_dx_m = float(grid["dx"])
    mesh_stride = max(1, int(np.ceil(max(intensity.shape) / 240)))
    intensity = intensity[::mesh_stride, ::mesh_stride]
    xc = xc[::mesh_stride]
    X, Y = np.meshgrid(xc / 1e-3, xc / 1e-3)
    fig = plt.figure(figsize=(9.2, 7.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        X,
        Y,
        intensity,
        cmap="magma",
        norm=colors.PowerNorm(gamma=0.55, vmin=0.0, vmax=1.0),
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        rasterized=True,
    )
    ax.set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        zlabel="normalised intensity",
        zlim=(0.0, 1.02),
    )
    ax.view_init(elev=50, azim=-52)
    ax.set_box_aspect((1.0, 1.0, 0.52))
    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.03)
    fig.colorbar(surface, ax=ax, label="normalised intensity", shrink=0.68, pad=0.08)
    title_id = result.case_id.replace("_", " ")
    ax.set_title(
        f"{title_id}: 3D intensity at z=60 mm\n"
        f"source N={source_n}; source dx={source_dx / 1e-6:.2f} um; "
        f"surface dx={surface_dx_m / 1e-6:.2f} um; crop +/-{halfwidth / 1e-3:.3f} mm",
        pad=18,
    )
    paths = _save(fig, root / "05_3d_maps" / f"{result.case_id.lower()}_3d_intensity_surface", dpi=420)
    plt.close(fig)
    return _record(
        f"{result.case_id}_3d_intensity",
        paths,
        cases=(result.case_id,),
        source=str(result.metadata["source_contract"]),
        native_grid_n=source_n,
        native_dx_m=source_dx,
        render_method=f"{render_method}; surface dx {surface_dx_m / 1e-6:.3f} um",
        normalisation="z60 intensity divided by its native/SAS plane maximum; height and colour encode the same intensity",
        crop_rule=f"ring-based focus crop +/-{halfwidth / 1e-3:.4g} mm",
        interpolation="none",
        render_downsampling=f"surface mesh native/SAS index stride {mesh_stride}",
    )


def plot_cross_case_hero(cases: Mapping[str, Phase2BCaseResult], root: Path) -> dict[str, Any]:
    """Compare the five canonical z60 beams with one physical crop and policy."""

    plt, _, _ = _mpl()
    order = ("G0", "B0", "V1", "V3", "H1_REALISTIC")
    labels = ("Gaussian", "Bessel", "vortex-Bessel l=1", "vortex-Bessel l=3", "realistic hex")
    half = 0.74e-3
    fig, axes = plt.subplots(1, 5, figsize=(17.0, 3.8), constrained_layout=True)
    image = None
    for ax, case_id, label in zip(axes, order, labels):
        result = cases[case_id]
        if result.sas_hero is not None:
            plane, xc = _sas_crop(result.sas_hero, half)
        else:
            plane, xc = _crop(result.selected_planes[PHASE2B_HEX_REFERENCE_M], _native_axis(result), half)
        extent = [xc[0] / 1e-3, xc[-1] / 1e-3, xc[0] / 1e-3, xc[-1] / 1e-3]
        image = ax.imshow(
            _normalise_peak(plane),
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="lanczos",
        )
        ax.set_title(label)
        ax.set_xlabel("x (mm)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y (mm)")
    if image is not None:
        fig.colorbar(image, ax=axes, label="independent peak-normalised intensity", shrink=0.78)
    fig.suptitle("Canonical beam families at z=60 mm | common physical crop", fontsize=14.5)
    paths = _save(fig, root / "01_case_inputs" / "canonical_beam_family_sas_hero", dpi=420)
    plt.close(fig)
    return _record(
        "canonical_cross_case_hero",
        paths,
        cases=order,
        source="PHASE 2A scalar chain plus accepted realistic H1 route",
        native_grid_n="512 scalar; 1024 H1",
        native_dx_m="19.53125e-6 scalar; 9.765625e-6 H1",
        render_method="SAS physical focus resampling at z60",
        normalisation="independent peak normalisation with fixed 0..1 scale",
        crop_rule="common +/-0.74 mm",
        interpolation="lanczos display only",
        paired_rule_id="cross_case_z60_common_crop",
    )


def plot_hex_early_mid_late(package: Phase2BHexPackage, root: Path) -> dict[str, Any]:
    """Plot matched continuous, averaged, and signed differences at three z planes."""

    plt, _, _ = _mpl()
    cont = package.continuous
    avg = package.averaged
    x = _native_axis(cont)
    half = min(cont.focus_halfwidth_m, avg.focus_halfwidth_m)
    z_values = (PHASE2B_HEX_EARLY_M, PHASE2B_HEX_REFERENCE_M, PHASE2B_HEX_LATE_M)
    pair_planes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    diff_limit = 0.0
    for z_m in z_values:
        c, xc = _crop(cont.selected_planes[z_m], x, half)
        a, _ = _crop(avg.selected_planes[z_m], x, half)
        c_eq = _equal_power(c)
        a_eq = _equal_power(a)
        common = max(float(np.max(c_eq)), float(np.max(a_eq)), EPS)
        delta = (c_eq - a_eq) / common
        diff_limit = max(diff_limit, float(np.max(np.abs(delta))))
        pair_planes.append((c_eq / common, a_eq / common, delta, xc))
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 11.0), constrained_layout=True)
    image = None
    diff_image = None
    for column, (z_m, values) in enumerate(zip(z_values, pair_planes)):
        c, a, delta, xc = values
        extent = [xc[0] / 1e-3, xc[-1] / 1e-3, xc[0] / 1e-3, xc[-1] / 1e-3]
        for row, (plane, label) in enumerate(((c, "continuous"), (a, "sector averaged"))):
            image = axes[row, column].imshow(
                plane,
                origin="lower",
                extent=extent,
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                interpolation="bicubic",
            )
            axes[row, column].set_title(f"{label} | z={z_m / 1e-3:.0f} mm")
        diff_image = axes[2, column].imshow(
            delta,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-diff_limit,
            vmax=diff_limit,
            interpolation="bicubic",
        )
        axes[2, column].set_title("continuous - averaged")
        for row in range(3):
            axes[row, column].set_xlabel("x (mm)")
            if column == 0:
                axes[row, column].set_ylabel("y (mm)")
            axes[row, column].set_aspect("equal")
    if image is not None:
        fig.colorbar(image, ax=axes[:2, :], label="equal-power, common-peak normalised intensity", shrink=0.75)
    if diff_image is not None:
        fig.colorbar(diff_image, ax=axes[2, :], label="signed shape difference", shrink=0.72)
    fig.suptitle("Realistic continuous versus sector-averaged H1 | matched crop and scales", fontsize=15)
    paths = _save(fig, root / "06_hex_comparisons" / "h1_continuous_vs_averaged_early_z60_late", dpi=400)
    plt.close(fig)
    return _record(
        "H1_continuous_averaged_early_mid_late",
        paths,
        cases=("H1_CONTINUOUS", "H1_AVERAGED"),
        source="MODE 2Y accepted realistic sequential common-4F route",
        native_grid_n=cont.native_grid_n,
        native_dx_m=cont.native_dx_m,
        render_method="native fixed-grid vector ASM planes",
        normalisation="equal power per plane; one common peak per pair; one signed difference limit across z",
        crop_rule=f"common +/-{half / 1e-3:.4g} mm",
        interpolation="bicubic display only",
        paired_rule_id="hex_cont_avg_all_z_common",
    )


def plot_hex_profiles(package: Phase2BHexPackage, root: Path) -> dict[str, Any]:
    """Plot matched native profile and sharpness comparisons at z60."""

    plt, _, _ = _mpl()
    c = package.continuous
    a = package.averaged
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5), constrained_layout=True)
    for result, label, colour, ls in (
        (c, "continuous", "#0072B2", "-"),
        (a, "sector averaged", "#D55E00", "--"),
    ):
        radial = np.asarray(result.radial_intensity, dtype=float)
        angular = np.asarray(result.angular_intensity, dtype=float)
        axes[0].plot(result.radial_radius_m / 1e-3, radial / max(float(np.max(radial)), EPS), label=label, color=colour, ls=ls)
        axes[1].plot(np.rad2deg(result.angular_theta_rad), angular / max(float(np.max(angular)), EPS), label=label, color=colour, ls=ls)
    axes[0].set(xlabel="radius (mm)", ylabel="normalised intensity", title="native radial profile")
    axes[0].set_xlim(0.0, 2.2 * c.ring_radius_m / 1e-3)
    axes[1].set(xlabel="angle (deg)", ylabel="normalised intensity", title="native ring profile")
    metrics = (
        ("edge_gradient_sharpness_mm_inv", "edge gradient", True),
        ("threshold_transition_width_mm", "80-20 width", False),
        ("bright_ridge_fwhm_mm", "ridge FWHM", False),
    )
    improvements = []
    for key, _, higher in metrics:
        cv = float(c.summary[key])
        av = float(a.summary[key])
        improvements.append((cv / av - 1.0) if higher else (av / cv - 1.0))
    bars = axes[2].bar([item[1] for item in metrics], np.asarray(improvements) * 100.0, color=("#0072B2", "#009E73", "#CC79A7"))
    axes[2].bar_label(bars, fmt="%+.1f%%", padding=3)
    axes[2].axhline(0.0, color="0.35", lw=0.8)
    axes[2].set(title="continuous morphology improvement", ylabel="relative improvement (%)")
    axes[2].tick_params(axis="x", rotation=18)
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle("H1 continuous versus sector-averaged native diagnostics at z=60 mm", fontsize=14.5)
    paths = _save(fig, root / "06_hex_comparisons" / "h1_continuous_vs_averaged_native_profiles")
    plt.close(fig)
    return _record(
        "H1_continuous_averaged_profiles",
        paths,
        cases=("H1_CONTINUOUS", "H1_AVERAGED"),
        source="MODE 2Y native N=1024 metrics",
        native_grid_n=c.native_grid_n,
        native_dx_m=c.native_dx_m,
        render_method="native one-dimensional reductions",
        normalisation="each profile divided by its native maximum; metrics unnormalised",
        crop_rule="radial plot through 2.2x V0 ring radius",
        interpolation="none",
        paired_rule_id="hex_cont_avg_profiles",
    )


def plot_highn_hex_hero(package: Phase2BHexPackage, root: Path) -> dict[str, Any] | None:
    """Plot the N=1536 continuous/averaged z60 pair using SAS focus arrays."""

    hero = package.highn_hero
    if not bool(hero.get("enabled")):
        return None
    plt, _, _ = _mpl()
    half = 0.76e-3
    arrays: dict[str, np.ndarray] = {}
    xc = None
    for label in ("continuous", "sector_averaged"):
        sas = {"intensity": hero["sas_planes"][label], "grid": hero["sas_grids"][label]}
        arrays[label], xc = _sas_crop(sas, half)
    c = _equal_power(arrays["continuous"])
    a = _equal_power(arrays["sector_averaged"])
    common = max(float(np.max(c)), float(np.max(a)), EPS)
    c /= common
    a /= common
    delta = c - a
    dmax = max(float(np.max(np.abs(delta))), EPS)
    extent = [xc[0] / 1e-3, xc[-1] / 1e-3, xc[0] / 1e-3, xc[-1] / 1e-3]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), constrained_layout=True)
    image = None
    for ax, plane, title in (
        (axes[0], c, "continuous local orientation"),
        (axes[1], a, "sector-averaged surrogate"),
    ):
        image = ax.imshow(plane, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=1.0, interpolation="lanczos")
        ax.set_title(title)
    diff = axes[2].imshow(delta, origin="lower", extent=extent, cmap="RdBu_r", vmin=-dmax, vmax=dmax, interpolation="lanczos")
    axes[2].set_title("equal-power signed difference")
    for ax in axes:
        ax.set(xlabel="x (mm)", ylabel="y (mm)")
        ax.set_aspect("equal")
    fig.colorbar(image, ax=axes[:2], label="common-normalised intensity", shrink=0.78)
    fig.colorbar(diff, ax=axes[2], label="signed difference", shrink=0.78)
    out_dx = float(hero["sas_metadata"]["continuous"]["output_dx_m"])
    fig.suptitle(
        f"High-N H1 hero at z=60 mm | native N={int(hero['native_grid_n'])}; SAS dx={out_dx / 1e-6:.2f} um",
        fontsize=14.5,
    )
    paths = _save(fig, root / "06_hex_comparisons" / "h1_highn_sas_continuous_vs_averaged_z60", dpi=440)
    plt.close(fig)
    return _record(
        "H1_highn_continuous_averaged_z60",
        paths,
        cases=("H1_CONTINUOUS", "H1_AVERAGED"),
        source="MODE 2Z-HN endpoint machinery",
        native_grid_n=int(hero["native_grid_n"]),
        native_dx_m=float(hero["native_dx_m"]),
        render_method="scalable angular spectrum physical focus resampling",
        normalisation="equal power then one common pair maximum; signed common-limit difference",
        crop_rule="common +/-0.76 mm",
        interpolation="lanczos display only",
        paired_rule_id="hex_highn_z60_common",
    )


def plot_hex_cross_route(package: Phase2BHexPackage, root: Path) -> dict[str, Any]:
    """Plot target through corrected routes with one crop and difference policy."""

    plt, _, _ = _mpl()
    cases = package.cross_route_cases
    half = 0.76e-3
    cropped: list[np.ndarray] = []
    x_axes: list[np.ndarray] = []
    for case in cases:
        array, x = _sas_crop({"intensity": case["sas_intensity"], "grid": case["sas_grid"]}, half)
        cropped.append(_equal_power(array))
        x_axes.append(x)
    reference = cropped[0]
    common_peaks = [arr / max(float(np.max(arr)), EPS) for arr in cropped]
    deltas = [arr - reference for arr in cropped]
    dmax = max(max(float(np.max(np.abs(delta))), EPS) for delta in deltas[1:])
    fig, axes = plt.subplots(2, len(cases), figsize=(18.5, 7.0), constrained_layout=True)
    image = None
    diff_image = None
    for column, case in enumerate(cases):
        x = x_axes[column]
        extent = [x[0] / 1e-3, x[-1] / 1e-3, x[0] / 1e-3, x[-1] / 1e-3]
        image = axes[0, column].imshow(common_peaks[column], origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=1.0, interpolation="lanczos")
        axes[0, column].set_title(str(case["label"]))
        diff_image = axes[1, column].imshow(deltas[column], origin="lower", extent=extent, cmap="RdBu_r", vmin=-dmax, vmax=dmax, interpolation="lanczos")
        axes[1, column].set_title("equal-power delta to target")
        for row in range(2):
            axes[row, column].set_xlabel("x (mm)")
            if column == 0:
                axes[row, column].set_ylabel("y (mm)")
            axes[row, column].set_aspect("equal")
    fig.colorbar(image, ax=axes[0, :], label="independent peak-normalised intensity", shrink=0.7)
    fig.colorbar(diff_image, ax=axes[1, :], label="signed equal-power difference", shrink=0.7)
    fig.suptitle("H1 target, ideal, realism, degradation, and correction | SAS z=60 mm", fontsize=15)
    paths = _save(fig, root / "06_hex_comparisons" / "h1_cross_route_realism_and_correction", dpi=400)
    plt.close(fig)
    return _record(
        "H1_cross_route_realism_correction",
        paths,
        cases=tuple(str(case["route_id"]) for case in cases),
        source="accepted MODE 2W-FIX sequential route and MODE 2S perturbation/correction machinery",
        native_grid_n=int(package.bench["config"].grid_n),
        native_dx_m=float(package.bench["data"]["grid"]["dx"]),
        render_method="SAS physical focus resampling at z60",
        normalisation="equal power; independent 0..1 morphology view; one signed difference limit",
        crop_rule="common +/-0.76 mm",
        interpolation="lanczos display only",
        paired_rule_id="hex_cross_route_common",
    )


def plot_energy_diagnostics(
    case_results: Mapping[str, Phase2BCaseResult],
    ledger_rows: Sequence[Mapping[str, Any]],
    slm_rows: Sequence[Mapping[str, Any]],
    root: Path,
) -> dict[str, Any]:
    """Plot Phase 2A ledger values without recomputing or combining losses."""

    plt, _, _ = _mpl()
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.2), constrained_layout=True)
    h1 = [row for row in ledger_rows if row["case_id"] == "H1"]
    variants = (
        "analytic_target_control",
        "ideal_optical_route",
        "realistic_fixed_bench_route",
        "mild_error_route",
        "deliberately_degraded_route",
    )
    colours = ("#999999", "#0072B2", "#009E73", "#E69F00", "#D55E00")
    for variant, colour in zip(variants, colours):
        rows = sorted((row for row in h1 if row["route_variant"] == variant), key=lambda row: int(row["row_index"]))
        axes[0, 0].plot(
            range(len(rows)),
            [float(row["cumulative_efficiency"]) for row in rows],
            label=variant.replace("_route", "").replace("_", " "),
            color=colour,
            marker="o",
            ms=2.8,
        )
    axes[0, 0].set(title="H1 cumulative Phase 2A ledger", xlabel="ledger row", ylabel="fraction of input")
    axes[0, 0].legend(frameon=False, fontsize=7.2)
    for case_id, colour in zip(("G0", "B0", "V1", "V3", "H1_REALISTIC"), colours):
        result = case_results[case_id]
        power = np.asarray(result.power_by_z, dtype=float)
        axes[0, 1].plot(result.z_values_m / 1e-3, power / max(float(power[0]), EPS), label=case_id.replace("_REALISTIC", ""), color=colour)
    axes[0, 1].set(title="native plane power versus z", xlabel="z (mm)", ylabel="P(z) / P(0)")
    axes[0, 1].legend(frameon=False)
    model_labels = [str(row["slm_model"]).replace("_", "\n") for row in slm_rows]
    modulated = [float(row["modulated_power_fraction"]) for row in slm_rows]
    unmodulated = [float(row["unmodulated_power_fraction"]) for row in slm_rows]
    x = np.arange(len(slm_rows))
    axes[0, 2].bar(x, modulated, color="#0072B2", label="modulated")
    axes[0, 2].bar(x, unmodulated, bottom=modulated, color="#E69F00", label="unmodulated / dead space")
    axes[0, 2].set_xticks(x, model_labels)
    axes[0, 2].set(title="explicit SLM fill-factor models", ylabel="output power fraction")
    axes[0, 2].legend(frameon=False)

    realistic_filter = sorted(
        (
            row for row in ledger_rows
            if row["route_variant"] == "realistic_fixed_bench_route"
            and row["factor_id"] == "simulated_selected_first_order"
        ),
        key=lambda row: str(row["case_id"]),
    )
    axes[1, 0].bar(
        [row["case_id"] for row in realistic_filter],
        [float(row["stage_efficiency"]) for row in realistic_filter],
        color="#009E73",
    )
    axes[1, 0].set(title="simulated selected first order", ylabel="stage efficiency", ylim=(0.0, 1.02))

    useful_ids = ("H1_REALISTIC", "H1_CONTINUOUS", "H1_AVERAGED")
    useful_values = [float(case_results[key].summary["useful_power_fraction"]) for key in useful_ids]
    axes[1, 1].bar(("realistic", "continuous", "averaged"), useful_values, color=("#009E73", "#0072B2", "#D55E00"))
    axes[1, 1].set(title="H1 useful-region power", ylabel="fraction of native plane power", ylim=(0.0, max(useful_values) * 1.25))

    peak_ids = ("G0", "B0", "V1", "V3", "H1_REALISTIC")
    peaks = [float(case_results[key].summary.get("peak_intensity", case_results[key].summary.get("peak_intensity_au"))) for key in peak_ids]
    peak_rel = np.asarray(peaks) / max(float(np.max(peaks)), EPS)
    axes[1, 2].bar([key.replace("_REALISTIC", "") for key in peak_ids], peak_rel, color=colours)
    axes[1, 2].set(title="z60 peak-intensity proxy", ylabel="relative to pack maximum", ylim=(0.0, 1.08))
    axes[1, 2].text(0.02, 0.96, "cross-family relative proxy only", transform=axes[1, 2].transAxes, va="top", fontsize=7.5)
    for ax in axes.ravel():
        ax.grid(alpha=0.2)
    fig.suptitle("Power concepts remain separate: stage throughput, propagated power, and SLM model", fontsize=14.5)
    paths = _save(fig, root / "07_energy_ledgers" / "phase2a_energy_and_throughput_diagnostics")
    plt.close(fig)
    return _record(
        "phase2a_energy_diagnostics",
        paths,
        cases=("G0", "B0", "V1", "V3", "H1"),
        source="outputs/validation/phase2a canonical_power_ledgers.csv and slm_model_comparison.csv",
        native_grid_n="stored Phase 2A ledgers; native propagation arrays",
        native_dx_m="case dependent",
        render_method="direct table plotting plus native plane-power integration",
        normalisation="ledger fraction of input; P(z)/P(0); explicit SLM fractions",
        crop_rule="not applicable",
        interpolation="none",
    )


__all__ = [
    "plot_3d_volume",
    "plot_case_profiles",
    "plot_case_slices",
    "plot_case_xy_bundle",
    "plot_cross_case_hero",
    "plot_energy_diagnostics",
    "plot_hex_cross_route",
    "plot_hex_early_mid_late",
    "plot_hex_profiles",
    "plot_highn_hex_hero",
]
