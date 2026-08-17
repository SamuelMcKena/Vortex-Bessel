"""Publication figures for the PHASE 2C objective/interface benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vbb_study.digital_twin.phase2b_figures import _mpl, _record, _save
from vbb_study.digital_twin.phase2c_objective_interface import (
    InterfaceBenchmarkResult,
    ObjectiveBenchmarkResult,
    PHASE2C_CASE_IDS,
    Phase2CBenchmark,
)


EPS = np.finfo(float).tiny
H1_SURFACE_DISPLAY_GAMMA = 0.42


def _peak_normalise(values: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(values, dtype=float), 0.0)
    return array / max(float(np.max(array)), EPS)


def _energy_normalise(values: np.ndarray) -> np.ndarray:
    array = np.maximum(np.asarray(values, dtype=float), 0.0)
    return array / max(float(np.sum(array)), EPS)


def _plotly_magma_colorscale(samples: int = 256) -> list[list[Any]]:
    """Return Plotly colours sampled from the same Matplotlib magma map used by 2D panels."""

    import matplotlib

    count = int(samples)
    if count < 2:
        raise ValueError("samples must be at least two")
    colourmap = matplotlib.colormaps["magma"]
    scale: list[list[Any]] = []
    for index in range(count):
        position = index / float(count - 1)
        red, green, blue, _ = colourmap(position)
        scale.append([
            position,
            f"rgb({round(255 * red)}, {round(255 * green)}, {round(255 * blue)})",
        ])
    return scale


def _extent_um(x_m: np.ndarray, y_m: np.ndarray) -> list[float]:
    x = np.asarray(x_m, dtype=float) / 1e-6
    y = np.asarray(y_m, dtype=float) / 1e-6
    return [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]


def _base_record(
    figure_id: str,
    paths: tuple[Path, Path],
    *,
    cases: tuple[str, ...],
    source: str,
    grid_n: int,
    dx_m: float,
    method: str,
    normalisation: str,
    crop: str,
    paired_rule: str = "",
    render_downsampling: str = "none",
    display_interpolation: str = "none",
) -> dict[str, Any]:
    row = _record(
        figure_id,
        paths,
        cases=cases,
        source=source,
        native_grid_n=grid_n,
        native_dx_m=dx_m,
        render_method=method,
        normalisation=normalisation,
        crop_rule=crop,
        interpolation=display_interpolation,
        paired_rule_id=paired_rule,
        render_downsampling=render_downsampling,
    )
    row.update({
        "stage": "PHASE 2C",
        "matched_plane_enforced": True,
        "display_interpolation_used_for_metrics": False,
    })
    return row


def plot_objective_comparison(result: ObjectiveBenchmarkResult, root: Path) -> dict[str, Any]:
    plt, _, _ = _mpl()
    scalar = _peak_normalise(result.scalar_intensity)
    vector = _peak_normalise(result.vector.intensity)
    difference = np.abs(_energy_normalise(result.scalar_intensity) - _energy_normalise(result.vector.intensity))
    difference /= max(float(np.max(difference)), EPS)
    extent = _extent_um(result.x_m, result.y_m)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), constrained_layout=True)
    for ax, plane, title in zip(
        axes,
        (scalar, vector, difference),
        ("scalar FFT objective", "vector Debye total", "absolute equal-energy difference"),
    ):
        image = ax.imshow(plane, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set(title=title, xlabel="x (um)", ylabel="y (um)", aspect="equal")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    metrics = result.metrics
    fig.suptitle(
        f"{result.case_id}: scalar versus vector objective | corr={float(metrics['scalar_vector_intensity_correlation']):.5f}; "
        f"Ez={100.0 * float(metrics['longitudinal_power_fraction']):.2f}%",
        fontsize=13.5,
    )
    paths = _save(fig, root / "objective" / f"{result.case_id.lower()}_scalar_vs_debye")
    plt.close(fig)
    return _base_record(
        f"{result.case_id}_objective_comparison",
        paths,
        cases=(result.case_id,),
        source="matched accepted pupil: existing scalar focus_to_focal_plane vs independent Cartesian Debye",
        grid_n=result.x_m.size,
        dx_m=float(np.median(np.diff(result.x_m))),
        method="native complex fields; no display interpolation",
        normalisation="scalar/vector plane peak separately; difference uses equal-energy arrays then its own maximum",
        crop="identical full declared objective output field of view",
        paired_rule="phase2c_objective_common_axes_crop",
    )


def plot_objective_profiles(result: ObjectiveBenchmarkResult, root: Path) -> dict[str, Any]:
    plt, _, _ = _mpl()
    scalar = _peak_normalise(result.scalar_intensity)
    vector = _peak_normalise(result.vector.intensity)
    mid_y = scalar.shape[0] // 2
    mid_x = scalar.shape[1] // 2
    x_um = np.asarray(result.x_m) / 1e-6
    y_um = np.asarray(result.y_m) / 1e-6
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True, sharey=True)
    axes[0].plot(x_um, scalar[mid_y], label="scalar", color="#2374a6")
    axes[0].plot(x_um, vector[mid_y], label="Debye", color="#d1495b", linestyle="--")
    axes[1].plot(y_um, scalar[:, mid_x], label="scalar", color="#2374a6")
    axes[1].plot(y_um, vector[:, mid_x], label="Debye", color="#d1495b", linestyle="--")
    axes[0].set(title="horizontal centre line", xlabel="x (um)", ylabel="peak-normalised intensity")
    axes[1].set(title="vertical centre line", xlabel="y (um)")
    for ax in axes:
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.22)
        ax.legend()
    fig.suptitle(f"{result.case_id}: matched scalar and Debye focal profiles", fontsize=13.2)
    paths = _save(fig, root / "profiles" / f"{result.case_id.lower()}_objective_profiles")
    plt.close(fig)
    return _base_record(
        f"{result.case_id}_objective_profiles",
        paths,
        cases=(result.case_id,),
        source="native scalar and Debye focal arrays",
        grid_n=result.x_m.size,
        dx_m=float(np.median(np.diff(result.x_m))),
        method="native centre-line extraction",
        normalisation="each focal intensity divided by its own plane peak",
        crop="full matched objective output axes",
        paired_rule="phase2c_objective_common_axes_profiles",
    )


def plot_debye_components(result: ObjectiveBenchmarkResult, root: Path) -> dict[str, Any]:
    plt, _, _ = _mpl()
    component_planes = (
        np.abs(result.vector.Ex) ** 2,
        np.abs(result.vector.Ey) ** 2,
        np.abs(result.vector.Ez) ** 2,
        result.vector.intensity,
    )
    maximum = max(float(np.max(result.vector.intensity)), EPS)
    extent = _extent_um(result.x_m, result.y_m)
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.8), constrained_layout=True)
    for ax, component, title in zip(axes, component_planes, ("|Ex|^2", "|Ey|^2", "|Ez|^2", "total")):
        image = ax.imshow(component / maximum, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set(title=title, xlabel="x (um)", ylabel="y (um)", aspect="equal")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fractions = result.vector.component_power_fractions
    fig.suptitle(
        f"{result.case_id}: Debye focal components | Ex {100*fractions['Ex_power_fraction']:.1f}%, "
        f"Ey {100*fractions['Ey_power_fraction']:.1f}%, Ez {100*fractions['Ez_power_fraction']:.1f}%",
        fontsize=13.2,
    )
    paths = _save(fig, root / "components" / f"{result.case_id.lower()}_debye_components")
    plt.close(fig)
    return _base_record(
        f"{result.case_id}_debye_components",
        paths,
        cases=(result.case_id,),
        source="independent vector Debye reference",
        grid_n=result.x_m.size,
        dx_m=float(np.median(np.diff(result.x_m))),
        method="native component intensities",
        normalisation="all components divided by the same total-intensity plane maximum",
        crop="full matched objective output axes",
        paired_rule="phase2c_components_common_scale",
    )


def plot_interface_comparison(result: InterfaceBenchmarkResult, root: Path) -> dict[str, Any]:
    plt, _, _ = _mpl()
    scalar = _peak_normalise(result.scalar_intensity)
    vector = _peak_normalise(result.vector.intensity)
    immediate_difference = np.abs(_energy_normalise(result.scalar_intensity) - _energy_normalise(result.vector.intensity))
    immediate_difference /= max(float(np.max(immediate_difference)), EPS)
    scalar_material = _peak_normalise(result.scalar_material.intensity)
    vector_material = _peak_normalise(result.vector_material.intensity)
    material_difference = np.abs(
        _energy_normalise(result.scalar_material.intensity) - _energy_normalise(result.vector_material.intensity)
    )
    material_difference /= max(float(np.max(material_difference)), EPS)
    extent = _extent_um(result.x_m, result.y_m)
    planes = (scalar, vector, immediate_difference, scalar_material, vector_material, material_difference)
    titles = (
        "scalar Fresnel: interface+",
        "vector s/p Fresnel: interface+",
        "interface+ difference",
        "scalar route: material plane",
        "vector route: material plane",
        "material-plane difference",
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.0), constrained_layout=True)
    for ax, plane, title in zip(axes.ravel(), planes, titles):
        image = ax.imshow(plane, origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=1.0)
        ax.set(title=title, xlabel="x (um)", ylabel="y (um)", aspect="equal")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    metrics = result.metrics
    fig.suptitle(
        f"{result.case_id}: planar interface benchmark | Tscalar={float(metrics['transmitted_power_fraction_scalar']):.5f}; "
        f"Tvector={float(metrics['transmitted_power_fraction_vector']):.5f}",
        fontsize=13.5,
    )
    paths = _save(fig, root / "interface" / f"{result.case_id.lower()}_scalar_vs_vector_fresnel")
    plt.close(fig)
    return _base_record(
        f"{result.case_id}_interface_comparison",
        paths,
        cases=(result.case_id,),
        source="same Debye air-side field through scalar normal-incidence and vector spectral Fresnel operators",
        grid_n=result.x_m.size,
        dx_m=float(np.median(np.diff(result.x_m))),
        method="native immediate-interface and matched-depth vector-ASM arrays",
        normalisation="each intensity peak separately; differences compare equal-energy arrays",
        crop="identical full output field and matched planes",
        paired_rule="phase2c_interface_common_axes_crop",
    )


def _surface_crop(
    intensity: np.ndarray,
    x_m: np.ndarray,
    halfwidth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_m, dtype=float)
    values = np.asarray(intensity, dtype=float)
    if values.shape != (x.size, x.size):
        raise ValueError(
            f"surface intensity shape {values.shape} does not match coordinate axis length {x.size}"
        )
    selected = np.flatnonzero(np.abs(x) <= float(halfwidth_m))
    if selected.size < 12:
        raise ValueError("H1 surface crop has fewer than 12 native samples")
    selection = slice(int(selected[0]), int(selected[-1]) + 1)
    return values[selection, selection], x[selection]


def _bandlimited_local_render_intensity(
    components: tuple[np.ndarray, ...],
    x_m: np.ndarray,
    factor: int,
    halfwidth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the native Fourier series densely over a bounded local field of view."""

    native_x = np.asarray(x_m, dtype=float)
    if native_x.ndim != 1 or native_x.size < 2:
        raise ValueError("x_m must be a one-dimensional axis")
    if int(factor) < 1:
        raise ValueError("factor must be positive")
    if not np.isfinite(halfwidth_m) or float(halfwidth_m) <= 0.0:
        raise ValueError("halfwidth_m must be finite and positive")
    native_shape = (native_x.size, native_x.size)
    native_dx = float(np.median(np.diff(native_x)))
    native_half_steps = max(1, int(np.floor(float(halfwidth_m) / native_dx)))
    render_half_steps = native_half_steps * int(factor)
    render_x = (
        float(native_x[native_x.size // 2])
        + np.arange(-render_half_steps, render_half_steps + 1, dtype=float)
        * native_dx
        / int(factor)
    )
    spatial_frequency = np.fft.fftshift(np.fft.fftfreq(native_x.size, d=native_dx))
    synthesis = np.exp(2j * np.pi * spatial_frequency[:, None] * render_x[None, :])
    intensity = np.zeros((render_x.size, render_x.size), dtype=float)
    for component in components:
        values = np.asarray(component, dtype=np.complex128)
        if values.shape != native_shape:
            raise ValueError("component shape does not match x_m")
        spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(values)))
        render_field = synthesis.T @ spectrum @ synthesis / float(native_x.size**2)
        intensity += np.abs(render_field) ** 2
    return intensity, render_x


def plot_h1_surface(
    intensity: np.ndarray,
    x_m: np.ndarray,
    root: Path,
    *,
    surface_id: str,
    title: str,
    halfwidth_m: float,
    source: str,
    native_x_m: np.ndarray,
    resample_factor: int,
) -> dict[str, Any]:
    plt, _, _ = _mpl()
    cropped, x = _surface_crop(intensity, x_m, halfwidth_m)
    normalised = _peak_normalise(cropped)
    X, Y = np.meshgrid(x / 1e-6, x / 1e-6, indexing="xy")
    fig = plt.figure(figsize=(9.2, 7.8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        X,
        Y,
        normalised,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=True,
        shade=False,
        rasterized=True,
    )
    intensity_label = r"normalised intensity $I/I_{max}$"
    ax.set(xlabel="x (um)", ylabel="y (um)", zlabel=intensity_label, zlim=(0.0, 1.02))
    ax.view_init(elev=35, azim=-45)
    ax.set_box_aspect((1.0, 1.0, 0.64))
    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.03)
    fig.colorbar(surface, ax=ax, label=intensity_label, shrink=0.68, pad=0.08)
    render_dx_m = float(np.median(np.diff(x_m)))
    native_dx_m = float(np.median(np.diff(native_x_m)))
    ax.set_title(
        f"{title}\nfixed transverse plane; render dx={render_dx_m / 1e-6:.3f} um "
        f"(native {native_dx_m / 1e-6:.3f} um); "
        f"crop +/-{halfwidth_m / 1e-6:.2f} um; linear 2D-parity colour",
        pad=18,
    )
    paths = _save(fig, root / "h1_3d_intensity_surfaces" / surface_id, dpi=420)
    plt.close(fig)
    row = _base_record(
        f"H1_3d_{surface_id}",
        paths,
        cases=("H1",),
        source=source,
        grid_n=int(np.asarray(native_x_m).size),
        dx_m=native_dx_m,
        method="fixed-plane intensity surface with local complex-field band-limited Fourier synthesis",
        normalisation=(
            "cropped plane intensity divided by its own native plane maximum; height and colour are "
            "identical linear peak-normalised intensity"
        ),
        crop=f"common H1 objective crop +/-{halfwidth_m / 1e-6:.4g} um",
        paired_rule="phase2c_h1_3d_common_crop_view_colour",
        render_downsampling="none; every cropped band-limited render sample rendered",
        display_interpolation=f"local complex-field Fourier synthesis x{int(resample_factor)}; metrics remain native",
    )
    row.update({
        "render_grid_n": int(np.asarray(x_m).size),
        "render_dx_m": render_dx_m,
        "display_resampling_factor": int(resample_factor),
        "native_samples_preserved": True,
    })
    return row


def write_h1_interactive_surface(
    intensity: np.ndarray,
    x_m: np.ndarray,
    root: Path,
    *,
    halfwidth_m: float,
    native_x_m: np.ndarray,
    resample_factor: int,
) -> Path:
    """Write a self-contained rotatable H1 intensity surface with linear/contrast modes."""

    import plotly.graph_objects as go

    cropped, x = _surface_crop(intensity, x_m, halfwidth_m)
    normalised = _peak_normalise(cropped)
    contrast = normalised**H1_SURFACE_DISPLAY_GAMMA
    magma_scale = _plotly_magma_colorscale()
    x_um = x / 1e-6
    output_path = root / "h1_3d_intensity_surfaces" / "h1_vector_debye_interactive.html"
    colourbar = {
        "title": {"text": "I/Imax", "side": "top"},
        "x": 0.92,
        "y": 0.41,
        "len": 0.68,
        "thickness": 22,
    }
    figure = go.Figure(
        data=[
            go.Surface(
                x=x_um,
                y=x_um,
                z=normalised,
                surfacecolor=normalised,
                customdata=normalised,
                colorscale=magma_scale,
                cmin=0.0,
                cmax=1.0,
                colorbar=colourbar,
                hovertemplate=(
                    "x=%{x:.3f} um<br>y=%{y:.3f} um<br>"
                    "I/Imax=%{customdata:.5f}<extra></extra>"
                ),
                lighting={
                    "ambient": 1.0,
                    "diffuse": 0.0,
                    "specular": 0.0,
                    "roughness": 1.0,
                    "fresnel": 0.0,
                },
                lightposition={"x": 0.0, "y": 0.0, "z": 1.0e5},
                visible=False,
            ),
            go.Heatmap(
                x=x_um,
                y=x_um,
                z=normalised,
                customdata=normalised,
                colorscale=magma_scale,
                zmin=0.0,
                zmax=1.0,
                colorbar=colourbar,
                zsmooth=False,
                hovertemplate=(
                    "x=%{x:.3f} um<br>y=%{y:.3f} um<br>"
                    "I/Imax=%{customdata:.5f}<extra></extra>"
                ),
                visible=True,
            ),
        ]
    )
    oblique_camera = {
        "eye": {"x": 0.92, "y": -0.92, "z": 0.72},
        "center": {"x": 0.0, "y": 0.0, "z": -0.08},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
        "projection": {"type": "perspective"},
    }
    figure.update_layout(
        title={
            "text": "H1 vector Debye intensity",
            "x": 0.04,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top",
            "font": {"size": 24},
        },
        template="plotly_white",
        autosize=True,
        margin={"l": 8, "r": 8, "b": 8, "t": 110},
        font={"size": 15, "color": "#24364f"},
        xaxis={
            "title": "x (um)",
            "domain": [0.05, 0.81],
            "range": [float(x_um[0]), float(x_um[-1])],
            "constrain": "domain",
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "title": "y (um)",
            "domain": [0.04, 0.80],
            "range": [float(x_um[0]), float(x_um[-1])],
            "scaleanchor": "x",
            "scaleratio": 1.0,
            "constrain": "domain",
            "showgrid": False,
            "zeroline": False,
        },
        annotations=[
            {
                "text": (
                    f"Fixed transverse plane | render {float(np.median(np.diff(x_m))) / 1e-6:.4f} um | "
                    f"native {float(np.median(np.diff(native_x_m))) / 1e-6:.4f} um | "
                    f"local Fourier x{int(resample_factor)} | crop +/-{halfwidth_m / 1e-6:.2f} um"
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.04,
                "y": 0.925,
                "showarrow": False,
                "xanchor": "left",
                "font": {"size": 13, "color": "#52637a"},
            }
        ],
        scene={
            "domain": {"x": [0.0, 0.86], "y": [0.0, 0.80]},
            "xaxis_title": "x (um)",
            "yaxis_title": "y (um)",
            "zaxis_title": "I/Imax",
            "xaxis": {"range": [float(x_um[0]), float(x_um[-1])]},
            "yaxis": {"range": [float(x_um[0]), float(x_um[-1])]},
            "zaxis": {"range": [0.0, 1.02]},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.62},
            "camera": oblique_camera,
            "dragmode": "orbit",
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.04,
                "y": 0.875,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Linear parity",
                        "method": "update",
                        "args": [
                            {
                                "visible": [False, True],
                                "z": [normalised, normalised],
                                "surfacecolor": [normalised, None],
                                "colorbar.title.text": ["I/Imax", "I/Imax"],
                            },
                            {"xaxis.visible": True, "yaxis.visible": True},
                        ],
                    },
                    {
                        "label": "Shape emphasis",
                        "method": "update",
                        "args": [
                            {
                                "visible": [True, False],
                                "z": [contrast, contrast],
                                "surfacecolor": [contrast, None],
                                "colorbar.title.text": [
                                    f"(I/Imax)^{H1_SURFACE_DISPLAY_GAMMA:.2f}",
                                    f"(I/Imax)^{H1_SURFACE_DISPLAY_GAMMA:.2f}",
                                ],
                            },
                            {
                                "scene.camera": oblique_camera,
                                "scene.zaxis.title": f"(I/Imax)^{H1_SURFACE_DISPLAY_GAMMA:.2f}",
                                "xaxis.visible": False,
                                "yaxis.visible": False,
                            },
                        ],
                    },
                ],
            },
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.43,
                "y": 0.875,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Oblique 3D",
                        "method": "update",
                        "args": [
                            {
                                "visible": [True, False],
                                "z": [normalised, normalised],
                                "surfacecolor": [normalised, None],
                                "colorbar.title.text": ["I/Imax", "I/Imax"],
                            },
                            {
                                "scene.camera": oblique_camera,
                                "scene.zaxis.title": "I/Imax",
                                "xaxis.visible": False,
                                "yaxis.visible": False,
                            },
                        ],
                    },
                    {
                        "label": "Top-down parity",
                        "method": "update",
                        "args": [
                            {
                                "visible": [False, True],
                                "z": [normalised, normalised],
                                "surfacecolor": [normalised, None],
                                "colorbar.title.text": ["I/Imax", "I/Imax"],
                            },
                            {"xaxis.visible": True, "yaxis.visible": True},
                        ],
                    },
                ],
            },
        ],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )
    return output_path


def generate_phase2c_figures(benchmark: Phase2CBenchmark, root: Path) -> list[dict[str, Any]]:
    """Write all required paired figures and return figure provenance rows."""

    records: list[dict[str, Any]] = []
    for subdir in ("objective", "interface", "components", "profiles", "h1_3d_intensity_surfaces"):
        (root / subdir).mkdir(parents=True, exist_ok=True)
    for case_id in PHASE2C_CASE_IDS:
        objective = benchmark.objective_cases[case_id]
        interface = benchmark.interface_cases[case_id]
        records.extend([
            plot_objective_comparison(objective, root),
            plot_objective_profiles(objective, root),
            plot_debye_components(objective, root),
            plot_interface_comparison(interface, root),
        ])
    h1_objective = benchmark.objective_cases["H1"]
    h1_interface = benchmark.interface_cases["H1"]
    resample_factor = int(benchmark.config.h1_surface_resample_factor)
    feature_radius_m = float(h1_objective.metrics["feature_or_ring_radius_vector_um"]) * 1e-6
    halfwidth = min(float(np.max(np.abs(h1_objective.x_m))), max(3.4 * feature_radius_m, 2.5e-6))
    scalar_render, render_x = _bandlimited_local_render_intensity(
        (h1_objective.scalar_Ex, h1_objective.scalar_Ey),
        h1_objective.x_m,
        resample_factor,
        halfwidth,
    )
    vector_render, _ = _bandlimited_local_render_intensity(
        (h1_objective.vector.Ex, h1_objective.vector.Ey, h1_objective.vector.Ez),
        h1_objective.x_m,
        resample_factor,
        halfwidth,
    )
    before_interface_render, _ = _bandlimited_local_render_intensity(
        (h1_interface.incident_Ex, h1_interface.incident_Ey, h1_interface.incident_Ez),
        h1_objective.x_m,
        resample_factor,
        halfwidth,
    )
    interface_render, _ = _bandlimited_local_render_intensity(
        (h1_interface.vector.Ex, h1_interface.vector.Ey, h1_interface.vector.Ez),
        h1_objective.x_m,
        resample_factor,
        halfwidth,
    )
    material_render, _ = _bandlimited_local_render_intensity(
        (
            h1_interface.vector_material.ex,
            h1_interface.vector_material.ey,
            h1_interface.vector_material.ez,
        ),
        h1_objective.x_m,
        resample_factor,
        halfwidth,
    )
    render_halfwidth = float(np.max(np.abs(render_x)))
    surfaces = (
        (scalar_render, "h1_scalar_objective", "H1 scalar objective intensity", "existing scalar objective"),
        (vector_render, "h1_vector_debye", "H1 vector Debye intensity", "independent Debye objective"),
        (
            before_interface_render,
            "h1_before_interface",
            "H1 immediately before interface",
            "Debye air-side interface plane with declared objective-NA support",
        ),
        (interface_render, "h1_after_vector_fresnel", "H1 immediately after vector Fresnel interface", "spectral vector Fresnel interface+"),
        (material_render, "h1_material_plane", "H1 matched material plane", "vector Fresnel plus declared vector ASM depth"),
    )
    for plane, surface_id, title, source in surfaces:
        record = plot_h1_surface(
            plane,
            render_x,
            root,
            surface_id=surface_id,
            title=title,
            halfwidth_m=render_halfwidth,
            source=source,
            native_x_m=h1_objective.x_m,
            resample_factor=resample_factor,
        )
        if surface_id == "h1_vector_debye":
            interactive_path = write_h1_interactive_surface(
                plane,
                render_x,
                root,
                halfwidth_m=render_halfwidth,
                native_x_m=h1_objective.x_m,
                resample_factor=resample_factor,
            )
            record.update({
                "interactive_html_path": str(interactive_path),
                "interactive_self_contained": True,
                "interactive_default_mode": "linear 2D-parity peak-normalised intensity",
                "interactive_alternate_mode": f"shape emphasis gamma={H1_SURFACE_DISPLAY_GAMMA:.2f}",
                "interactive_default_view": "exact top-down heatmap parity",
                "interactive_alternate_view": "perspective oblique 3D",
                "interactive_downsampling": "none; every cropped band-limited focal sample rendered",
            })
        records.append(record)
    return records


__all__ = ["generate_phase2c_figures", "plot_h1_surface", "write_h1_interactive_surface"]
