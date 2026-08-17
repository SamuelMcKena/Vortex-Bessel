"""Stage H.2 hexagon study: air characterisation + in-medium survival.

STUDY 1 — characterise the hexagonal beam in AIR at the focused surface plane.
Sweep the physical knobs (axicon index, apex angle, segment count) and measure
with the honest intensity-only metrics from vbb_hexagon_metrics.

STUDY 2 — feed the best air hexagon into Cr:ZnSe via the existing through-sample
machinery; track six-fold vs depth; compare ideal/lab and corrected/uncorrected.

No polarization element changes: all modifications are to the physical axicon and
segmented-waveplate geometry only.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vbb_study.config import um as BT_UM
from . import vbb_hexagon_metrics as vhm
from . import vbb_polarized_train as vpt
from . import vbb_style

_EPS = 1.0e-30


# ---------------------------------------------------------------------------
# Study 1 — helpers
# ---------------------------------------------------------------------------

def _find_ring_radius_m(I: np.ndarray, grid: dict[str, Any]) -> float:
    """Find the actual ring radius from the azimuthal-average radial peak."""
    R = np.asarray(grid["R"], dtype=float)
    arr = np.asarray(I, dtype=float)
    max_r = float(np.max(R))
    bins = np.linspace(0.0, max_r, 280)
    centres = 0.5 * (bins[:-1] + bins[1:])
    idx = np.clip(np.digitize(R.ravel(), bins) - 1, 0, bins.size - 2)
    radial = np.zeros(centres.size, dtype=float)
    counts = np.zeros_like(radial)
    np.add.at(radial, idx, arr.ravel())
    np.add.at(counts, idx, 1.0)
    radial /= np.maximum(counts, 1.0)
    # Exclude inner 10 % of image to avoid defect-dominated peak
    search = centres > 0.10 * max_r
    if not np.any(search):
        return float(centres[np.argmax(radial)])
    return float(centres[search][np.argmax(radial[search])])


def _get_Ex_Ey_Ez(result: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get Ex, Ey, Ez from either run_preset_train or run_polarized_train result."""
    if "state" in result:
        s = result["state"]
        return np.asarray(s.Ex, dtype=complex), np.asarray(s.Ey, dtype=complex), np.asarray(s.Ez, dtype=complex)
    Ex = np.asarray(result["exit_field"]["Ex"], dtype=complex)
    Ey = np.asarray(result["exit_field"]["Ey"], dtype=complex)
    return Ex, Ey, np.zeros_like(Ex)


def _metrics_for_result(
    result: dict[str, Any],
    *,
    ring_r_m: float | None = None,
) -> dict[str, Any]:
    """Extract hexagon acceptance metrics from a train result dict."""
    I = np.asarray(result["intensity"], dtype=float)
    grid = result["grid"]
    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float)
    ring_r = ring_r_m if ring_r_m is not None else _find_ring_radius_m(I, grid)
    acc = vhm.hexagon_acceptance(I, R, PHI, ring_r)
    sfx = acc["sixfold"]
    axicon_meta = next(
        (h for h in reversed(result.get("history", [])) if h.get("kind") == "physical_axicon"),
        {},
    )
    metrics = result.get("metrics", {})
    tp_abs = float(metrics.get("tp_abs", axicon_meta.get("tp_abs", np.nan)))
    ts_abs = float(metrics.get("ts_abs", axicon_meta.get("ts_abs", np.nan)))
    contrast = float(metrics.get("fresnel_amplitude_contrast", np.nan))
    if not np.isfinite(contrast) and np.isfinite(tp_abs) and np.isfinite(ts_abs):
        contrast = float(abs(tp_abs - ts_abs) / max(0.5 * (tp_abs + ts_abs), _EPS))
    return {
        "ring_r_found_um": float(ring_r / BT_UM),
        "order6_over_order0": float(sfx["order6_over_order0"]),
        "order6_over_non_dc": float(acc.get("order6_over_non_dc", sfx["order6_over_order0"])),
        "localisation_ratio": float(acc["localisation_ratio"]),
        "dark_core_depth": float(acc["dark_core_depth"]),
        "edge_contrast": float(acc["edge_contrast"]),
        "corner_edge_ratio": float(acc["corner_edge_ratio"]),
        "hexagon_pass": bool(acc["pass"]),
        "fresnel_amplitude_contrast": contrast,
        "tp_abs": tp_abs,
        "ts_abs": ts_abs,
        "throughput": float(metrics.get("total_power_au_m2", np.nan)),
    }


def segmented_hexagon_preset_from_config(config: vpt.PolarizedTrainConfig) -> vpt.PresetTrainConfig:
    """Build the named train preset using the Study-H2 swept physical knobs."""

    return vpt.preset_train_config(
        "segmented_vector_hexagon",
        n_axicon=float(config.n_axicon),
        axicon_base_angle_deg=float(config.axicon_base_angle_deg),
        sector_count=max(2, int(config.segment_count)),
    )


def run_segmented_vector_hexagon(
    config: vpt.PolarizedTrainConfig,
    *,
    realism: str = "ideal",
) -> dict[str, Any]:
    """Run the segmented-vector preset with physical knobs from ``config``."""

    preset = segmented_hexagon_preset_from_config(config)
    result = vpt.run_preset_train(config, preset, realism=realism)
    result["config"] = config
    return result


def compute_air_sixfold_zone_um(
    result: dict[str, Any],
    ring_r_m: float | None = None,
    *,
    zone_fraction: float = 0.5,
) -> dict[str, Any]:
    """Air z-extent over which order6 stays above zone_fraction × max(order6).

    Uses the propagation stack already computed by run_polarized_train.
    """
    prop = result.get("propagation", {})
    stack = np.asarray(prop.get("intensity_stack", []), dtype=float)
    z_m = np.asarray(prop.get("z_values_m", []), dtype=float)
    grid = result["grid"]
    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float)
    ring_r = ring_r_m if ring_r_m is not None else _find_ring_radius_m(
        np.asarray(result["intensity"], dtype=float), grid
    )

    if stack.ndim != 3 or z_m.size == 0:
        config = result.get("config") or result.get("preset")
        if config is None:
            return {"air_sixfold_zone_um": np.nan, "order6_z_profile": np.array([]), "z_um": np.array([])}
        Ex, Ey, _ = _get_Ex_Ey_Ez(result)
        z_m = np.linspace(0.0, float(getattr(config, "z_max_m", 110e-6)), int(getattr(config, "z_points", 45)))
        sub_prop = vpt.vector_angular_spectrum_propagate(
            Ex, Ey, result["grid"],
            wavelength_m=float(getattr(config, "wavelength_m", 1.029e-6)),
            n_medium=float(getattr(config, "n_medium", 1.0)),
            z_values_m=z_m,
        )
        stack = np.asarray(sub_prop["intensity_stack"], dtype=float)

    order6_z = np.zeros(len(z_m), dtype=float)
    pass_z = np.zeros(len(z_m), dtype=bool)
    for iz in range(len(z_m)):
        sfx = vhm.sixfold_from_intensity(stack[iz], R, PHI, ring_r)
        order6_z[iz] = sfx["order6_over_order0"]
        pass_z[iz] = bool(vhm.hexagon_acceptance(stack[iz], R, PHI, ring_r)["pass"])

    peak_o6 = float(np.max(order6_z))
    threshold = zone_fraction * peak_o6
    above = np.flatnonzero(order6_z >= threshold)
    if above.size >= 2:
        zone_um = float((z_m[above[-1]] - z_m[above[0]]) / BT_UM)
        z_start_um = float(z_m[above[0]] / BT_UM)
        z_end_um = float(z_m[above[-1]] / BT_UM)
    else:
        zone_um = z_start_um = z_end_um = np.nan

    return {
        "air_sixfold_zone_um": zone_um,
        "air_sixfold_z_start_um": z_start_um,
        "air_sixfold_z_end_um": z_end_um,
        "peak_order6": peak_o6,
        "zone_fraction": zone_fraction,
        "order6_z_profile": order6_z,
        "hexagon_pass_z": pass_z,
        "accepted_plane_count": int(np.count_nonzero(pass_z)),
        "accepted_any": bool(np.any(pass_z)),
        "z_um": z_m / BT_UM,
    }


# ---------------------------------------------------------------------------
# Study 1 — knob sweeps
# ---------------------------------------------------------------------------

def run_air_knob_sweep(
    base_config: vpt.PolarizedTrainConfig,
    *,
    n_axicon_values: Sequence[float] = (1.46, 1.70, 2.00, 2.50, 3.00),
    angle_values: Sequence[float] = (20.0, 32.0, 45.0, 55.0, 65.0),
    segment_counts: Sequence[int] = (3, 6, 9, 12),
    realism: str = "ideal",
) -> pd.DataFrame:
    """Sweep n_axicon, apex angle, segment count independently.

    Each knob is swept alone while the others are held at the base_config value.
    Returns a DataFrame with one row per simulation point.
    """
    rows: list[dict[str, Any]] = []

    def _run_and_record(cfg: vpt.PolarizedTrainConfig, sweep: str, knob_val: float) -> None:
        result = run_segmented_vector_hexagon(cfg, realism=realism)
        m = _metrics_for_result(result)
        zone = compute_air_sixfold_zone_um(result)
        rows.append({
            "sweep": sweep,
            "knob_value": float(knob_val),
            "n_axicon": float(cfg.n_axicon),
            "angle_deg": float(cfg.axicon_base_angle_deg),
            "segment_count": int(cfg.segment_count),
            **m,
            "air_sixfold_zone_um": float(zone["air_sixfold_zone_um"]),
        })

    # Sweep 1: n_axicon
    for n_ax in n_axicon_values:
        cfg = replace(base_config, n_axicon=float(n_ax))
        _run_and_record(cfg, "n_axicon", n_ax)

    # Sweep 2: apex angle
    for ang in angle_values:
        cfg = replace(base_config, axicon_base_angle_deg=float(ang))
        _run_and_record(cfg, "angle_deg", ang)

    # Sweep 3: segment count (use even counts for sector pairs)
    for sc in segment_counts:
        cfg = replace(base_config, segment_count=int(sc * 2))  # pairs → total sectors
        _run_and_record(cfg, "segment_count", sc)

    return pd.DataFrame(rows)


def find_best_config(df: pd.DataFrame, base_config: vpt.PolarizedTrainConfig) -> dict[str, Any]:
    """Return the single-knob settings that maximise order6 and report the recipe."""
    best_n_ax = df[df["sweep"] == "n_axicon"].sort_values("order6_over_order0", ascending=False).iloc[0]
    best_ang = df[df["sweep"] == "angle_deg"].sort_values("order6_over_order0", ascending=False).iloc[0]
    best_sc = df[df["sweep"] == "segment_count"].sort_values("order6_over_order0", ascending=False).iloc[0]

    best_order6 = max(float(best_n_ax["order6_over_order0"]),
                      float(best_ang["order6_over_order0"]),
                      float(best_sc["order6_over_order0"]))
    any_pass = bool(df["hexagon_pass"].any()) if "hexagon_pass" in df.columns else False

    recipe_config = replace(
        base_config,
        n_axicon=float(best_n_ax["n_axicon"]),
        axicon_base_angle_deg=float(best_ang["angle_deg"]),
        segment_count=int(best_sc["segment_count"]),
    )
    feasible = any_pass and best_order6 >= 0.15
    return {
        "best_order6": best_order6,
        "best_n_axicon": float(best_n_ax["n_axicon"]),
        "best_angle_deg": float(best_ang["angle_deg"]),
        "best_segment_count": int(best_sc["segment_count"]),
        "any_hexagon_pass": any_pass,
        "design_path_exists": feasible,
        "recipe_config": recipe_config,
        "verdict": (
            f"Design path EXISTS: order6 = {best_order6:.4f} at n_axicon={best_n_ax['n_axicon']:.2f}, "
            f"angle={best_ang['angle_deg']:.0f}°, segments={int(best_sc['segment_count'])}. "
            f"Recipe: n_axicon={best_n_ax['n_axicon']:.2f}, angle={best_ang['angle_deg']:.0f}°, segments={int(best_sc['segment_count'])}."
        ) if feasible else (
            f"Vector-axicon route does NOT produce an accepted hexagon: best order6 = {best_order6:.4f}, "
            f"but hexagon acceptance is {any_pass}. A dedicated polarization/polygon-shaping optic is required "
            f"for a visually hexagonal intensity pattern."
        ),
    }


# ---------------------------------------------------------------------------
# Study 1 — SurfaceField emission
# ---------------------------------------------------------------------------

def make_surface_field_hexagon(
    result: dict[str, Any],
) -> Any:
    """Build a SurfaceField-compatible object from a polarized train result.

    The exit_field (Ex, Ey) at z=0 is wrapped with the train grid so it can be
    handed off to the through-sample machinery (Stage D).
    """
    from . import vbb_studies as _vs
    Ex, Ey, Ez = _get_Ex_Ey_Ez(result)
    grid = dict(result["grid"])
    cfg = result.get("config") or getattr(result.get("state"), "metadata", {}).get("config")
    meta = {"source": "vbb_hexagon_study.make_surface_field_hexagon"}
    if cfg is not None:
        meta.update({
            "vector_element": str(getattr(cfg, "vector_element", "")),
            "n_axicon": float(getattr(cfg, "n_axicon", np.nan)),
            "axicon_base_angle_deg": float(getattr(cfg, "axicon_base_angle_deg", np.nan)),
            "segment_count": int(getattr(cfg, "segment_count", 0)),
        })
    return _vs.SurfaceField(Ex=Ex, Ey=Ey, Ez=Ez, grid=grid, z_surface_m=0.0, medium_before=1.0, metadata=meta)


# ---------------------------------------------------------------------------
# Study 2 — in-medium propagation
# ---------------------------------------------------------------------------

def propagate_hexagon_in_medium(
    surface_Ex: np.ndarray,
    surface_Ey: np.ndarray,
    grid: dict[str, Any],
    *,
    wavelength_m: float,
    n_medium: float = 2.44,
    z_max_um: float = 300.0,
    z_points: int = 61,
    correction_phase: np.ndarray | None = None,
) -> dict[str, Any]:
    """Propagate (Ex, Ey) in a medium using vectorial ASM.

    Parameters
    ----------
    correction_phase : optional 2D phase array [rad] to apply before propagation;
                       models interface wavefront correction.
    """
    Ex = np.asarray(surface_Ex, dtype=complex)
    Ey = np.asarray(surface_Ey, dtype=complex)
    if correction_phase is not None:
        corr = np.exp(1j * np.asarray(correction_phase, dtype=float))
        Ex = Ex * corr
        Ey = Ey * corr
    z_values = np.linspace(0.0, float(z_max_um) * BT_UM, int(z_points))
    prop = vpt.vector_angular_spectrum_propagate(
        Ex, Ey, grid,
        wavelength_m=float(wavelength_m),
        n_medium=float(n_medium),
        z_values_m=z_values,
    )
    return {
        "n_medium": float(n_medium),
        "wavelength_m": float(wavelength_m),
        "z_values_m": z_values,
        "intensity_stack": np.asarray(prop["intensity_stack"], dtype=float),
        "xz": np.asarray(prop["xz"], dtype=float),
        "z_um": z_values / BT_UM,
        "grid": grid,
        "correction_applied": correction_phase is not None,
        "k_dot_e_rms": float(prop["k_dot_e_rms"]),
    }


def interface_correction_phase(
    grid: dict[str, Any],
    *,
    n_medium: float = 2.44,
    wavelength_m: float,
    target_depth_m: float,
) -> np.ndarray:
    """Simple defocus correction for a flat air/medium interface.

    Compensates for the dominant spherical-aberration-like wavefront shift at
    the interface. Approximates the Zernike-fit correction used in Stage D.
    """
    R = np.asarray(grid["R"], dtype=float)
    k0 = 2.0 * np.pi / float(wavelength_m)
    # The flat interface introduces a path-length difference for off-axis rays.
    # Leading-order correction: quadratic defocus term.
    # This is an approximation; Stage D computes it from a Zernike fit.
    phase_corr = k0 * (float(n_medium) - 1.0) * R ** 2 / (2.0 * float(target_depth_m))
    return phase_corr - float(np.mean(phase_corr))  # piston-remove


def in_medium_sixfold_survival(
    medium_prop: dict[str, Any],
    ring_r_m: float,
    *,
    zone_fraction: float = 0.5,
) -> dict[str, Any]:
    """Survival length: depth over which order6 stays above zone_fraction × max."""
    stack = np.asarray(medium_prop["intensity_stack"], dtype=float)
    z_m = np.asarray(medium_prop["z_values_m"], dtype=float)
    grid = medium_prop["grid"]
    R = np.asarray(grid["R"], dtype=float)
    PHI = np.asarray(grid["PHI"], dtype=float)

    order6_z = np.zeros(len(z_m), dtype=float)
    localisation_z = np.zeros(len(z_m), dtype=float)
    pass_z = np.zeros(len(z_m), dtype=bool)
    for iz in range(len(z_m)):
        sfx = vhm.sixfold_from_intensity(stack[iz], R, PHI, ring_r_m)
        order6_z[iz] = sfx["order6_over_order0"]
        localisation_z[iz] = vhm.localisation_ratio(stack[iz])
        pass_z[iz] = bool(vhm.hexagon_acceptance(stack[iz], R, PHI, ring_r_m)["pass"])

    peak_o6 = float(np.nanmax(order6_z))
    threshold = zone_fraction * peak_o6
    above = np.flatnonzero(order6_z >= threshold)
    if above.size >= 2:
        survival_um = float((z_m[above[-1]] - z_m[above[0]]) / BT_UM)
        z_start_um = float(z_m[above[0]] / BT_UM)
        z_end_um = float(z_m[above[-1]] / BT_UM)
    else:
        survival_um = z_start_um = z_end_um = np.nan

    return {
        "survival_length_um": survival_um,
        "z_start_um": z_start_um,
        "z_end_um": z_end_um,
        "peak_order6": peak_o6,
        "order6_z": order6_z,
        "localisation_z": localisation_z,
        "hexagon_pass_z": pass_z,
        "accepted_plane_count": int(np.count_nonzero(pass_z)),
        "accepted_any": bool(np.any(pass_z)),
        "z_um": z_m / BT_UM,
        "zone_fraction": zone_fraction,
        "correction_applied": bool(medium_prop.get("correction_applied", False)),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_knob_sweep(df: pd.DataFrame, output_path: str | Path, *, ref_order6: float = 0.247) -> Path:
    """Save order6 vs each sweep knob with reference and acceptance lines."""
    vbb_style.apply_style()
    sweeps = [("n_axicon", "axicon index n"), ("angle_deg", "axicon apex angle [°]"), ("segment_count", "segment pairs")]
    n = len(sweeps)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.0), constrained_layout=True)
    for ax, (sweep, xlabel) in zip(axes, sweeps):
        sub = df[df["sweep"] == sweep].sort_values("knob_value")
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.plot(sub["knob_value"], sub["order6_over_order0"], "o-", color="#0072B2", label="order6/order0")
        ax.axhline(0.15, ls=":", lw=1.0, color="0.5", label="visual order6 threshold")
        ax.axhline(ref_order6, ls="--", lw=1.0, color="#009E73", label=f"synthetic ref={ref_order6:.3f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("order6 / order0")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Air sixfold harmonic vs each physical knob (ideal realism)")
    caption = (
        "Knob sweeps for the segmented-vector-axicon route measured at z=0 (surface plane). "
        "Blue: order6/order0 from the intensity FFT. Dotted: visual order6 threshold; full acceptance also "
        "requires dominant order 6 and order6/non-DC strength. "
        "Dashed: reference level 0.247 for a 50%-modulated synthetic hexagon."
    )
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "knob_sweep"})
    plt.close(fig)
    return out


def plot_air_sixfold_zone(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save air sixfold zone length vs the strongest knob."""
    vbb_style.apply_style()
    best_sweep = df.groupby("sweep")["order6_over_order0"].max().idxmax()
    sub = df[df["sweep"] == best_sweep].sort_values("knob_value")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    axes[0].plot(sub["knob_value"], sub["order6_over_order0"], "o-", color="#0072B2")
    axes[0].set_xlabel(best_sweep)
    axes[0].set_ylabel("order6 / order0")
    axes[0].set_title(f"Sixfold signal vs {best_sweep}")
    axes[1].plot(sub["knob_value"], sub["air_sixfold_zone_um"], "s-", color="#D55E00")
    axes[1].set_xlabel(best_sweep)
    axes[1].set_ylabel("air sixfold-harmonic zone [µm]")
    axes[1].set_title(f"Air harmonic zone vs {best_sweep}")
    fig.suptitle(f"Air sixfold harmonic: best knob = {best_sweep}")
    caption = (
        f"Air order6 and sixfold-harmonic zone vs the strongest knob ({best_sweep}). "
        "The harmonic zone is the z-range over which order6 stays above 50 % of its peak; "
        "it is not a visual hexagon-pass criterion."
    )
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "air_sixfold_zone"})
    plt.close(fig)
    return out


def plot_in_medium_survival(
    air_survival: dict[str, Any],
    medium_cases: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save air vs in-medium sixfold survival comparison."""
    vbb_style.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)

    ax = axes[0]
    ax.plot(air_survival["z_um"], air_survival["order6_z_profile"], "-", label="air (n=1)")
    for case in medium_cases:
        label = f"glass n={case['n_medium']:.2f} {'corr' if case['correction_applied'] else 'uncorr'}"
        ax.plot(case["z_um"], case["order6_z"], "--", label=label)
    ax.axhline(0.15, ls=":", lw=0.8, color="0.5")
    ax.set_xlabel("z [µm]")
    ax.set_ylabel("order6 / order0")
    ax.set_title("Sixfold signal vs depth")
    ax.legend(fontsize=8, frameon=False)

    ax2 = axes[1]
    labels = ["air"] + [f"n={c['n_medium']:.2f} {'corr' if c['correction_applied'] else 'uncorr'}" for c in medium_cases]
    zones = [float(air_survival.get("air_sixfold_zone_um", np.nan))] + [float(c.get("survival_length_um", np.nan)) for c in medium_cases]
    colors = ["#0072B2"] + ["#D55E00" if not c["correction_applied"] else "#009E73" for c in medium_cases]
    ax2.bar(labels, zones, color=colors, alpha=0.8)
    ax2.set_ylabel("sixfold-harmonic zone / length [µm]")
    ax2.set_title("Air vs in-medium harmonic zone")
    ax2.tick_params(axis="x", labelrotation=25)

    fig.suptitle("Sixfold harmonic survival: air vs in-medium")
    caption = (
        "Left: order6/order0 vs propagation depth for air and Cr:ZnSe (corrected/uncorrected). "
        "Right: sixfold-harmonic zone comparison. Dashed line: visual order6 threshold; full acceptance also "
        "requires dominant order 6 and order6/non-DC strength."
    )
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "in_medium_survival"})
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# HEXAGON_AMPLIFY.md writer
# ---------------------------------------------------------------------------

def write_hexagon_amplify_doc(
    recipe: dict[str, Any],
    df_sweep: pd.DataFrame,
    air_zone_um: float,
    medium_survival_um: float,
    output_path: str | Path,
) -> Path:
    """Write the HEXAGON_AMPLIFY.md verdict document."""
    best_o6 = float(recipe["best_order6"])
    feasible = bool(recipe["design_path_exists"])
    lines = [
        "# HEXAGON_AMPLIFY — Air sixfold-harmonic study",
        "",
        "## Summary",
        "",
        recipe["verdict"],
        "",
        "## Parameter sweep results",
        "",
        f"Best order6/order0 achieved: **{best_o6:.4f}**",
        f"Accepted hexagon found: **{'yes' if bool(recipe.get('any_hexagon_pass', False)) else 'no'}**",
        "",
        "| sweep knob | best value | best order6 |",
        "|---|---|---|",
    ]
    for sweep in ("n_axicon", "angle_deg", "segment_count"):
        sub = df_sweep[df_sweep["sweep"] == sweep]
        if sub.empty:
            continue
        best_row = sub.sort_values("order6_over_order0", ascending=False).iloc[0]
        lines.append(f"| {sweep} | {best_row['knob_value']:.2f} | {best_row['order6_over_order0']:.4f} |")
    lines += [
        "",
        "## Design recipe (strongest single knob per parameter)",
        "",
        f"- n_axicon: **{recipe['best_n_axicon']:.2f}**",
        f"- apex angle: **{recipe['best_angle_deg']:.0f}°**",
        f"- segment count: **{recipe['best_segment_count']}** total sectors",
        "",
        "## Zone lengths",
        "",
        f"- Air sixfold-harmonic zone: **{air_zone_um:.0f} µm** (z-range where order6 ≥ 50 % of peak)",
        f"- In-medium sixfold-harmonic survival: **{medium_survival_um:.0f} µm** in Cr:ZnSe",
        "",
        "These zone lengths are harmonic diagnostics only. They do not mean the intensity image "
        "passes the visual hexagon-acceptance test.",
        "",
        "## Verdict",
        "",
    ]
    if feasible:
        lines += [
            f"The physical knob sweep reaches order6 = {best_o6:.4f}, which is within or approaching "
            "the 0.15–0.25 range for a clearly hexagonal pattern. A dedicated polarization element "
            "(segmented waveplate) combined with a high-index physical axicon provides the design path.",
            "",
            "Hardware requirements:",
            "- **Segmented waveplate or q-plate**: SLMs alone cannot produce this effect.",
            "- **Physical axicon with significant index contrast** (n_axicon as specified above).",
            "- Steeper apex angles amplify the Fresnel s/p split but raise fabrication difficulty.",
        ]
    else:
        lines += [
            f"The vector-axicon Fresnel route does **not** produce a visually accepted hexagon at this wavelength: "
            f"best order6 = {best_o6:.4f}, and no sweep point passes the full intensity acceptance test. "
            "The Fresnel s/p amplitude contrast at 1029 nm is a few percent even for high-index "
            "axicons. A dedicated polarization-shaping optic (polygon beam shaper, spiral phase plate "
            "array, or polarization-converting element) is required for strong hexagonal contrast.",
            "",
            "The vector-axicon route should be considered a weak/nonlocal sixfold-harmonic bias, "
            "not a primary hexagon generator.",
        ]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


__all__ = [
    "compute_air_sixfold_zone_um",
    "find_best_config",
    "in_medium_sixfold_survival",
    "interface_correction_phase",
    "make_surface_field_hexagon",
    "plot_air_sixfold_zone",
    "plot_in_medium_survival",
    "plot_knob_sweep",
    "propagate_hexagon_in_medium",
    "run_air_knob_sweep",
    "run_segmented_vector_hexagon",
    "segmented_hexagon_preset_from_config",
    "write_hexagon_amplify_doc",
]
