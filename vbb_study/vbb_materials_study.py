"""Legacy Stage F material-planning helpers.

Any thresholded geometry returned here is a planning-level optical comparison
unless a separate experimentally calibrated material-response model is supplied
and labelled. These helpers do not predict ablation, void formation,
refractive-index change, or weld success.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from vbb_study.config import EPS as BT_EPS, TwinConfig, uJ as BT_UJ, um as BT_UM
from vbb_study.design import compute_design_from_config
from vbb_study.facade import core as _bt
from . import vbb_capsule, vbb_materials, vbb_style


# ---------------------------------------------------------------------------
# Required API from spec
# ---------------------------------------------------------------------------

def incubated_threshold(Fth1: float, N: float, S: float) -> float:
    """Incubated planning threshold proxy: ``Fth(N) = Fth1 * N^(S-1)``.

    Parameters
    ----------
    Fth1 : single-pulse threshold proxy [J/cm^2]
    N    : effective pulse count
    S    : incubation exponent; with S < 1, incubation lowers the proxy threshold
    """
    # Planning proxy only; not a calibrated ablation or modification law.
    return float(Fth1) * float(N) ** (float(S) - 1.0)


def above_threshold_mask_rz(fluence_rz: np.ndarray, Fth: float) -> np.ndarray:
    """Boolean planning mask where ``fluence_rz >= Fth``.

    The mask is not a calibrated material-modification prediction.
    """
    return np.asarray(fluence_rz, dtype=float) >= float(Fth)


def feature_geometry(
    mask_rz: np.ndarray,
    r_axis_um: np.ndarray,
    z_axis_um: np.ndarray,
) -> dict[str, Any]:
    """Planning shape metrics from a pre-thresholded r-z boolean mask.

    Parameters
    ----------
    mask_rz   : (n_r, n_z) above-threshold boolean mask
    r_axis_um : (n_r,) radial axis [µm]  (may be signed or positive)
    z_axis_um : (n_z,) axial axis [µm]

    Returns
    -------
    dict with:
        feature_length_um, mean_diameter_um, diameter_std_um,
        sidewall_straightness, endcap_rounding, top_bottom_asymmetry,
        and capsule_index planning descriptors.
    """
    mask = np.asarray(mask_rz, dtype=bool)
    r_m = np.abs(np.asarray(r_axis_um, dtype=float)) * BT_UM
    z_m = np.asarray(z_axis_um, dtype=float) * BT_UM

    radius_m = np.full(len(z_m), np.nan, dtype=float)
    for iz in range(len(z_m)):
        col = mask[:, iz] if mask.ndim == 2 and mask.shape[1] == len(z_m) else (
            mask[iz, :] if mask.ndim == 2 and mask.shape[0] == len(z_m) else np.asarray([], dtype=bool)
        )
        if np.any(col):
            radius_m[iz] = float(np.max(r_m[col]))

    valid = np.isfinite(radius_m) & (radius_m > 0.0)
    if np.count_nonzero(valid) < 3:
        return {
            "feature_length_um": 0.0,
            "mean_diameter_um": 0.0,
            "diameter_std_um": np.nan,
            "sidewall_straightness": 0.0,
            "endcap_rounding": 0.0,
            "top_bottom_asymmetry": np.nan,
            "capsule_index": 0.0,
        }

    rv = radius_m[valid]
    zv = z_m[valid]
    length_um = float((zv[-1] - zv[0]) / BT_UM)
    mean_r = float(np.mean(rv))
    mean_diameter_um = float(2.0 * mean_r / BT_UM)
    diameter_std_um = float(2.0 * np.std(rv) / BT_UM)
    diameter_cv = float(np.std(2.0 * rv) / (np.mean(2.0 * rv) + BT_EPS))
    slope = float(np.polyfit(zv / BT_UM, rv / BT_UM, 1)[0])
    sidewall_straightness = float(1.0 / (1.0 + 6.0 * diameter_cv + abs(slope)))

    edge_n = max(1, int(0.15 * len(rv)))
    r_start = float(np.mean(rv[:edge_n]))
    r_end = float(np.mean(rv[-edge_n:]))
    asym = float(abs(r_start - r_end) / (mean_r + BT_EPS))
    endcap_rounding = float(np.clip(1.0 - 0.5 * (r_start + r_end) / (mean_r + BT_EPS), 0.0, 1.0))

    flat_score = float(1.0 / (1.0 + 8.0 * diameter_cv))
    sym_score = float(1.0 / (1.0 + 4.0 * asym))
    capsule_index = float(np.clip(0.55 * flat_score + 0.30 * sidewall_straightness + 0.15 * sym_score, 0.0, 1.0))

    return {
        "feature_length_um": length_um,
        "mean_diameter_um": mean_diameter_um,
        "diameter_std_um": diameter_std_um,
        "sidewall_straightness": sidewall_straightness,
        "endcap_rounding": endcap_rounding,
        "top_bottom_asymmetry": asym,
        "capsule_index": capsule_index,
    }


def axial_flatten_apodization(strength: float) -> dict[str, Any]:
    """Radial amplitude apodization transitioning from Gaussian to annular input.

    For an axicon Bessel-Gauss beam, a Gaussian input produces a teardrop-shaped
    above-threshold region (bright early, tapers axially). Emphasising larger SLM
    radii feeds the farther Bessel zone, flattening the profile toward a capsule.

    This is a planning proxy: rigorous implementation requires complex-amplitude
    SLM encoding or a separate amplitude modulator.

    Parameters
    ----------
    strength : 0 = no modification (Gaussian input, teardrop result)
               >0 = ring amplitude emphasis at R = strength × w_slm → capsule

    Returns
    -------
    dict with:
        mask_fn      : callable(R_m, *, beam_radius_m) -> np.ndarray [0, 1]
        light_cost   : fraction of Gaussian input power attenuated by the mask
        strength, peak_radius_factor, description, proxy_note
    """
    s = max(0.0, float(strength))

    def mask_fn(R_m: np.ndarray, *, beam_radius_m: float = 1.0) -> np.ndarray:
        r = np.asarray(R_m, dtype=float)
        w = max(float(beam_radius_m), 1e-30)
        if s == 0.0:
            return np.ones_like(r)
        sigma = w / 2.0
        raw = np.exp(-((r - s * w) ** 2) / (2.0 * sigma ** 2))
        return np.clip(raw / (float(np.max(raw)) + 1e-30), 0.0, 1.0)

    # Light cost: fraction of Gaussian power removed by the amplitude mask
    rn = np.linspace(0.0, 5.0, 4000)            # R / w_slm (dimensionless)
    g2 = np.exp(-2.0 * rn ** 2)                  # Gaussian intensity (unnorm.)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    gauss_power = float(_trapz(g2 * rn, rn))  # ∝ ∫ r G² dr
    if s == 0.0:
        light_cost = 0.0
    else:
        raw_mask = np.exp(-((rn - s) ** 2) / (2.0 * 0.5 ** 2))
        raw_mask /= float(np.max(raw_mask)) + 1e-30
        apod_power = float(_trapz(raw_mask ** 2 * g2 * rn, rn))
        light_cost = max(0.0, 1.0 - apod_power / (gauss_power + 1e-30))

    return {
        "strength": s,
        "mask_fn": mask_fn,
        "light_cost": float(light_cost),
        "peak_radius_factor": float(s),          # R_peak / w_slm
        "description": (
            f"ring-Gaussian SLM amplitude proxy: peak at R = {s:.2f}×w_slm, "
            f"σ = 0.5×w_slm; light_cost = {light_cost:.3f}"
        ),
        "proxy_note": (
            "Planning proxy only. Rigorous implementation requires complex-amplitude "
            "SLM encoding or a separate amplitude modulator."
        ),
    }


# ---------------------------------------------------------------------------
# Holographic zone constraint (analytical)
# ---------------------------------------------------------------------------

def holographic_max_zone_um(
    base_config: TwinConfig,
    target_spot_um: float,
    *,
    ell: int | None = None,
) -> float:
    """Maximum Bessel zone achievable via holographic first-order isolation [µm].

    The first-order isolation fails when the axicon cone ring frequency
    (cone_lpmm) equals the SLM carrier minus the filter radius.  Solved
    analytically from the inverse-design geometry.
    """
    ell_use = int(ell if ell is not None else base_config.target.ell)
    carrier_lpmm = float(base_config.slm.carrier_lpmm)
    filter_lpmm = float(base_config.slm.first_order_filter_radius_lpmm)
    margin_lpmm = 0.06    # ≈ 1 frequency bin at N=512, dx=32 µm
    max_cone_lpmm = carrier_lpmm - filter_lpmm - margin_lpmm
    if max_cone_lpmm <= 0.0:
        return 0.0

    # Probe design with dummy zone length just to get kr_sample
    cfg_probe = replace(
        base_config,
        target=replace(
            base_config.target,
            ell=ell_use,
            target_core_diameter_m=float(target_spot_um) * BT_UM,
            target_bessel_length_m=1.0 * BT_UM,
        ),
    )
    design = compute_design_from_config(cfg_probe)
    kr_sample = float(design.kr_sample_m_inv)
    k_medium = float(cfg_probe.laser.k0 * cfg_probe.material.refractive_index)
    w_slm = float(cfg_probe.laser.beam_radius_on_slm_m)
    # From: cone_lpmm = L × kr_sample² / (2π × 1000 × k_medium × w_slm)
    # L_max = max_cone_lpmm × 2π × 1000 × k_medium × w_slm / kr_sample²
    L_max_m = max_cone_lpmm * 2.0 * math.pi * 1e3 * k_medium * w_slm / (kr_sample ** 2)
    return float(L_max_m / BT_UM)


# ---------------------------------------------------------------------------
# Capsule sweep on an existing journey (no re-propagation)
# ---------------------------------------------------------------------------

def capsule_sweep_from_journey(
    journey: Any,
    config: TwinConfig,
    *,
    strengths: Sequence[float] = (0.0, 0.35, 0.70, 1.0, 1.5),
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply the axial-flatness apodization proxy at varying strengths.

    Uses ``vbb_capsule.apply_axial_flatness_proxy`` on the in-medium volume.
    No re-propagation is needed; this stands in for SLM amplitude apodization
    as a planning proxy.

    Parameters
    ----------
    journey : FullJourneyResult from ``vbb_studies.run_full_source_to_sample``
    config  : the TwinConfig used to produce the journey
    strengths : apodization strengths to sweep (0 = no modification)

    Returns
    -------
    df    : DataFrame, one row per strength
    cases : list of dicts with volume, fluence maps, masks, and metrics
    """
    sample_vol = journey.sample_result.volume_result
    pulse_energy_J = float(config.energy.pulse_energy_at_sample_J)
    threshold = vbb_materials.incubated_threshold_J_cm2(config.material, config.laser.rep_rate_Hz)
    n_pulses = vbb_materials.effective_pulses(config.material, config.laser.rep_rate_Hz)

    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    for s in strengths:
        apod = axial_flatten_apodization(float(s))
        vol_s = vbb_capsule.apply_axial_flatness_proxy(sample_vol, float(s))
        F_xz = vbb_capsule._absolute_centerline_fluence_xz(vol_s, pulse_energy_J=pulse_energy_J)
        z_m = np.asarray(vol_s["z"], dtype=float)
        x_m = np.asarray(vol_s["crop_grid"]["x"], dtype=float)

        shape = vbb_capsule.capsule_shape_metrics(F_xz, x_m, z_m, threshold)
        mask = above_threshold_mask_rz(F_xz, threshold)
        geom = feature_geometry(mask, x_m / BT_UM, z_m / BT_UM)

        # Threshold clearance: does the beam exceed Fth anywhere in the zone?
        threshold_cleared = bool(float(np.nanmax(F_xz)) >= threshold)
        # Fraction of z-planes where at least one pixel exceeds Fth
        n_planes_above = int(np.sum(np.any(F_xz >= threshold, axis=0)))
        zone_above_fraction = float(n_planes_above) / max(1, F_xz.shape[1])

        rows.append({
            "strength": float(s),
            "light_cost": float(apod["light_cost"]),
            "capsule_index": float(shape["capsule_index"]),
            "teardrop_index": float(shape["teardrop_index"]),
            "sidewall_straightness": float(shape["sidewall_straightness"]),
            "endcap_roundness": float(shape["endcap_roundness"]),
            "top_bottom_asymmetry": float(shape["top_bottom_asymmetry"]),
            "mean_diameter_um": float(shape["mean_diameter_um"]),
            "feature_length_um": float(shape["feature_length_um"]),
            "diameter_cv": float(shape["diameter_cv"]),
            "threshold_J_cm2": threshold,
            "threshold_cleared": threshold_cleared,
            "n_planes_above_threshold": n_planes_above,
            "zone_fraction_above_threshold": zone_above_fraction,
            "feature_geometry_capsule_index": float(geom["capsule_index"]),
            "effective_pulses": float(n_pulses),
        })
        cases.append({
            "strength": float(s),
            "volume": vol_s,
            "F_xz": F_xz,
            "mask": mask,
            "threshold_J_cm2": threshold,
            "shape_metrics": shape,
            "feature_geometry": geom,
            "apodization": apod,
            "x_m": x_m,
            "z_m": z_m,
        })

    return pd.DataFrame(rows), cases


# ---------------------------------------------------------------------------
# Extended design solver with gap reporting
# ---------------------------------------------------------------------------

def design_solver_with_gap(
    base_config: TwinConfig,
    *,
    target_zone_um: float = 150.0,
    spot_samples_um: Sequence[float] = (2.0, 3.0, 4.0, 5.0),
    zone_samples_um: Sequence[float] = (50.0, 100.0, 150.0, 200.0, 300.0),
    energy_samples_uJ: Sequence[float] = (5.0, 10.0, 20.0),
    ell_values: Sequence[int] = (0, 3),
) -> pd.DataFrame:
    """Design sweep with explicit holographic-vs-physical zone gap reporting.

    For each (ell, spot, zone, energy) point, reports:
      - cone_lpmm vs carrier: whether holographic first-order isolation works
      - max_holo_zone_um: analytical maximum holographic zone for this spot
      - gap_holo_to_target_um: target_zone - max_holo_zone (negative = achievable)
      - sampling validity (Nyquist) for the physical route
      - hardware reachability

    The "gap to 150 µm" honesty: if max_holo_zone < 150, holographic cannot
    reach the target at these parameters. The physical route is limited by Nyquist
    and hardware but not by the carrier constraint.
    """
    from vbb_study import vbb_regime

    carrier_lpmm = float(base_config.slm.carrier_lpmm)
    filter_lpmm = float(base_config.slm.first_order_filter_radius_lpmm)

    rows: list[dict[str, Any]] = []
    for ell in ell_values:
        for spot_um in spot_samples_um:
            max_holo_um = holographic_max_zone_um(base_config, float(spot_um), ell=int(ell))
            for zone_um in zone_samples_um:
                for energy_uJ in energy_samples_uJ:
                    cfg = replace(
                        base_config,
                        target=replace(
                            base_config.target,
                            ell=int(ell),
                            target_core_diameter_m=float(spot_um) * BT_UM,
                            target_bessel_length_m=float(zone_um) * BT_UM,
                        ),
                        energy=replace(base_config.energy, pulse_energy_in_J=float(energy_uJ) * BT_UJ),
                    )
                    design = compute_design_from_config(cfg)
                    cone_lpmm = float(abs(design.kr_slm_m_inv) / (2.0 * math.pi) / 1e3)
                    holo_feasible = bool(cone_lpmm + filter_lpmm < carrier_lpmm - 0.06)

                    sampling = vbb_regime.sampling_validity(cfg, design)
                    hw = bool(_bt()._hardware_reachable(cfg, design))

                    rows.append({
                        "ell": int(ell),
                        "target_spot_um": float(spot_um),
                        "target_zone_um": float(zone_um),
                        "input_energy_uJ": float(energy_uJ),
                        "kr_sample_m_inv": float(design.kr_sample_m_inv),
                        "kr_slm_m_inv": float(design.kr_slm_m_inv),
                        "cone_lpmm": cone_lpmm,
                        "carrier_lpmm": carrier_lpmm,
                        "filter_lpmm": filter_lpmm,
                        "gamma_slm_deg": float(design.gamma_slm_deg),
                        "w0_sample_um": float(design.w0_sample_m / BT_UM),
                        "mapping_mode": str(design.mapping_mode),
                        "objective_map_source": str(design.objective_map_source),
                        "objective_map_demag": float(design.objective_map_demag),
                        "predicted_bessel_length_um": float(design.predicted_bessel_length_m / BT_UM),
                        # Holographic route
                        "holo_first_order_feasible": holo_feasible,
                        "max_holo_zone_um": float(max_holo_um),
                        "gap_holo_to_target_um": float(zone_um - max_holo_um),
                        # Physical route
                        "phys_sampling_valid": bool(sampling["valid"]),
                        "phys_sampling_violations": ";".join(sampling["violations"]),
                        "phys_samples_per_feature": float(sampling["samples_per_feature"]),
                        # Hardware
                        "hardware_reachable": hw,
                        "holo_feasible_and_hardware": bool(holo_feasible and hw),
                        "phys_feasible_and_hardware": bool(sampling["valid"] and hw),
                    })

    df = pd.DataFrame(rows)
    # Summary columns for the target zone
    df["meets_target_zone_holo"] = df["holo_first_order_feasible"] & (df["target_zone_um"] <= df["max_holo_zone_um"])
    df["meets_target_zone_phys"] = df["phys_feasible_and_hardware"]
    return df


# ---------------------------------------------------------------------------
# Fabrication geometry proxies
# ---------------------------------------------------------------------------

def fabrication_geometry_report(
    case: dict[str, Any],
    *,
    ell: int = 0,
) -> dict[str, Any]:
    """Return threshold-geometry planning proxies for one XZ case.

    Parameters
    ----------
    case : dict with keys 'F_xz', 'x_m', 'z_m', 'threshold_J_cm2'
    ell  : orbital angular momentum (>0 means an annular threshold proxy)

    Legacy key names are kept for compatibility, but these values are not
    calibrated depressed-cladding, TGV, or weld-success predictions.
    """
    F_xz = np.asarray(case["F_xz"], dtype=float)
    x_m = np.asarray(case["x_m"], dtype=float)
    z_m = np.asarray(case["z_m"], dtype=float)
    threshold = float(case["threshold_J_cm2"])

    mask = case.get("mask")
    if mask is None:
        mask = above_threshold_mask_rz(F_xz, threshold)
    mask = np.asarray(mask, dtype=bool)

    # TGV taper: outer radius vs depth
    outer_by_z = np.full(len(z_m), np.nan, dtype=float)
    for iz in range(len(z_m)):
        col = mask[:, iz] if mask.shape[1] == len(z_m) else mask[iz, :]
        if np.any(col):
            outer_by_z[iz] = float(np.max(np.abs(x_m[col])) / BT_UM)

    valid = np.isfinite(outer_by_z)
    if np.count_nonzero(valid) >= 2:
        ov = outer_by_z[valid]
        zv = z_m[valid] / BT_UM
        taper_um = float(np.nanmax(ov) - np.nanmin(ov))
        mean_r_um = float(np.nanmean(ov))
        taper_fraction = float(taper_um / (mean_r_um + BT_EPS))
        slope_um_per_um = float(np.polyfit(zv, ov, 1)[0])
        r_top_um = float(ov[0])
        r_bot_um = float(ov[-1])
        weld_symmetry = float(np.clip(1.0 - abs(r_top_um - r_bot_um) / (mean_r_um + BT_EPS), 0.0, 1.0))
    else:
        taper_um = mean_r_um = taper_fraction = slope_um_per_um = np.nan
        r_top_um = r_bot_um = weld_symmetry = np.nan

    # Depressed-cladding ring metrics (ell > 0 → annular above-threshold zone)
    if int(ell) > 0 and np.count_nonzero(valid) >= 2:
        inner_by_z = np.full(len(z_m), np.nan, dtype=float)
        for iz in range(len(z_m)):
            col = mask[:, iz] if mask.shape[1] == len(z_m) else mask[iz, :]
            if np.any(col):
                absr = np.abs(x_m[col]) / BT_UM
                inner_by_z[iz] = float(np.min(absr))
        iv = np.isfinite(inner_by_z)
        ring_outer_um = mean_r_um
        ring_inner_um = float(np.nanmean(inner_by_z[iv])) if np.any(iv) else np.nan
        ring_width_um = float(ring_outer_um - (ring_inner_um or 0.0)) if np.isfinite(ring_inner_um) else np.nan
    else:
        ring_outer_um = ring_inner_um = ring_width_um = np.nan

    return {
        "tgv_mean_radius_um": mean_r_um,
        "tgv_taper_um": taper_um,
        "tgv_taper_fraction": taper_fraction,
        "tgv_slope_um_per_um": slope_um_per_um,
        "threshold_radius_mean_um": mean_r_um,
        "threshold_taper_um": taper_um,
        "threshold_taper_fraction": taper_fraction,
        "threshold_radius_slope_um_per_um": slope_um_per_um,
        "tgv_top_radius_um": r_top_um,
        "tgv_bottom_radius_um": r_bot_um,
        "weld_symmetry": weld_symmetry,
        "threshold_symmetry_proxy": weld_symmetry,
        "cladding_ring_outer_um": ring_outer_um,
        "cladding_ring_inner_um": ring_inner_um,
        "cladding_ring_width_um": ring_width_um,
        "annular_threshold_proxy_outer_um": ring_outer_um,
        "annular_threshold_proxy_inner_um": ring_inner_um,
        "annular_threshold_proxy_width_um": ring_width_um,
        "ell": int(ell),
        "outer_radius_by_z_um": outer_by_z,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_rz_fluence_pair(
    ideal_case: dict[str, Any],
    lab_case: dict[str, Any],
    output_path: str | Path,
    *,
    title: str = "in-medium fluence | threshold contour",
    correction: str = "corrected",
) -> Path:
    """Save matched ideal-vs-lab r-z fluence maps with threshold contours.

    Both panels share one linear intensity scale (per PLOTTING_AND_SAMPLING.md).
    """
    vbb_style.apply_style()
    cases = [ideal_case, lab_case]
    labels = ["ideal", f"lab (realistic, {correction})"]
    vmax = max(float(np.nanmax(c["F_xz"])) for c in cases) + BT_EPS

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    artist = None
    for ax, case, label in zip(axes, cases, labels):
        F = np.asarray(case["F_xz"], dtype=float)
        x_um = np.asarray(case["x_m"], dtype=float) / BT_UM
        z_um = np.asarray(case["z_m"], dtype=float) / BT_UM
        thr = float(case["threshold_J_cm2"])
        artist = ax.imshow(
            vbb_style.display_scale(F / vmax, gamma=0.55, normalise=False),
            origin="lower", aspect="auto",
            extent=[float(z_um[0]), float(z_um[-1]), float(x_um[0]), float(x_um[-1])],
            cmap=vbb_style.INTENSITY_CMAP, vmin=0.0, vmax=1.0,
        )
        if float(np.nanmin(F)) <= thr <= float(np.nanmax(F)):
            ax.contour(z_um, x_um, F, levels=[thr], colors="white", linewidths=1.2)
        ax.set_xlabel("z [µm, sample plane]")
        ax.set_ylabel("x [µm, sample plane]")
        ax.set_title(f"{label} | Fth = {thr:.3g} J/cm²")

    if artist is not None:
        fig.colorbar(artist, ax=axes, shrink=0.88).set_label(
            "matched linear fluence [a.u.; γ=0.55 display only]"
        )
    fig.suptitle(title)
    caption = (
        "Ideal and lab-realistic in-medium centerline fluence maps with the incubated "
        f"threshold contour (white, Fth = {float(ideal_case['threshold_J_cm2']):.3g} J/cm²). "
        "Matched linear intensity scale across panels; gamma=0.55 for display only. "
        "Model is a simulation-side planning proxy."
    )
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "rz_fluence_pair"})
    plt.close(fig)
    return out


def plot_capsule_morph(
    df: pd.DataFrame,
    cases: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Save teardrop→capsule morphology sweep figure with r-z thumbnails."""
    vbb_style.apply_style()
    n_cases = len(cases)
    fig = plt.figure(figsize=(14.0, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, max(2, n_cases), height_ratios=[1.1, 1.0])

    # Top row: r-z fluence thumbnails at each apodization strength
    vmax = max(float(np.nanmax(c["F_xz"])) for c in cases) + BT_EPS
    for i, case in enumerate(cases):
        ax = fig.add_subplot(gs[0, i])
        F = np.asarray(case["F_xz"], dtype=float)
        x_um = np.asarray(case["x_m"], dtype=float) / BT_UM
        z_um = np.asarray(case["z_m"], dtype=float) / BT_UM
        thr = float(case["threshold_J_cm2"])
        ax.imshow(
            vbb_style.display_scale(F / vmax, gamma=0.55, normalise=False),
            origin="lower", aspect="auto",
            extent=[float(z_um[0]), float(z_um[-1]), float(x_um[0]), float(x_um[-1])],
            cmap=vbb_style.INTENSITY_CMAP, vmin=0.0, vmax=1.0,
        )
        if float(np.nanmin(F)) <= thr <= float(np.nanmax(F)):
            ax.contour(z_um, x_um, F, levels=[thr], colors="white", linewidths=1.0)
        ax.set_title(f"s={case['strength']:.2f}", fontsize=9)
        ax.set_xlabel("z [µm]", fontsize=7)
        if i == 0:
            ax.set_ylabel("x [µm]", fontsize=7)

    # Bottom row: metrics vs strength
    df_s = df.sort_values("strength")
    ax1 = fig.add_subplot(gs[1, :2])
    ax1.plot(df_s["strength"], df_s["capsule_index"], marker="o", label="capsule_index (combined)")
    ax1.plot(df_s["strength"], df_s["feature_geometry_capsule_index"], marker="s", ls="--",
             label="capsule_index (mask)")
    ax1.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax1.set_xlabel("apodization strength")
    ax1.set_ylabel("capsule index [0–1]")
    ax1.legend(fontsize=8, frameon=False)

    ax2 = fig.add_subplot(gs[1, 2:])
    ax2.plot(df_s["strength"], df_s["mean_diameter_um"], marker="o", label="mean diameter")
    ax2.plot(df_s["strength"], df_s["feature_length_um"], marker="s", ls="--", label="feature length")
    ax2b = ax2.twinx()
    ax2b.plot(df_s["strength"], df_s["light_cost"] * 100, marker="^", color="#CC79A7",
              ls=":", label="light cost [%]")
    ax2b.set_ylabel("light cost [%]", color="#CC79A7")
    ax2.set_xlabel("apodization strength")
    ax2.set_ylabel("size [µm, sample plane]")
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, fontsize=8, frameon=False)

    fig.suptitle("Teardrop → Capsule morphology (apodization proxy sweep)")
    caption = (
        "Axial-flatness apodization proxy sweep. Top: in-medium fluence thumbnails with "
        "threshold contour (white) at each strength. Bottom-left: capsule index rises with "
        "strength; bottom-right: feature dimensions and light cost (proxy planning model only)."
    )
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "capsule_morph_sweep"})
    plt.close(fig)
    return out


def plot_design_feasibility(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    target_zone_um: float = 150.0,
    spot_range_um: tuple[float, float] = (2.0, 5.0),
) -> Path:
    """Save design feasibility map (spot × zone) with holographic/physical feasibility marked."""
    vbb_style.apply_style()
    ell_vals = sorted(df["ell"].unique())
    ncols = max(2, len(ell_vals))
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.6), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    for ax, ell in zip(axes, ell_vals):
        sub = df[df["ell"] == ell]
        for _, row in sub.iterrows():
            holo_ok = bool(row["holo_first_order_feasible"])
            phys_ok = bool(row["phys_feasible_and_hardware"])
            if holo_ok:
                color, marker = "#0072B2", "o"
            elif phys_ok:
                color, marker = "#E69F00", "s"
            else:
                color, marker = "#D55E00", "x"
            ax.scatter(
                float(row["target_spot_um"]),
                float(row["target_zone_um"]),
                c=color, marker=marker, s=80, zorder=5, alpha=0.85,
            )
        ax.axhline(target_zone_um, color="#009E73", lw=1.5, ls="--", label=f"target {target_zone_um} µm")
        ax.axvspan(spot_range_um[0], spot_range_um[1], color="#E0E0E0", alpha=0.3, label="general spot range")
        ax.set_xlabel("target spot diameter [µm, sample]")
        ax.set_ylabel("target Bessel zone [µm, sample]")
        ax.set_title(f"ell = {ell}")

    legend_handles = [
        Patch(color="#0072B2", label="holographic feasible"),
        Patch(color="#E69F00", label="physical feasible only"),
        Patch(color="#D55E00", label="neither route feasible"),
    ]
    axes[0].legend(handles=legend_handles, fontsize=8, frameon=False)

    # Annotate max holographic zone for general target spot (3 µm)
    for ax, ell in zip(axes, ell_vals):
        sub3 = df[(df["ell"] == ell) & (df["target_spot_um"] == 3.0)]
        if not sub3.empty:
            max_h = float(sub3["max_holo_zone_um"].iloc[0])
            ax.axhline(max_h, color="#0072B2", lw=1.0, ls=":", alpha=0.7)
            ax.text(spot_range_um[0] + 0.05, max_h + 3, f"max holo zone ≈{max_h:.0f} µm",
                    fontsize=7, color="#0072B2")

    fig.suptitle(f"Design feasibility: holographic vs physical route, target zone = {target_zone_um} µm")
    caption = (
        f"Design feasibility map. Blue circle = holographic first-order isolation achievable; "
        f"orange square = physical route only (holographic cone > carrier); "
        f"red cross = neither route feasible at these parameters. "
        f"Dashed green line: target zone {target_zone_um} µm. "
        f"Dotted blue lines: analytical maximum holographic zone per ell at D=3 µm. "
        f"Shaded band: general spot-size range 2–5 µm."
    )
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "design_feasibility"})
    plt.close(fig)
    return out


def plot_feature_geometry_table(
    df_sweep: pd.DataFrame,
    df_solver: pd.DataFrame,
    fab: dict[str, Any],
    output_path: str | Path,
    *,
    title: str = "Feature geometry summary",
) -> Path:
    """Save the feature geometry table figure (text table)."""
    vbb_style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), constrained_layout=True)
    fig.suptitle(title)

    # Panel 1: capsule sweep summary
    ax = axes[0]
    ax.axis("off")
    s0 = df_sweep.sort_values("strength")
    table_data = [
        [f"{r['strength']:.2f}", f"{r['capsule_index']:.3f}", f"{r['mean_diameter_um']:.2f}",
         f"{r['feature_length_um']:.1f}", f"{r['light_cost']*100:.1f}%",
         "Y" if r["threshold_cleared"] else "N"]
        for _, r in s0.iterrows()
    ]
    col_labels = ["strength", "capsule_idx", "diam µm", "len µm", "light cost", "Fth met"]
    t = ax.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1.0, 1.3)
    ax.set_title("Apodization sweep (s=0 teardrop → s=1.5 capsule)", fontsize=9)

    # Panel 2: design solver gap table (general spot range, 150 µm target)
    ax = axes[1]
    ax.axis("off")
    target_rows = df_solver[df_solver["target_zone_um"] == 150.0].sort_values(
        ["ell", "target_spot_um"]
    )
    if target_rows.empty:
        target_rows = df_solver.sort_values(["target_zone_um", "ell", "target_spot_um"]).head(8)
    sol_data = [
        [f"ell={int(r['ell'])}", f"{r['target_spot_um']:.1f}", f"{r['target_zone_um']:.0f}",
         f"{r['max_holo_zone_um']:.0f}",
         f"{r['gap_holo_to_target_um']:.0f}",
         "Y" if r["holo_first_order_feasible"] else "N",
         "Y" if r["phys_feasible_and_hardware"] else "N"]
        for _, r in target_rows.iterrows()
    ]
    sol_labels = ["ell", "spot µm", "zone µm", "max holo µm", "gap µm", "holo?", "phys?"]
    t2 = ax.table(cellText=sol_data, colLabels=sol_labels, loc="center", cellLoc="center")
    t2.auto_set_font_size(False); t2.set_fontsize(8); t2.scale(1.0, 1.3)
    ax.set_title("Design solver: zone gap to 150 µm target", fontsize=9)

    # Panel 3: fabrication geometry
    ax = axes[2]
    ax.axis("off")
    fab_data = [
        ["TGV mean radius [µm]", f"{fab.get('tgv_mean_radius_um', np.nan):.2f}"],
        ["TGV taper [µm]", f"{fab.get('tgv_taper_um', np.nan):.2f}"],
        ["TGV taper fraction", f"{fab.get('tgv_taper_fraction', np.nan):.3f}"],
        ["Weld symmetry", f"{fab.get('weld_symmetry', np.nan):.3f}"],
        ["Cladding ring outer [µm]", f"{fab.get('cladding_ring_outer_um', np.nan):.2f}"],
        ["Cladding ring width [µm]", f"{fab.get('cladding_ring_width_um', np.nan):.2f}"],
        ["ell", str(fab.get("ell", 0))],
        ["measured feature diam [µm]", "—"],
        ["measured feature length [µm]", "—"],
    ]
    t3 = ax.table(cellText=fab_data, colLabels=["metric", "value (proxy)"],
                  loc="center", cellLoc="center")
    t3.auto_set_font_size(False); t3.set_fontsize(8); t3.scale(1.0, 1.3)
    ax.set_title("Fabrication geometry (s=0, no apodization)", fontsize=9)

    caption = (
        "Feature geometry summary table. Left: apodization sweep metrics showing "
        "teardrop→capsule progression. Centre: design solver gap to 150 µm target "
        "for holographic vs physical routes. Right: fabrication geometry proxies "
        "with blank measured-vs-predicted calibration columns."
    )
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "feature_geometry_table"})
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "above_threshold_mask_rz",
    "axial_flatten_apodization",
    "capsule_sweep_from_journey",
    "design_solver_with_gap",
    "fabrication_geometry_report",
    "feature_geometry",
    "holographic_max_zone_um",
    "incubated_threshold",
    "plot_capsule_morph",
    "plot_design_feasibility",
    "plot_feature_geometry_table",
    "plot_rz_fluence_pair",
]
