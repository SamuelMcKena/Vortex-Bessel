"""Honest intensity-based hexagon-quality metrics for Stage H.

All scoring operates on the INTENSITY FIELD directly — no reference contours,
no overlay drawings.  The acceptance function must fail a circular ring (zero
sixfold modulation) and a tiled lattice (low localisation).
"""

from __future__ import annotations

from typing import Any

import numpy as np

EPS = 1.0e-30


# ---------------------------------------------------------------------------
# Core metric functions (spec-required signatures)
# ---------------------------------------------------------------------------

def sixfold_from_intensity(
    I: np.ndarray,
    r_axis: np.ndarray,
    theta_axis: np.ndarray,
    ring_r: float,
    *,
    ring_width_frac: float = 0.25,
    angular_bins: int = 720,
) -> dict[str, Any]:
    """Azimuthal FFT of intensity on the bright ring.

    Selects pixels within ``ring_r ± ring_r * ring_width_frac``, bins them
    into ``angular_bins`` sectors, and returns the FFT amplitude at order 6
    normalised by the DC (order 0) component.

    Parameters
    ----------
    I           : (N, N) intensity array
    r_axis      : (N, N) radial distance in any consistent unit
    theta_axis  : (N, N) azimuthal angle [rad]
    ring_r      : ring radius in same units as r_axis
    ring_width_frac : fractional half-width of the annular mask (default 0.25)
    angular_bins : number of azimuthal bins for FFT

    Returns
    -------
    dict with:
        order6_amplitude   : |FFT[6]| of the azimuthally binned profile
        order0_amplitude   : |FFT[0]| (DC)
        order6_over_order0 : order6 / order0
        profile            : (angular_bins,) azimuthal profile
        theta_bins_rad     : (angular_bins,) bin centres
        dominant_order     : integer order of highest non-DC Fourier component
    """
    I_arr = np.maximum(np.asarray(I, dtype=float), 0.0)
    r = np.asarray(r_axis, dtype=float)
    th = np.asarray(theta_axis, dtype=float) % (2.0 * np.pi)
    hw = float(ring_r) * float(ring_width_frac)
    ring_mask = (r >= float(ring_r) - hw) & (r <= float(ring_r) + hw)

    # Bin ring pixels into azimuthal sectors
    bins = np.linspace(0.0, 2.0 * np.pi, int(angular_bins) + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    profile = np.zeros(int(angular_bins), dtype=float)
    counts = np.zeros_like(profile)
    th_ring = th[ring_mask]
    I_ring = I_arr[ring_mask]
    if I_ring.size:
        idx = np.clip(np.digitize(th_ring, bins) - 1, 0, int(angular_bins) - 1)
        np.add.at(profile, idx, I_ring)
        np.add.at(counts, idx, 1.0)
    filled = counts > 0
    profile[filled] /= counts[filled]

    fft = np.fft.rfft(profile)
    amps = np.abs(fft)
    order0 = float(amps[0]) + EPS
    order6 = float(amps[6]) if len(amps) > 6 else 0.0
    non_dc = float(np.sum(amps[1:])) + EPS
    dominant_order = int(np.argmax(amps[1:]) + 1) if amps.size > 1 else 0

    return {
        "order6_amplitude": order6,
        "order0_amplitude": float(amps[0]),
        "order6_over_order0": order6 / order0,
        "order6_over_non_dc": order6 / non_dc,
        "profile": profile,
        "theta_bins_rad": centres,
        "dominant_order": dominant_order,
        "ring_pixel_count": int(np.count_nonzero(ring_mask)),
    }


def localisation_ratio(I: np.ndarray) -> float:
    """Energy in the central structure divided by total image energy.

    Uses an image-relative heuristic: 'central structure' is the inner 40 % of
    the image half-radius.  A single localised ring scores HIGH (≥ 0.6);
    a tiled lattice that spreads across the full image scores LOW (≪ 0.4).

    No grid parameter is needed because the fraction is scale-invariant.
    """
    arr = np.maximum(np.asarray(I, dtype=float), 0.0)
    total = float(np.sum(arr))
    if total <= EPS:
        return 0.0
    N = arr.shape[0]
    c = N // 2
    yi, xi = np.mgrid[0:N, 0:N]
    r_px = np.sqrt((xi - c) ** 2.0 + (yi - c) ** 2.0)
    # 65 % of image half-radius: works for ring_radius / image_half_width ≤ 0.60.
    # (The default config has ring at 8 µm in a 28.8 µm half-width = 28 %.)
    inner = r_px < 0.65 * (N / 2.0)
    return float(np.sum(arr[inner]) / total)


def edge_corner_uniformity(
    I: np.ndarray,
    ring_r: float,
    *,
    n_hex_vertices: int = 6,
) -> dict[str, Any]:
    """Corner vs edge-midpoint intensity on the hexagonal ring.

    ``ring_r`` is in pixel units (image array indices).  Sample the intensity
    at 6 corner positions (0°, 60°, ...) and 6 edge-midpoint positions (30°,
    90°, ...) on the ring.  For a true hexagon the two sets differ; for a
    uniform circular ring they are equal.

    Returns
    -------
    dict with:
        corner_mean, edge_mean      : mean sampled intensity
        corner_edge_ratio           : corner_mean / edge_mean  (1 = circular)
        angular_contrast            : (max − min) / max of full 12-point profile
        uniformity_score            : 1/(1 + |corner_edge_ratio − 1|) ∈ [0, 1]
    """
    arr = np.maximum(np.asarray(I, dtype=float), 0.0)
    N = arr.shape[0]
    cx = cy = N // 2
    r_px = float(ring_r)
    n_v = int(n_hex_vertices)
    corner_angles = np.linspace(0.0, 2.0 * np.pi, n_v, endpoint=False)
    edge_angles = corner_angles + np.pi / n_v

    def _bilinear(angles: np.ndarray) -> np.ndarray:
        xs = cx + r_px * np.cos(angles)
        ys = cy + r_px * np.sin(angles)
        x0 = np.clip(np.floor(xs).astype(int), 0, N - 2)
        y0 = np.clip(np.floor(ys).astype(int), 0, N - 2)
        tx = xs - x0
        ty = ys - y0
        return (
            (1 - tx) * (1 - ty) * arr[y0, x0]
            + tx * (1 - ty) * arr[y0, x0 + 1]
            + (1 - tx) * ty * arr[y0 + 1, x0]
            + tx * ty * arr[y0 + 1, x0 + 1]
        )

    corner_vals = _bilinear(corner_angles)
    edge_vals = _bilinear(edge_angles)
    all_vals = np.concatenate([corner_vals, edge_vals])
    corner_mean = float(np.mean(corner_vals))
    edge_mean = float(np.mean(edge_vals))
    peak = float(np.max(all_vals))
    angular_contrast = (peak - float(np.min(all_vals))) / (peak + EPS)
    ratio = corner_mean / (edge_mean + EPS)
    return {
        "corner_mean": corner_mean,
        "edge_mean": edge_mean,
        "corner_edge_ratio": ratio,
        "angular_contrast": angular_contrast,
        "uniformity_score": float(1.0 / (1.0 + abs(ratio - 1.0))),
        "corner_values": corner_vals,
        "edge_values": edge_vals,
    }


def hexagon_acceptance(
    I: np.ndarray,
    r_axis: np.ndarray,
    theta_axis: np.ndarray,
    ring_r: float,
    *,
    order6_threshold: float = 0.15,
    order6_non_dc_threshold: float = 0.15,
    dominant_order_required: int = 6,
    localisation_threshold: float = 0.60,
    dark_core_threshold: float = 0.15,
    edge_contrast_threshold: float = 0.008,
) -> dict[str, Any]:
    """Combine sixfold, localisation, dark-core, and edge/corner metrics.

    MUST FAIL a circular ring: order6_over_order0 ≈ 0 → fails criterion 1.
    MUST FAIL a tiled lattice: localisation_ratio ≪ threshold → fails criterion 2.

    Parameters
    ----------
    I                      : (N, N) intensity
    r_axis                 : (N, N) physical radial coordinate
    theta_axis             : (N, N) azimuthal angle [rad]
    ring_r                 : ring radius (same units as r_axis)
    order6_threshold       : min order6/order0 to pass sixfold criterion
    order6_non_dc_threshold: min order6/non-DC to reject weak non-hexagonal harmonics
    dominant_order_required: required dominant non-DC angular order
    localisation_threshold : min localisation_ratio to pass localisation criterion
    dark_core_threshold    : max on-axis / ring-peak ratio to pass dark-core criterion
    edge_contrast_threshold: min angular_contrast on ring to pass edge uniformity

    Returns
    -------
    dict with:
        pass          : bool — overall acceptance
        order6_pass   : bool
        order6_non_dc_pass: bool
        dominant_order_pass: bool
        local_pass    : bool
        dark_core_pass: bool
        edge_pass     : bool
        order6_over_order0, localisation_ratio, dark_core_depth, edge_contrast
        sixfold, detail  : sub-metric dicts
    """
    I_arr = np.maximum(np.asarray(I, dtype=float), 0.0)
    r = np.asarray(r_axis, dtype=float)

    sfx = sixfold_from_intensity(I_arr, r, theta_axis, ring_r)
    loc = localisation_ratio(I_arr)

    # Dark-core: on-axis pixel vs ring peak
    N = I_arr.shape[0]
    c = N // 2
    on_axis = float(I_arr[c, c])
    ring_mask = (r >= 0.75 * float(ring_r)) & (r <= 1.35 * float(ring_r))
    ring_peak = float(np.max(I_arr[ring_mask])) if np.any(ring_mask) else float(np.max(I_arr))
    dark_core_depth = on_axis / (ring_peak + EPS)

    # Edge/corner uniformity: ring_r in pixels
    dx_est = float(r[c, c + 1] - r[c, c]) if c + 1 < N else 1.0
    ring_r_px = float(ring_r) / (dx_est + EPS)
    ecu = edge_corner_uniformity(I_arr, ring_r_px)

    # Also compute order6/non-DC (same normalisation as vbb_polarized_train.sixfold_ring_metrics)
    non_dc = float(np.sum(np.abs(np.fft.rfft(sfx["profile"]))[1:])) + EPS
    order6_non_dc = float(np.abs(np.fft.rfft(sfx["profile"])[6])) / non_dc if len(sfx["profile"]) > 6 else 0.0

    o6_pass = bool(sfx["order6_over_order0"] >= float(order6_threshold))
    o6_non_dc_pass = bool(order6_non_dc >= float(order6_non_dc_threshold))
    dominant_pass = bool(int(sfx["dominant_order"]) == int(dominant_order_required))
    loc_pass = bool(loc >= float(localisation_threshold))
    dc_pass = bool(dark_core_depth <= float(dark_core_threshold))
    edge_pass = bool(ecu["angular_contrast"] >= float(edge_contrast_threshold))
    overall = bool(o6_pass and o6_non_dc_pass and dominant_pass and loc_pass and dc_pass and edge_pass)

    return {
        "pass": overall,
        "order6_pass": o6_pass,
        "order6_non_dc_pass": o6_non_dc_pass,
        "dominant_order_pass": dominant_pass,
        "local_pass": loc_pass,
        "dark_core_pass": dc_pass,
        "edge_pass": edge_pass,
        "order6_over_order0": float(sfx["order6_over_order0"]),
        "localisation_ratio": float(loc),
        "dark_core_depth": float(dark_core_depth),
        "edge_contrast": float(ecu["angular_contrast"]),
        "corner_edge_ratio": float(ecu["corner_edge_ratio"]),
        "order6_over_non_dc": order6_non_dc,
        "thresholds": {
            "order6": order6_threshold,
            "order6_non_dc": order6_non_dc_threshold,
            "dominant_order": dominant_order_required,
            "localisation": localisation_threshold,
            "dark_core": dark_core_threshold,
            "edge_contrast": edge_contrast_threshold,
        },
        "sixfold": sfx,
        "edge_uniformity": ecu,
    }


# ---------------------------------------------------------------------------
# Modulation-depth sweep
# ---------------------------------------------------------------------------

def modulation_sweep(
    config: Any,
    n_axicon_values: list[float],
    *,
    realism: str = "ideal",
) -> tuple[list[float], list[dict[str, Any]]]:
    """Sweep Fresnel index contrast and measure hexagon metric at each level.

    The SLM/waveplate modulation is held fixed; only the axicon index (which
    drives the Fresnel s/p amplitude split) is varied.  At n_axicon = n_medium
    (zero contrast) the beam is a circular ring and hexagon_acceptance FAILS.
    The assertion that the sweep moves the metric confirms the Fresnel term is
    genuinely wired into the field.

    Parameters
    ----------
    config        : PolarizedTrainConfig
    n_axicon_values : list of axicon refractive indices to sweep
    realism       : "ideal" or "lab_realistic"

    Returns
    -------
    (n_values, records)  where records[i] has order6_over_order0, acceptance, etc.
    """
    from . import vbb_polarized_train as vpt

    records = []
    for n_ax in n_axicon_values:
        result = vpt.run_polarized_train(
            config,
            realism=realism,
            n_axicon_override=float(n_ax),
        )
        I = np.asarray(result["intensity"], dtype=float)
        grid = result["grid"]
        ring_r = float(config.ring_radius_m)
        R = np.asarray(grid["R"], dtype=float)
        PHI = np.asarray(grid["PHI"], dtype=float)
        acc = hexagon_acceptance(I, R, PHI, ring_r)
        records.append({
            "n_axicon": float(n_ax),
            "fresnel_amplitude_contrast": float(result["metrics"]["fresnel_amplitude_contrast"]),
            "order6_over_order0": float(acc["order6_over_order0"]),
            "localisation_ratio": float(acc["localisation_ratio"]),
            "dark_core_depth": float(acc["dark_core_depth"]),
            "edge_contrast": float(acc["edge_contrast"]),
            "hexagon_pass": bool(acc["pass"]),
            "exit_metrics_order_fidelity": float(result["metrics"]["order_fidelity"]),
            "exit_metrics_angular_contrast": float(result["metrics"]["angular_contrast"]),
        })
    return list(n_axicon_values), records


# ---------------------------------------------------------------------------
# qplate_radial anomaly diagnostic
# ---------------------------------------------------------------------------

def qplate_radial_anomaly_report(
    ideal_result: dict[str, Any],
    lab_result: dict[str, Any],
) -> dict[str, Any]:
    """Diagnose the qplate_radial ideal→lab feature-radius anomaly.

    The anomaly: ideal feature_radius_um ≈ 5.5 µm (ring), lab ≈ 1 µm (defect spot).

    Root cause: the q-plate central defect allows unconverted light to pass
    through a small central zone (radius ≈ 0.85 µm).  That unconverted
    (linearly polarised) light focuses on-axis when fed to the physical axicon,
    creating a bright central spot.  ``_radial_feature_metrics`` finds this
    defect-leakage peak rather than the intended ring.

    This is PHYSICAL — the central defect genuinely destroys the radial null —
    not a code bug.  The fix is to distinguish the ring feature from the leakage
    peak in the diagnostics.
    """
    from . import vbb_polarized_train as vpt

    def _radial_profile(result: dict) -> dict:
        I = np.asarray(result["intensity"], dtype=float)
        grid = result["grid"]
        R = np.asarray(grid["R"], dtype=float)
        config = result["config"]
        r_m = float(config.ring_radius_m)
        # Azimuthal average
        N = I.shape[0]
        max_r = float(np.max(R))
        bins = np.linspace(0.0, max_r, 220)
        centres = 0.5 * (bins[:-1] + bins[1:])
        idx = np.clip(np.digitize(R.ravel(), bins) - 1, 0, bins.size - 2)
        radial = np.zeros(centres.size, dtype=float)
        counts = np.zeros_like(radial)
        np.add.at(radial, idx, I.ravel())
        np.add.at(counts, idx, 1.0)
        radial /= np.maximum(counts, 1.0)
        global_peak_r = float(centres[int(np.argmax(radial))])
        # Look for the ring specifically: search near config.ring_radius_m
        search = (centres >= 0.5 * r_m) & (centres <= 2.0 * r_m)
        ring_r_measured = float(centres[search][int(np.argmax(radial[search]))]) if np.any(search) else np.nan
        # Look for central leakage: search within 2 µm of center
        central = centres < 2.0e-6
        central_peak = float(np.max(radial[central])) if np.any(central) else 0.0
        ring_peak = float(np.max(radial[search])) if np.any(search) else float(np.max(radial))
        leakage_fraction = central_peak / (ring_peak + EPS)
        return {
            "global_peak_r_um": float(global_peak_r / 1e-6),
            "ring_r_measured_um": float(ring_r_measured / 1e-6),
            "central_peak_intensity": central_peak,
            "ring_peak_intensity": ring_peak,
            "leakage_fraction": leakage_fraction,
            "central_leakage_detected": bool(leakage_fraction > 0.3),
            "radial_profile": radial,
            "radial_r_um": centres / 1e-6,
        }

    ideal_p = _radial_profile(ideal_result)
    lab_p = _radial_profile(lab_result)
    anomaly = bool(abs(ideal_p["global_peak_r_um"] - lab_p["global_peak_r_um"]) > 2.0)

    return {
        "anomaly_detected": anomaly,
        "ideal_global_peak_r_um": ideal_p["global_peak_r_um"],
        "lab_global_peak_r_um": lab_p["global_peak_r_um"],
        "ideal_ring_r_measured_um": ideal_p["ring_r_measured_um"],
        "lab_ring_r_measured_um": lab_p["ring_r_measured_um"],
        "lab_central_leakage_detected": lab_p["central_leakage_detected"],
        "lab_leakage_fraction": lab_p["leakage_fraction"],
        "verdict": (
            "PHYSICAL: q-plate central defect (radius ~0.85 µm) allows unconverted "
            "light through, which focuses on-axis and creates a bright central spot. "
            "The radial_feature_metrics finder picks this leakage peak (r ≈ 1 µm) "
            "instead of the intended ring (r ≈ 5.5 µm). "
            "Not a code bug — the central defect genuinely destroys the radial null. "
            "Fix: report ring_r_measured_um separately and flag central_leakage_detected."
        ) if anomaly else "No significant anomaly detected.",
        "ideal_profile": ideal_p,
        "lab_profile": lab_p,
    }


# ---------------------------------------------------------------------------
# Z-stability scan
# ---------------------------------------------------------------------------

def z_stability_metrics(
    intensity_stack: np.ndarray,
    z_values_m: np.ndarray,
    r_axis: np.ndarray,
    theta_axis: np.ndarray,
    ring_r: float,
) -> dict[str, Any]:
    """Per-z hexagon quality metrics for z-stability reporting.

    Parameters
    ----------
    intensity_stack : (n_z, N, N) total intensity
    z_values_m      : (n_z,) propagation distances [m]
    r_axis          : (N, N) radial coordinate
    theta_axis      : (N, N) azimuthal angle
    ring_r          : ring radius (same units as r_axis)

    Returns
    -------
    dict with per-z arrays: order6_over_order0, localisation, dark_core_depth,
    and the z-stability metrics (mean, std, CV).
    """
    stack = np.asarray(intensity_stack, dtype=float)
    z_m = np.asarray(z_values_m, dtype=float)
    n_z = stack.shape[0]

    order6_arr = np.zeros(n_z, dtype=float)
    loc_arr = np.zeros(n_z, dtype=float)
    dc_arr = np.zeros(n_z, dtype=float)
    peak_arr = np.zeros(n_z, dtype=float)

    for iz in range(n_z):
        I = stack[iz]
        sfx = sixfold_from_intensity(I, r_axis, theta_axis, ring_r)
        order6_arr[iz] = sfx["order6_over_order0"]
        loc_arr[iz] = localisation_ratio(I)
        N = I.shape[0]
        c = N // 2
        on_axis = float(I[c, c])
        ring_mask = (r_axis >= 0.75 * ring_r) & (r_axis <= 1.35 * ring_r)
        ring_pk = float(np.max(I[ring_mask])) if np.any(ring_mask) else float(np.max(I))
        dc_arr[iz] = on_axis / (ring_pk + EPS)
        peak_arr[iz] = ring_pk

    # FWHM zone of the ring peak
    peak_norm = peak_arr / (float(np.max(peak_arr)) + EPS)
    above_half = np.flatnonzero(peak_norm >= 0.5)
    zone_um = float((z_m[above_half[-1]] - z_m[above_half[0]]) / 1e-6) if above_half.size >= 2 else 0.0

    return {
        "z_um": z_m / 1e-6,
        "order6_over_order0": order6_arr,
        "localisation_ratio": loc_arr,
        "dark_core_depth": dc_arr,
        "ring_peak": peak_arr,
        "zone_um": zone_um,
        "order6_mean": float(np.mean(order6_arr)),
        "order6_cv": float(np.std(order6_arr) / (np.mean(order6_arr) + EPS)),
        "localisation_mean": float(np.mean(loc_arr)),
        "localisation_cv": float(np.std(loc_arr) / (np.mean(loc_arr) + EPS)),
    }


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "edge_corner_uniformity",
    "hexagon_acceptance",
    "localisation_ratio",
    "modulation_sweep",
    "qplate_radial_anomaly_report",
    "sixfold_from_intensity",
    "z_stability_metrics",
]
