"""Stage 8C.3R.1 free-space reference-plane validation figures.

Three diagnostic figures (optical/fluence only, n = 1.0, no material model):

  * ``plot_reference_plane_energy_axis_validation`` - energy waterfall, raw
    power/transmission, commanded-vs-actual axis, XZ centre trajectories,
    expected-vs-measured tilt slope, FOV/crop reliability, claim boundary.
  * ``plot_individual_sensitivity_atlas`` - one upstream error family per row
    (XY, XZ, difference, metrics + translation/deformation/clipping/contamination
    classification + numerical-reliability state).
  * ``plot_fov_convergence_check`` - baseline + perturbations, standard vs
    expanded grid/FOV, with reliability labels.

``final_export_allowed=False`` everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402

from vbb_study.digital_twin.component_plane_pipeline import (
    ComponentPlaneConfig,
    run_component_plane_pipeline,
)
from vbb_study.digital_twin.component_plane_metrics import (
    run_component_plane_scenario,
    classify_translation_vs_deformation,
    compute_axis_tracking,
    stack_to_fluence,
    run_response_curve,
    build_response_curve_families,
    DIAGNOSTIC_SWEEP_LABEL,
    _nearest_index,
)
from vbb_study.digital_twin.component_plane_validation import (
    compute_energy_audit,
    validate_beam_tilt,
    fov_convergence_check,
    zero_control_equivalence,
)
from vbb_study.digital_twin.annular_axis_tracking import (
    RAW_PEAK_LABEL,
    estimate_annular_axis,
    track_axis_trajectory,
)

_RELIAB_COLOR = {
    "numerically_reliable": "#1b5e20",
    "caution_crop_limited": "#ef6c00",
    "invalid_out_of_frame": "#b71c1c",
}

FIGURE_STATUS = "diagnostic_allowed"
MODEL_STATUS = "optical_prediction"
_DIAG_NOTE = "Diagnostic sensitivity sweep. Not an experimentally measured laboratory tolerance."

# Individual upstream-error families (NOT the combined stress test).
DEFAULT_ATLAS_SCENARIOS = (
    "vortex_misregistration",
    "axicon_misregistration",
    "input_decentre_slm_area",
    "beam_tilt_pupil",
    "pupil_decentre_clip",
    "zernike_defocus",
    "zernike_astigmatism",
    "zernike_coma",
    "zernike_spherical",
    "zero_order_leakage",
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _badges(fig: Any, y: float = 0.93) -> None:
    for bx, txt, ec, fc in [
        (0.045, "DIAGNOSTIC ONLY", "#0d47a1", "#e3f2fd"),
        (0.165, "NO MATERIAL RESPONSE", "#4a148c", "#f3e5f5"),
        (0.330, "FREE-SPACE n=1.0", "#1b5e20", "#e8f5e9"),
        (0.455, "COMPONENT-PLANE PHYSICS", "#bf360c", "#fff3e0"),
    ]:
        fig.text(bx, y, txt, ha="left", va="center", fontsize=9, fontweight="bold",
                 color=ec, bbox=dict(boxstyle="round,pad=0.3", facecolor=fc, edgecolor=ec, lw=1.1))


def _edge_ticks(ax: Any, ext: tuple, ring_x: float, ring_y: float) -> None:
    xmin, xmax, ymin, ymax = ext
    ty, tx = 0.05 * (ymax - ymin), 0.05 * (xmax - xmin)
    ax.plot([0, 0], [ymin, ymin + ty], color="white", lw=2.0, zorder=6, clip_on=False)
    ax.plot([ring_x, ring_x], [ymin, ymin + ty], color="cyan", lw=2.0, zorder=7, clip_on=False)
    ax.plot([xmin, xmin + tx], [0, 0], color="white", lw=2.0, zorder=6, clip_on=False)
    ax.plot([xmin, xmin + tx], [ring_y, ring_y], color="cyan", lw=2.0, zorder=7, clip_on=False)


def _card(ax: Any, title: str, lines: list[str], face: str, edge: str, fs: float = 8.0) -> None:
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               facecolor=face, edgecolor=edge, lw=1.3, clip_on=False))
    ax.text(0.04, 0.95, title, transform=ax.transAxes, fontsize=fs + 1.5,
            fontweight="bold", va="top", ha="left", color=edge)
    ax.text(0.04, 0.83, "\n".join(lines), transform=ax.transAxes, fontsize=fs,
            va="top", ha="left", family="monospace", color="#1a1a1a")


def _single_run_reliability(ax_metrics: Mapping[str, Any]) -> str:
    oof = float(ax_metrics["out_of_frame_fraction"])
    margin = float(ax_metrics["field_of_view_margin_um"])
    if oof > 0.02 or margin < 0.0:
        return "invalid_out_of_frame"
    if oof > 0.005:
        return "caution_crop_limited"
    return "numerically_reliable"


def _ext(x: np.ndarray, y: np.ndarray) -> tuple:
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def _save(fig: Any, output_path: str | Path | None, dpi: int, title: str, stage: str) -> None:
    if output_path is None:
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", metadata={
        "Title": title, "stage": stage, "final_export_allowed": "False",
        "Description": f"{stage} free-space reference-plane diagnostic; n=1.0; no material response.",
    })


# ---------------------------------------------------------------------------
# Figure 1 - energy + axis validation
# ---------------------------------------------------------------------------


def plot_reference_plane_energy_axis_validation(
    scenario: str = "pupil_decentre_clip",
    *,
    config: ComponentPlaneConfig | None = None,
    tilt_mrad: float = 24.0,
    output_path: str | Path | None = None,
    dpi: int = 165,
) -> "matplotlib.figure.Figure":
    """Figure 1: free-space energy + axis validation."""
    config = config or ComponentPlaneConfig()
    res = run_component_plane_scenario(scenario, config=config)
    sel = int(res.selected_plane_index)
    base_au = compute_energy_audit(res.baseline, fov_reliable=True)
    mild_au = compute_energy_audit(res.mild, baseline_run=res.baseline, fov_reliable=True)
    sev_au = compute_energy_audit(res.severe, baseline_run=res.baseline, fov_reliable=True)
    tilt = validate_beam_tilt(tilt_mrad, config=config)
    fov = fov_convergence_check({}, config=config)
    equiv = zero_control_equivalence(config)

    sev_st = res.severe.propagated_stack
    x = np.asarray(sev_st.x_um, float); z = np.asarray(sev_st.z_um, float)
    Fs = np.asarray(stack_to_fluence(sev_st).fluence_zyx_j_cm2, float)
    ext = _ext(x, x)

    fig = plt.figure(figsize=(17.5, 13.0), facecolor="white")
    gs = fig.add_gridspec(3, 3, left=0.07, right=0.965, top=0.86, bottom=0.06,
                          hspace=0.42, wspace=0.30, height_ratios=[1.0, 1.05, 0.95])
    fig.suptitle("Stage 8C.3R.1 Free-Space Reference-Plane Energy & Axis Validation\n"
                 f"{res.scenario.title}  (n=1.0 free-space reference; no material model)",
                 x=0.045, y=0.975, ha="left", va="top", fontsize=16, fontweight="bold")
    _badges(fig)

    # (0,0:2) energy waterfall
    axw = fig.add_subplot(gs[0, :2])
    planes = ["input", "after input", "after SLM", "after pupil = ref plane"]
    def chain(au):
        led = au["per_plane_ledger"]
        return [au["input_pulse_energy_uJ"]] + [r["energy_after_uJ"] for r in led]
    for au, lab, col in [(base_au, "baseline", "#37474f"), (mild_au, "mild", "#f57f17"), (sev_au, "severe", "#b71c1c")]:
        axw.plot(range(len(planes)), chain(au)[:len(planes)], "-o", color=col, label=lab, lw=2)
    axw.set_xticks(range(len(planes))); axw.set_xticklabels(planes, fontsize=8.5)
    axw.set_ylabel("pulse energy (uJ)"); axw.set_title("Energy waterfall through component planes", fontsize=11, fontweight="bold")
    axw.grid(alpha=0.3); axw.legend(fontsize=9)

    # (0,2) transmission + encircled bars
    axb = fig.add_subplot(gs[0, 2])
    labels = ["baseline", "mild", "severe"]
    transm = [base_au["transmitted_fraction"], mild_au["transmitted_fraction"], sev_au["transmitted_fraction"]]
    enc = [base_au["encircled_energy_fraction_near_peak"], mild_au["encircled_energy_fraction_near_peak"], sev_au["encircled_energy_fraction_near_peak"]]
    xb = np.arange(3)
    axb.bar(xb - 0.2, transm, 0.4, label="transmitted frac", color="#1565c0")
    axb.bar(xb + 0.2, enc, 0.4, label="encircled@peak", color="#ef6c00")
    axb.set_xticks(xb); axb.set_xticklabels(labels, fontsize=8.5); axb.set_ylim(0, 1.05)
    axb.set_title("Transmission & encircled energy", fontsize=10.5, fontweight="bold"); axb.legend(fontsize=8)

    # (1,0) severe XY fluence + axis edge ticks
    axxy = fig.add_subplot(gs[1, 0])
    im = axxy.imshow(Fs[sel], origin="lower", extent=ext, cmap="viridis", aspect="equal")
    _edge_ticks(axxy, ext, res.severe_axis["ring_centre_x_um"], res.severe_axis["ring_centre_y_um"])
    axxy.set_title("Severe reference-plane XY fluence\n(white=commanded, cyan=measured ticks)", fontsize=10, fontweight="bold")
    axxy.set_xlabel("x (um)"); axxy.set_ylabel("y (um)")
    fig.colorbar(im, ax=axxy, fraction=0.046, pad=0.02, label="J/cm^2")

    # (1,1) XZ centre trajectories
    axt = fig.add_subplot(gs[1, 1])
    for axm, lab, col in [(res.baseline_axis, "baseline", "#37474f"), (res.mild_axis, "mild", "#f57f17"), (res.severe_axis, "severe", "#b71c1c")]:
        axt.plot(z, np.asarray(axm["centre_trajectory_x_um"], float), "-", color=col, lw=1.8, label=lab)
    axt.axhline(0.0, color="0.6", ls=":", lw=1.0)
    axt.set_xlabel("z (um)"); axt.set_ylabel("beam-centre x (um)")
    axt.set_title("Commanded (x=0) vs actual centre trajectory", fontsize=10.5, fontweight="bold")
    axt.grid(alpha=0.3); axt.legend(fontsize=8)

    # (1,2) tilt expected vs measured
    axti = fig.add_subplot(gs[1, 2])
    tilt_run = run_component_plane_pipeline({"enable_beam_tilt": True, "beam_tilt_x_mrad": tilt_mrad}, config=config)
    tax = compute_axis_tracking(tilt_run.propagated_stack)
    zt = np.asarray(tilt_run.propagated_stack.z_um, float)
    meas = np.asarray(tax["centre_trajectory_x_um"], float)
    z0, z1 = tilt["valid_z_fit_range_um"]
    axti.plot(zt, meas, "o", ms=3, color="#00838f", label="measured centre")
    zz = np.linspace(z0, z1, 20)
    axti.plot(zz, meas[_nearest_index(zt, z0)] + tilt["expected_slope_x"] * (zz - z0),
              "--", color="#c62828", lw=2, label="analytical kx/kz")
    axti.set_xlabel("z (um)"); axti.set_ylabel("centre x (um)")
    axti.set_title(f"Tilt {tilt_mrad:g} mrad: expected vs measured\nrel err {tilt['relative_slope_error']*100:.1f}%  ({tilt['grid_pixels_of_displacement']:.1f} px)",
                   fontsize=10, fontweight="bold")
    axti.grid(alpha=0.3); axti.legend(fontsize=8)

    # (2,0) energy audit card
    _card(fig.add_subplot(gs[2, 0]), "Energy / normalisation audit", [
        f"input energy        : {base_au['input_pulse_energy_uJ']:.2f} uJ",
        "                      base / mild / severe",
        f"transmitted frac    : {base_au['transmitted_fraction']:.3f}/{mild_au['transmitted_fraction']:.3f}/{sev_au['transmitted_fraction']:.3f}",
        f"ref-plane energy uJ : {base_au['sample_pulse_energy_uJ']:.2f}/{mild_au['sample_pulse_energy_uJ']:.2f}/{sev_au['sample_pulse_energy_uJ']:.2f}",
        f"peak fluence J/cm2  : {base_au['peak_fluence_J_cm2']:.1f}/{mild_au['peak_fluence_J_cm2']:.1f}/{sev_au['peak_fluence_J_cm2']:.1f}",
        f"peak/ref-energy     : {base_au['peak_to_reference_energy_ratio']:.3f}/{mild_au['peak_to_reference_energy_ratio']:.3f}/{sev_au['peak_to_reference_energy_ratio']:.3f}",
        f"renorm factor       : {sev_au['renormalisation_factor']:.1f}  (no pre-clip renorm)",
        f"energy_accounting_valid : {sev_au['energy_accounting_valid']}",
        f"severe peak rise    : {sev_au['peak_rise_status']}",
    ], "#eceff1", "#37474f")

    # (2,1) axis card
    sa = res.severe_axis
    _card(fig.add_subplot(gs[2, 1]), "Commanded vs actual beam axis (severe)", [
        "commanded axis      : (0.000, 0.000) um",
        f"ring centre         : ({sa['ring_centre_x_um']:.3f}, {sa['ring_centre_y_um']:.3f}) um",
        f"intensity centroid  : ({sa['intensity_centroid_x_um']:.3f}, {sa['intensity_centroid_y_um']:.3f}) um",
        f"peak point          : ({sa['peak_x_um']:.3f}, {sa['peak_y_um']:.3f}) um",
        f"ring fit quality    : {sa['ring_fit_quality']:.3f}",
        f"radial axis error   : {sa['radial_axis_error_um']:.3f} um",
        f"steering x/y        : {sa['beam_steering_angle_x_mrad']:.2f}/{sa['beam_steering_angle_y_mrad']:.2f} mrad",
        f"trajectory fit R2   : {sa['trajectory_fit_quality']:.3f}",
        f"FOV margin          : {sa['field_of_view_margin_um']:.2f} um",
    ], "#e3f2fd", "#0d47a1")

    # (2,2) tilt + FOV + equivalence + claim
    _card(fig.add_subplot(gs[2, 2]), "Validation & claim boundary", [
        f"zero-control equivalent : {equiv['equivalent_within_tolerance']}",
        f"  cfield/I/F sim ~       {equiv['complex_field_similarity']:.6f}",
        f"  ring meas/anal um      {equiv['ring_radius_measured_um']:.2f}/{equiv['ring_radius_analytical_um']:.2f}",
        f"tilt {tilt_mrad:g} mrad rel err   {tilt['relative_slope_error']*100:.1f}% ({tilt['agrees_within_tolerance']})",
        f"  px displacement        {tilt['grid_pixels_of_displacement']:.1f}",
        f"FOV reliability         {fov['metric_convergence_status']}",
        f"  peak conv rel diff     {fov['peak_fluence_rel_diff']*100:.2f}%",
        "",
        "claim: optical fluence diagnostic only;",
        "free-space n=1.0; no material model;",
        "final_export_allowed=False.",
    ], "#e8f5e9", "#1b5e20")

    fig.cpr1_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3r1_reference_plane_validation", "scenario": scenario,
        "equivalent": bool(equiv["equivalent_within_tolerance"]),
        "tilt_rel_err": float(tilt["relative_slope_error"]),
        "fov_status": fov["metric_convergence_status"],
        "severe_peak_rise_status": sev_au["peak_rise_status"],
        "final_export_allowed": False,
    }
    _save(fig, output_path, dpi, "Stage 8C.3R.1 Reference-Plane Energy & Axis Validation",
          "stage8c3r1_reference_plane_validation")
    return fig


# ---------------------------------------------------------------------------
# Figure 2 - individual sensitivity atlas
# ---------------------------------------------------------------------------


def plot_individual_sensitivity_atlas(
    scenarios: Sequence[str] = DEFAULT_ATLAS_SCENARIOS,
    *,
    config: ComponentPlaneConfig | None = None,
    output_path: str | Path | None = None,
    dpi: int = 150,
) -> "matplotlib.figure.Figure":
    """Figure 2: one upstream-error family per row (baseline/severe/diff + metrics)."""
    config = config or ComponentPlaneConfig()
    results = [run_component_plane_scenario(k, config=config) for k in scenarios]
    nrow = len(results)

    fig = plt.figure(figsize=(15.0, 2.55 * nrow + 1.4), facecolor="white")
    gs = fig.add_gridspec(nrow, 4, left=0.10, right=0.975,
                          top=1.0 - 0.9 / (2.55 * nrow + 1.4), bottom=0.04,
                          hspace=0.55, wspace=0.22, width_ratios=[1, 1, 1, 1.25])
    fig.suptitle("Stage 8C.3R.1 Individual Diagnostic Sensitivity Atlas  "
                 "(free-space n=1.0; no material model)\n" + _DIAG_NOTE,
                 x=0.04, y=0.995, ha="left", va="top", fontsize=14, fontweight="bold")

    for r, res in enumerate(results):
        sel = int(res.selected_plane_index)
        b_st, s_st = res.baseline.propagated_stack, res.severe.propagated_stack
        x = np.asarray(b_st.x_um, float); z = np.asarray(b_st.z_um, float)
        yc = _nearest_index(x, 0.0)
        Fb = np.asarray(stack_to_fluence(b_st).fluence_zyx_j_cm2, float)
        Fs = np.asarray(stack_to_fluence(s_st).fluence_zyx_j_cm2, float)
        ext = _ext(x, x); ext_xz = (float(z.min()), float(z.max()), float(x.min()), float(x.max()))
        vmax = max(float(Fb[sel].max()), float(Fs[sel].max()))
        diff = Fs[sel] - Fb[sel]
        dabs = max(float(np.abs(diff).max()), 1e-12)

        ax0 = fig.add_subplot(gs[r, 0])
        ax0.imshow(Fb[sel], origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        ax0.set_ylabel(res.scenario.key, fontsize=8.5, fontweight="bold")
        if r == 0: ax0.set_title("baseline XY", fontsize=10, fontweight="bold")
        ax1 = fig.add_subplot(gs[r, 1])
        im1 = ax1.imshow(Fs[sel], origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        _edge_ticks(ax1, ext, res.severe_axis["ring_centre_x_um"], res.severe_axis["ring_centre_y_um"])
        if r == 0: ax1.set_title("severe XY + axis ticks", fontsize=10, fontweight="bold")
        fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
        ax2 = fig.add_subplot(gs[r, 2])
        ax2.imshow(Fs[:, yc, :].T, origin="lower", extent=ext_xz, cmap="viridis", aspect="auto")
        if r == 0: ax2.set_title("severe XZ", fontsize=10, fontweight="bold")
        ax2.set_xlabel("z (um)" if r == nrow - 1 else "")

        cls = res.severe_class; en_b = res.baseline_energy; en_s = res.severe_energy
        verdict = []
        if cls["translation_dominated_boolean"]: verdict.append("translation")
        if cls["residual_shape_deformation_score"] > 0.15 and not cls["translation_dominated_boolean"]: verdict.append("deformation")
        if cls["throughput_loss_fraction"] > 0.05: verdict.append("clipping")
        if cls["core_contamination_fraction"] > 0.02: verdict.append("core-contam")
        rel = _single_run_reliability(res.severe_axis)
        axc = fig.add_subplot(gs[r, 3])
        _card(axc, res.scenario.title, [
            f"severe: {res.scenario.severe_label}",
            f"transmit {en_s['transmitted_fraction']:.3f}  peakF {en_s['peak_fluence_J_cm2']:.0f}",
            f"reg sim {cls['registered_similarity_score']:.2f}  resid {cls['residual_shape_deformation_score']:.2f}",
            f"cshift {cls['centroid_shift_um']:.2f}um  thru_loss {cls['throughput_loss_fraction']:.2f}",
            f"core-contam {cls['core_contamination_fraction']:.3f}",
            f"class: {', '.join(verdict) if verdict else 'near-baseline'}",
            f"reliability: {rel}",
        ], "#fafafa", "#455a64", fs=7.4)

    fig.cpr1_metadata = {"stage": "stage8c3r1_individual_sensitivity_atlas",  # type: ignore[attr-defined]
                         "scenarios": list(scenarios), "final_export_allowed": False}
    _save(fig, output_path, dpi, "Stage 8C.3R.1 Individual Sensitivity Atlas",
          "stage8c3r1_individual_sensitivity_atlas")
    return fig


# ---------------------------------------------------------------------------
# Figure 3 - FOV convergence
# ---------------------------------------------------------------------------


def plot_fov_convergence_check(
    cases: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    config: ComponentPlaneConfig | None = None,
    output_path: str | Path | None = None,
    dpi: int = 160,
) -> "matplotlib.figure.Figure":
    """Figure 3: standard vs expanded grid/FOV reliability for several cases."""
    config = config or ComponentPlaneConfig()
    if cases is None:
        cases = {
            "baseline": {},
            "beam tilt 24 mrad": {"enable_beam_tilt": True, "beam_tilt_x_mrad": 24.0},
            "pupil clip+decentre": {"enable_pupil_clipping": True, "pupil_radius_um": 12.0, "pupil_decentre_x_um": 12.0},
            "decentre+SLM area": {"enable_beam_decentre": True, "beam_decentre_x_um": 16.0,
                                  "enable_slm_active_area": True, "slm_active_width_um": 30.0, "slm_active_height_um": 30.0},
        }
    checks = {name: fov_convergence_check(ctrl, config=config) for name, ctrl in cases.items()}
    # Deliberately undersized FOV to demonstrate the reliability labels trigger.
    from dataclasses import replace as _replace
    undersized = _replace(config, grid_N=max(64, int(config.grid_N * 0.35)))
    checks["undersized FOV (demo)"] = fov_convergence_check({}, config=undersized)
    names = list(checks)

    fig = plt.figure(figsize=(15.0, 8.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.97, top=0.82, bottom=0.30,
                          hspace=0.45, wspace=0.25)
    fig.suptitle("Stage 8C.3R.1 Crop / FOV Convergence Check  (standard vs expanded grid; free-space n=1.0)",
                 x=0.045, y=0.965, ha="left", va="top", fontsize=15, fontweight="bold")
    _badges(fig, y=0.90)

    xb = np.arange(len(names))
    axp = fig.add_subplot(gs[0, 0])
    axp.bar(xb - 0.2, [checks[n]["standard"]["peak_fluence"] for n in names], 0.4, label="standard", color="#1565c0")
    axp.bar(xb + 0.2, [checks[n]["expanded"]["peak_fluence"] for n in names], 0.4, label="expanded", color="#ef6c00")
    axp.set_xticks(xb); axp.set_xticklabels(names, fontsize=7.5, rotation=15)
    axp.set_ylabel("peak fluence J/cm^2"); axp.set_title("Peak fluence: standard vs expanded FOV", fontsize=10.5, fontweight="bold")
    axp.legend(fontsize=8)

    axr = fig.add_subplot(gs[0, 1])
    axr.bar(xb - 0.2, [checks[n]["peak_fluence_rel_diff"] * 100 for n in names], 0.4, label="peak rel diff %", color="#6a1b9a")
    axr.bar(xb + 0.2, [checks[n]["out_of_frame_fraction"] * 100 for n in names], 0.4, label="out-of-frame %", color="#c62828")
    axr.axhline(5.0, color="0.4", ls="--", lw=1, label="5% convergence")
    axr.set_xticks(xb); axr.set_xticklabels(names, fontsize=7.5, rotation=15)
    axr.set_ylabel("%"); axr.set_title("Convergence & FOV spill", fontsize=10.5, fontweight="bold"); axr.legend(fontsize=8)

    axm = fig.add_subplot(gs[1, :])
    rowvals = np.asarray([[checks[n]["ring_centre_difference_um"] for n in names],
                          [checks[n]["axis_trajectory_difference_um"] for n in names],
                          [checks[n]["raw_peak_position_difference_um"] for n in names]])
    axm.imshow(rowvals, aspect="auto", cmap="Blues")
    axm.set_yticks([0, 1, 2])
    axm.set_yticklabels(["ring centre diff (um)", "axis traj diff (um)", "raw peak diff (um) [DIAG ONLY]"], fontsize=8)
    axm.set_xticks(xb); axm.set_xticklabels(names, fontsize=8)
    for j, n in enumerate(names):
        axm.text(j, 0, f"{checks[n]['ring_centre_difference_um']:.3f}", ha="center", va="center", fontsize=8)
        axm.text(j, 1, f"{checks[n]['axis_trajectory_difference_um']:.2f}", ha="center", va="center", fontsize=8)
        axm.text(j, 2, f"{checks[n]['raw_peak_position_difference_um']:.2f}", ha="center", va="center", fontsize=8, color="#777")
    axm.set_title("Standard-vs-expanded differences (ring/axis drive reliability; raw peak is diagnostic only)",
                  fontsize=10.5, fontweight="bold")

    rows = []
    for n in names:
        c = checks[n]
        rows.append(f"{n:22s} N {c['standard_grid_N']}->{c['expanded_grid_N']}  "
                    f"ringDiff {c['ring_centre_difference_um']:5.3f}um  peakRel {c['peak_fluence_rel_diff']*100:5.2f}%  "
                    f"oof {c['out_of_frame_fraction']*100:5.2f}%  -> {c['metric_convergence_status']}")
    fig.text(0.07, 0.245, "Reliability labels (numerically_reliable / caution_crop_limited / invalid_out_of_frame):",
             fontsize=9.5, fontweight="bold", va="top")
    fig.text(0.07, 0.215, "\n".join(rows), fontsize=8.4, va="top", family="monospace")
    fig.text(0.07, 0.045, "Diagnostic only; free-space n=1.0; no material model; final_export_allowed=False.",
             fontsize=9, va="top", color="#1b5e20", fontweight="bold")

    fig.cpr1_metadata = {"stage": "stage8c3r1_fov_convergence",  # type: ignore[attr-defined]
                         "reliability": {n: checks[n]["metric_convergence_status"] for n in names},
                         "final_export_allowed": False}
    _save(fig, output_path, dpi, "Stage 8C.3R.1 FOV Convergence Check", "stage8c3r1_fov_convergence")
    return fig


# ---------------------------------------------------------------------------
# Stage 8C.3R.2 - annular axis tracking validation
# ---------------------------------------------------------------------------


def _xy_axis_markers(
    ax: Any,
    est: Mapping[str, Any],
    *,
    show_raw_peak_diagnostic: bool = True,
) -> None:
    ax.scatter([0.0], [0.0], marker="+", s=140, c="white", linewidths=2.0,
               label="commanded axis", zorder=5)
    ax.scatter([est["ring_centre_x_um"]], [est["ring_centre_y_um"]], marker="x",
               s=90, c="#00e5ff", linewidths=2.0, label="ring fit", zorder=6)
    ax.scatter([est["core_centre_x_um"]], [est["core_centre_y_um"]], marker="o",
               s=42, facecolors="none", edgecolors="#ff80ab", linewidths=1.8,
               label="core fit", zorder=6)
    if show_raw_peak_diagnostic:
        ax.scatter([est["brightest_pixel_x_um"]], [est["brightest_pixel_y_um"]],
                   marker=".", s=80, c="#ffb300", label="raw peak diag", zorder=7)


def plot_annular_axis_tracking_validation(
    *,
    config: ComponentPlaneConfig | None = None,
    tilt_mrad: float = 24.0,
    axis_estimator_mode: str = "auto",
    show_raw_peak_diagnostic: bool = True,
    output_path: str | Path | None = None,
    dpi: int = 165,
) -> "matplotlib.figure.Figure":
    """C3R.2 validation: fitted annular axis, raw-peak demotion and tilt slope."""
    config = config or ComponentPlaneConfig()
    base = run_component_plane_pipeline({}, config=config)
    tilted = run_component_plane_pipeline(
        {"enable_beam_tilt": True, "beam_tilt_x_mrad": tilt_mrad}, config=config
    )
    b_st = base.propagated_stack
    t_st = tilted.propagated_stack
    x = np.asarray(b_st.x_um, float)
    y = np.asarray(b_st.y_um, float)
    z = np.asarray(b_st.z_um, float)
    sel = int(np.argmax(b_st.intensity_zyx.max(axis=(1, 2))))
    b_f = np.asarray(stack_to_fluence(b_st).fluence_zyx_j_cm2, float)
    t_f = np.asarray(stack_to_fluence(t_st).fluence_zyx_j_cm2, float)
    b_est = estimate_annular_axis(b_st.intensity_zyx[sel], x, y)
    t_est = estimate_annular_axis(t_st.intensity_zyx[sel], x, y)
    traj = track_axis_trajectory(t_st.intensity_zyx, x, y, z, estimator_mode=axis_estimator_mode)
    tilt = validate_beam_tilt(tilt_mrad, config=config)
    y0 = _nearest_index(y, 0.0)
    ext = _ext(x, y)
    ext_xz = (float(z.min()), float(z.max()), float(x.min()), float(x.max()))

    fig = plt.figure(figsize=(16.2, 10.4), facecolor="white")
    gs = fig.add_gridspec(2, 3, left=0.065, right=0.965, top=0.83, bottom=0.08,
                          hspace=0.38, wspace=0.30)
    fig.suptitle("Stage 8C.3R.2 Annular Axis Tracking Validation\n"
                 "Raw annular peak is diagnostic only; fitted ring/core axis drives steering and FOV checks",
                 x=0.045, y=0.972, ha="left", va="top", fontsize=15.5, fontweight="bold")
    _badges(fig, y=0.895)

    vmax_xy = max(float(b_f[sel].max()), float(t_f[sel].max()))
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(b_f[sel], origin="lower", extent=ext, cmap="viridis",
                     vmin=0.0, vmax=vmax_xy, aspect="equal")
    _xy_axis_markers(ax0, b_est.data, show_raw_peak_diagnostic=show_raw_peak_diagnostic)
    ax0.set_title("Clean baseline XY fluence", fontsize=10.5, fontweight="bold")
    ax0.set_xlabel("x (um)"); ax0.set_ylabel("y (um)")
    ax0.legend(fontsize=7.6, loc="upper right", framealpha=0.78)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.02, label="J/cm^2")

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(t_f[sel], origin="lower", extent=ext, cmap="viridis",
                     vmin=0.0, vmax=vmax_xy, aspect="equal")
    _xy_axis_markers(ax1, t_est.data, show_raw_peak_diagnostic=show_raw_peak_diagnostic)
    ax1.annotate("", xy=(t_est["beam_axis_x_um"], t_est["beam_axis_y_um"]), xytext=(0.0, 0.0),
                 arrowprops=dict(arrowstyle="->", color="white", lw=1.8))
    ax1.set_title(f"Tilted XY field ({tilt_mrad:g} mrad)", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("x (um)"); ax1.set_ylabel("y (um)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02, label="J/cm^2")

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(t_f[:, y0, :].T, origin="lower", extent=ext_xz,
                     cmap="viridis", aspect="auto")
    ax2.plot(z, traj["axis_x_by_z_um"], color="#00e5ff", lw=2.0, label="fitted axis")
    zlo, zhi = traj["valid_z_fit_range_um"]
    zz = np.linspace(float(zlo), float(zhi), 50) if np.isfinite(zlo) else z
    ax2.plot(zz, tilt["expected_slope_x"] * zz, "--", color="white", lw=1.8,
             label="expected kx/kz")
    ax2.set_title("Tilted XZ fluence + fitted trajectory", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("z (um)"); ax2.set_ylabel("x (um)")
    ax2.legend(fontsize=8, loc="lower right")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02, label="J/cm^2")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(z, traj["axis_x_by_z_um"], "o", ms=3.6, color="#00838f", label="measured x axis")
    ax3.plot(zz, tilt["expected_slope_x"] * zz, "--", color="#c62828", lw=2.0,
             label="expected slope")
    ax3.axhline(0.0, color="0.65", ls=":", lw=1.0)
    ax3.set_xlabel("z (um)"); ax3.set_ylabel("axis x (um)")
    ax3.set_title("Expected vs measured steering line", fontsize=10.5, fontweight="bold")
    ax3.grid(alpha=0.28); ax3.legend(fontsize=8)

    _card(fig.add_subplot(gs[1, 1]), "Estimator hierarchy and result", [
        f"axis_estimator_mode : {axis_estimator_mode} (auto recommended)",
        "1 ring_fit, 2 core_fit, 3 roi_centroid",
        "4 phase_singularity if complex phase is supplied",
        "raw brightest pixel : diagnostic only",
        f"baseline ring/core  : ({b_est['ring_centre_x_um']:.3f}, {b_est['ring_centre_y_um']:.3f}) / "
        f"({b_est['core_centre_x_um']:.3f}, {b_est['core_centre_y_um']:.3f}) um",
        f"baseline raw peak   : ({b_est['brightest_pixel_x_um']:.2f}, {b_est['brightest_pixel_y_um']:.2f}) um",
        f"ring Q/circularity  : {b_est['ring_fit_quality']:.3f} / {b_est['ring_circularity']:.3f}",
        f"valid z range       : {traj['valid_z_fit_range_um'][0]:.1f} to {traj['valid_z_fit_range_um'][1]:.1f} um",
        f"valid plane fraction: {traj['valid_plane_fraction']:.2f}",
    ], "#e3f2fd", "#0d47a1", fs=7.8)

    _card(fig.add_subplot(gs[1, 2]), "Tilt validation and claim boundary", [
        f"commanded tilt x    : {tilt['commanded_tilt_x_mrad']:.2f} mrad",
        f"expected slope x    : {tilt['expected_slope_x']:.6f}",
        f"measured slope x    : {tilt['measured_slope_x']:.6f}",
        f"absolute error      : {tilt['absolute_error']:.6f}",
        f"relative error      : {tilt['relative_error']*100:.2f} %",
        f"trajectory fit Q    : {tilt['fit_quality']:.3f}",
        f"grid displacement   : {tilt['grid_resolved_displacement']:.1f} px",
        "",
        "free-space n=1.0 optical/fluence diagnostic only",
        "no material response; final_export_allowed=False",
    ], "#e8f5e9", "#1b5e20", fs=7.8)

    fig.c3r2_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3r2_annular_axis_tracking_validation",
        "baseline_axis_error_um": float(b_est["beam_axis_error_um"]),
        "baseline_raw_peak_radius_um": float(np.hypot(b_est["brightest_pixel_x_um"], b_est["brightest_pixel_y_um"])),
        "tilt_relative_error": float(tilt["relative_error"]),
        "axis_estimator_mode": axis_estimator_mode,
        "raw_peak_status": RAW_PEAK_LABEL,
        "final_export_allowed": False,
    }
    _save(fig, output_path, dpi, "Stage 8C.3R.2 Annular Axis Tracking Validation",
          "stage8c3r2_annular_axis_tracking_validation")
    return fig


# ---------------------------------------------------------------------------
# Stage 8C.3R.2 - response curves
# ---------------------------------------------------------------------------


def _reliability_color(status: str) -> str:
    return _RELIAB_COLOR.get(status, "#546e7a")


def plot_individual_response_curves(
    *,
    config: ComponentPlaneConfig | None = None,
    curve_results: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    dpi: int = 155,
) -> "matplotlib.figure.Figure":
    """C3R.2 multi-point response-curve atlas, one panel per perturbation family."""
    config = config or ComponentPlaneConfig()
    families = build_response_curve_families()
    curves = dict(curve_results or {})
    for key in families:
        if key not in curves:
            curves[key] = run_response_curve(key, config=config)

    fig = plt.figure(figsize=(16.8, 12.0), facecolor="white")
    gs = fig.add_gridspec(5, 2, left=0.065, right=0.97, top=0.865, bottom=0.085,
                          hspace=0.70, wspace=0.34)
    fig.suptitle("Stage 8C.3R.2 Individual Diagnostic Response Curves\n"
                 + DIAGNOSTIC_SWEEP_LABEL,
                 x=0.045, y=0.972, ha="left", va="top", fontsize=15.5, fontweight="bold")
    _badges(fig, y=0.91)

    for i, key in enumerate(families):
        res = curves[key]
        rows = list(res.rows)
        xvals = np.array([r["param_value"] for r in rows], dtype=float)
        axis_err = np.array([r["axis_error_um"] for r in rows], dtype=float)
        residual = np.array([r["residual_shape_deformation"] for r in rows], dtype=float)
        loss = 1.0 - np.array([r["transmitted_fraction"] for r in rows], dtype=float)
        core = np.array([r["core_contamination_fraction"] for r in rows], dtype=float)
        ring_loss = 1.0 - np.array([r["ring_fit_quality"] for r in rows], dtype=float)
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax2 = ax.twinx()
        ax.plot(xvals, axis_err, "-o", color="#1565c0", lw=1.9, ms=4.0, label="axis error (um)")
        ax2.plot(xvals, residual, "-s", color="#2e7d32", lw=1.5, ms=3.4, label="residual deformation")
        ax2.plot(xvals, loss, "-^", color="#ef6c00", lw=1.4, ms=3.4, label="throughput loss")
        ax2.plot(xvals, core, "-d", color="#6a1b9a", lw=1.3, ms=3.2, label="core contamination")
        ax2.plot(xvals, ring_loss, ":", color="#455a64", lw=1.5, label="1-ring fit Q")
        ymax = max(float(np.nanmax(axis_err)) * 1.18, 0.5)
        ax.set_ylim(0.0, ymax)
        ax2.set_ylim(0.0, max(0.18, min(1.0, float(max(np.nanmax(residual), np.nanmax(loss), np.nanmax(core), np.nanmax(ring_loss))) * 1.35 + 0.02)))
        ax.set_title(res.family.title, fontsize=10.1, fontweight="bold")
        ax.set_xlabel(res.family.param_label, fontsize=8.5)
        ax.set_ylabel("axis error (um)", fontsize=8.2, color="#1565c0")
        ax2.set_ylabel("fraction / normalized metric", fontsize=8.2)
        ax.grid(alpha=0.24)
        ymin, ymax2 = ax2.get_ylim()
        yr = ymin + 0.94 * (ymax2 - ymin)
        for xv, row in zip(xvals, rows):
            ax2.scatter([xv], [yr], s=28, marker="s",
                        color=_reliability_color(str(row["numerical_reliability"])),
                        edgecolor="white", linewidth=0.4, zorder=8)
        lines = ax.get_lines() + ax2.get_lines()
        labels = [ln.get_label() for ln in lines]
        ax.legend(lines, labels, fontsize=6.9, loc="upper left", framealpha=0.82)
        last = rows[-1]
        ax.text(0.985, 0.04,
                f"E {last['reference_plane_pulse_energy_uJ']:.1f} uJ  "
                f"U {last['azimuthal_uniformity']:.2f}  C {last['ring_circularity']:.2f}\n"
                f"dark {last['central_darkness_contrast']:.2f}  "
                f"peak/E {last['peak_to_reference_energy_ratio']:.2f}  "
                f"{last['numerical_reliability']}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.9,
                bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.82, pad=2.5))

    fig.text(0.066, 0.037,
             "Reliability squares: green=numerically_reliable, orange=caution_crop_limited, red=invalid_out_of_frame. "
             "Curves are free-space optical diagnostics, not laboratory tolerances.",
             fontsize=9, color="#1b5e20", fontweight="bold")
    fig.c3r2_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3r2_individual_response_curves",
        "families": list(families),
        "diagnostic_label": DIAGNOSTIC_SWEEP_LABEL,
        "final_export_allowed": False,
    }
    _save(fig, output_path, dpi, "Stage 8C.3R.2 Individual Response Curves",
          "stage8c3r2_individual_response_curves")
    return fig


# ---------------------------------------------------------------------------
# Stage 8C.3R.2 - free-space summary
# ---------------------------------------------------------------------------


def plot_free_space_study_summary(
    *,
    config: ComponentPlaneConfig | None = None,
    curve_results: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    dpi: int = 165,
) -> "matplotlib.figure.Figure":
    """Compact C3R.2 summary for meetings and notebook inline rendering."""
    config = config or ComponentPlaneConfig()
    base = run_component_plane_pipeline({}, config=config)
    st = base.propagated_stack
    sel = int(np.argmax(st.intensity_zyx.max(axis=(1, 2))))
    est = estimate_annular_axis(st.intensity_zyx[sel], st.x_um, st.y_um)
    tilt = validate_beam_tilt(24.0, config=config)
    fov = fov_convergence_check({}, config=config)
    families = build_response_curve_families()
    curves = dict(curve_results or {})
    for key in families:
        if key not in curves:
            curves[key] = run_response_curve(key, config=config)

    findings = []
    for key, res in curves.items():
        rows = list(res.rows)
        max_axis = max(float(r["axis_error_um"]) for r in rows)
        max_resid = max(float(r["residual_shape_deformation"]) for r in rows)
        max_core = max(float(r["core_contamination_fraction"]) for r in rows)
        max_loss = max(1.0 - float(r["transmitted_fraction"]) for r in rows)
        findings.append((key, max_axis, max_resid, max_core, max_loss))
    top_axis = max(findings, key=lambda t: t[1])
    top_shape = max(findings, key=lambda t: t[2])
    top_core = max(findings, key=lambda t: t[3])
    top_loss = max(findings, key=lambda t: t[4])

    fig = plt.figure(figsize=(15.6, 9.2), facecolor="white")
    gs = fig.add_gridspec(3, 3, left=0.055, right=0.965, top=0.83, bottom=0.08,
                          hspace=0.34, wspace=0.28)
    fig.suptitle("Stage 8C.3R.2 Free-Space Study Summary\n"
                 "Annular axis tracking and diagnostic sensitivity lock",
                 x=0.045, y=0.97, ha="left", va="top", fontsize=15.5, fontweight="bold")
    _badges(fig, y=0.90)

    _card(fig.add_subplot(gs[0, 0]), "Boundary", [
        "free-space reference plane",
        f"n_medium              : {config.n_medium:.1f}",
        "endpoint              : intended sample entrance in air",
        "material/interface    : disabled",
        "writing/3D/GUI        : not started",
        "final_export_allowed  : False",
    ], "#e8f5e9", "#1b5e20", fs=8.0)

    _card(fig.add_subplot(gs[0, 1]), "Represented component planes", [
        "input complex field",
        "SLM phase/device plane",
        "objective pupil aperture/phase",
        "free-space reference entrance",
        "locked angular-spectrum propagation",
        f"z planes              : {config.n_z}",
        f"grid                  : {config.grid_N} x {config.grid_N}, dx={config.dx_um:.2f} um",
    ], "#eceff1", "#37474f", fs=8.0)

    _card(fig.add_subplot(gs[0, 2]), "Axis hierarchy", [
        "1 fitted annular ring centre",
        "2 fitted dark-core centre",
        "3 central ROI centroid",
        "4 phase singularity when complex phase exists",
        "raw brightest pixel   : diagnostic only",
        f"baseline axis error   : {est['beam_axis_error_um']:.3f} um",
        f"baseline raw peak     : ({est['brightest_pixel_x_um']:.2f}, {est['brightest_pixel_y_um']:.2f}) um",
    ], "#e3f2fd", "#0d47a1", fs=7.8)

    _card(fig.add_subplot(gs[1, 0]), "Physically active controls", [
        "vortex/axicon centre offsets",
        "input beam decentre and tilt",
        "finite SLM active area",
        "objective pupil radius/decentre",
        "defocus, astigmatism, coma, spherical",
        "zero-order leakage",
        "passive losses carried into fluence energy",
    ], "#fff3e0", "#bf360c", fs=7.8)

    _card(fig.add_subplot(gs[1, 1]), "Warning-only / future", [
        "no explicit 4F Fourier filter plane",
        "relay imaging errors remain warning-only",
        "mask rotation/apex defect not modelled",
        "jitter/drift ensembles remain future",
        "material/interface/dose/nonlinear/thermal future",
        "no calibrated tolerance claim",
    ], "#f3e5f5", "#4a148c", fs=7.8)

    _card(fig.add_subplot(gs[1, 2]), "Validation state", [
        f"tilt 24 mrad rel err  : {tilt['relative_error']*100:.2f} %",
        f"tilt fit quality      : {tilt['fit_quality']:.3f}",
        f"grid displacement     : {tilt['grid_resolved_displacement']:.1f} px",
        f"FOV status            : {fov['metric_convergence_status']}",
        f"ring diff std/expanded: {fov['ring_centre_difference_um']:.3f} um",
        f"raw peak diff diag    : {fov['raw_peak_position_difference_um']:.2f} um",
    ], "#fffde7", "#f57f17", fs=7.8)

    _card(fig.add_subplot(gs[2, 0]), "Top sensitivity findings", [
        f"largest axis error    : {top_axis[0]} ({top_axis[1]:.2f} um)",
        f"largest shape resid   : {top_shape[0]} ({top_shape[2]:.2f})",
        f"largest core contam   : {top_core[0]} ({top_core[3]:.3f})",
        f"largest throughput loss: {top_loss[0]} ({top_loss[4]:.2f})",
        "diagnostic sensitivity sweep",
        "not measured lab tolerance",
    ], "#fafafa", "#455a64", fs=7.6)

    _card(fig.add_subplot(gs[2, 1]), "Energy accounting", [
        f"input pulse energy    : {st.input_pulse_energy_uJ:.2f} uJ",
        f"reference energy      : {st.reference_plane_pulse_energy_uJ:.2f} uJ",
        f"transmitted fraction  : {st.transmitted_fraction:.3f}",
        "no per-plane renormalisation to pre-clip energy",
        "fluence scaled to surviving reference-plane energy",
    ], "#e0f2f1", "#00695c", fs=7.8)

    _card(fig.add_subplot(gs[2, 2]), "Lock statement", [
        "C3R.2 fixes annular metric validity.",
        "FOV convergence ignores raw ring-peak wandering.",
        "Tilt validation uses fitted axis trajectory.",
        "Response curves are diagnostic rankings only.",
        "Stage 8D remains blocked until this lock is accepted.",
    ], "#ffebee", "#b71c1c", fs=7.8)

    fig.c3r2_metadata = {  # type: ignore[attr-defined]
        "stage": "stage8c3r2_free_space_study_summary",
        "fov_status": fov["metric_convergence_status"],
        "diagnostic_label": DIAGNOSTIC_SWEEP_LABEL,
        "final_export_allowed": False,
    }
    _save(fig, output_path, dpi, "Stage 8C.3R.2 Free-Space Study Summary",
          "stage8c3r2_free_space_study_summary")
    return fig
