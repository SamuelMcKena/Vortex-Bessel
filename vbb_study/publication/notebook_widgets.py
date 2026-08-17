"""Interactive ipywidgets helpers for exploratory notebook sessions.

These widgets are intended for local parameter exploration only.  They run
fast (preset='fast') cases and do NOT update any locked-stage outputs.

Usage in a notebook cell::

    from vbb_study.publication import notebook_widgets as nbw
    panel = nbw.interactive_quicklook(base, method='physical')
    display(panel)
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import ipywidgets as widgets
    from IPython.display import display as _ipy_display
    _WIDGETS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WIDGETS_AVAILABLE = False


_EPS = 1.0e-30


def _apply_param_overrides(base, **kw) -> Any:
    """Return a new TwinConfig derived from *base* with the given param overrides."""
    from vbb_study.config import um
    cfg = base
    if "ell" in kw:
        cfg = replace(cfg, target=replace(cfg.target, ell=int(kw["ell"])))
    if "target_core_diameter_um" in kw:
        cfg = replace(cfg, target=replace(cfg.target, target_core_diameter_m=float(kw["target_core_diameter_um"]) * um))
    if "target_bessel_length_um" in kw:
        cfg = replace(cfg, target=replace(cfg.target, target_bessel_length_m=float(kw["target_bessel_length_um"]) * um))
    if "objective_NA" in kw:
        cfg = replace(cfg, objective=replace(cfg.objective, NA=float(kw["objective_NA"])))
    if "blaze_period_px" in kw:
        cfg = replace(cfg, slm=replace(cfg.slm, blaze_period_px=int(kw["blaze_period_px"])))
    if "slm2_stroke_levels" in kw:
        lvl = kw["slm2_stroke_levels"]
        cfg = replace(cfg, physical_axicon=replace(cfg.physical_axicon, slm2_stroke_levels=None if lvl == 0 else int(lvl)))
    return cfg


def _run_and_plot(base, method, preset, param_kw, out_widget) -> None:
    """Run a quick case and refresh the output widget with diagnostic plots."""
    from vbb_study.config import um
    import bessel_twin_core as bt
    from vbb_study import vbb_style

    cfg = _apply_param_overrides(base, **param_kw)
    try:
        result = bt.run_case(cfg, preset=preset, path="ideal", case_id="quicklook")
    except Exception as exc:
        with out_widget:
            out_widget.clear_output(wait=True)
            print(f"[quicklook] run_case failed: {exc}")
        return

    metrics = result.get("metrics", {})
    volume = result.get("volume", {})
    design = result.get("design")

    vbb_style.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)

    # --- Axial trace ---
    ax = axes[0]
    z_um = np.asarray(volume.get("z", []), dtype=float) / um
    peak = np.asarray(volume.get("peak", []), dtype=float)
    if z_um.size and peak.size:
        peak_n = peak / (float(np.max(peak)) + _EPS)
        ax.plot(z_um, peak_n, color="#0072B2", lw=2)
        ax.axhline(0.5, color="0.4", ls=":", lw=1)
        zone_s = metrics.get("zone_start_um")
        zone_e = metrics.get("zone_end_um")
        if zone_s is not None and zone_e is not None:
            ax.axvspan(zone_s, zone_e, color="#009E73", alpha=0.18, label=f"FWHM {metrics.get('bessel_zone_um', 0):.0f} um")
            ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("z [um]")
    ax.set_ylabel("norm. peak intensity")
    ax.set_title("axial trace")

    # --- Radial profile at peak plane ---
    ax = axes[1]
    crop_grid = volume.get("crop_grid")
    slices = volume.get("slices")
    peak_idx = int(volume.get("peak_index", 0))
    if slices is not None and crop_grid is not None:
        try:
            I_slice = np.asarray(slices[peak_idx], dtype=float)
            x_g = np.asarray(crop_grid.get("x", []), dtype=float) / um
            cx, cy = I_slice.shape[1] // 2, I_slice.shape[0] // 2
            row = I_slice[cy, :]
            row_n = row / (float(np.max(row)) + _EPS)
            ax.plot(x_g, row_n, color="#D55E00", lw=2)
            fd = metrics.get("feature_diameter_um")
            if fd is not None:
                ax.axvspan(-fd / 2, fd / 2, color="#E69F00", alpha=0.2, label=f"core {fd:.2f} um")
                ax.legend(frameon=False, fontsize=8)
        except (IndexError, TypeError):
            ax.text(0.5, 0.5, "no slice data", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, "no radial data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("x [um]")
    ax.set_ylabel("norm. intensity")
    ax.set_title(f"radial profile @ peak  (ell={int(design.ell if design else 0)})")

    # --- Metrics table ---
    ax = axes[2]
    ax.axis("off")
    metric_labels = [
        ("feature_diameter_um",     "core diam [um]"),
        ("bessel_zone_um",          "Bessel zone [um]"),
        ("peak_fluence_J_cm2",      "peak fluence [J/cm²]"),
        ("side_to_core_peak_ratio", "side/core ratio"),
        ("canonical_zone_um",       "canonical zone [um]"),
        ("strict_bessel_region_um", "strict zone [um]"),
    ]
    rows = []
    for key, label in metric_labels:
        val = metrics.get(key)
        if val is None:
            txt = "—"
        elif isinstance(val, float):
            txt = f"{val:.3g}"
        else:
            txt = str(val)
        rows.append([label, txt])
    if rows:
        tbl = ax.table(
            cellText=rows,
            colLabels=["metric", "value"],
            loc="center",
            cellLoc="left",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 1.45)
    valid = result.get("validity_report", {}).get("valid", True)
    ax.set_title(f"metrics  (valid: {valid})", fontsize=9)

    with out_widget:
        out_widget.clear_output(wait=True)
        plt.show()


def interactive_quicklook(
    base,
    *,
    method: str = "holographic",
    preset: str = "fast",
    ell_range: tuple[int, int] = (0, 6),
    core_um_range: tuple[float, float] = (0.5, 10.0),
    zone_um_range: tuple[float, float] = (25.0, 600.0),
    na_range: tuple[float, float] = (0.1, 0.8),
    blaze_range: tuple[int, int] = (8, 40),
) -> Any:
    """Return an ipywidgets panel for interactive beam parameter exploration.

    Parameters
    ----------
    base:
        A ``TwinConfig`` to use as the starting point.  Edits are non-destructive.
    method:
        ``'holographic'`` or ``'physical'``.  Controls which sliders are shown.
    preset:
        Simulation preset (``'fast'`` is recommended for interactivity).

    Returns
    -------
    ``ipywidgets.VBox`` — call ``display(panel)`` in the notebook.
    """
    if not _WIDGETS_AVAILABLE:
        raise ImportError(
            "ipywidgets is not installed.  Run: pip install ipywidgets"
        )

    from vbb_study.config import um

    # Read starting values from base config
    start_ell = int(getattr(base.target, "ell", 3))
    start_core = float(getattr(base.target, "target_core_diameter_m", 3.0 * um)) / um
    start_zone = float(getattr(base.target, "target_bessel_length_m", 150.0 * um)) / um
    start_na = float(getattr(base.objective, "NA", 0.45))
    start_blaze = int(getattr(base.slm, "blaze_period_px", 20))
    start_stroke = int(getattr(base.physical_axicon, "slm2_stroke_levels", 0) or 0)

    # --- Widgets ---
    w_ell = widgets.IntSlider(
        value=start_ell,
        min=ell_range[0], max=ell_range[1], step=1,
        description="ell:",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="380px"),
    )
    w_core = widgets.FloatSlider(
        value=round(start_core, 2),
        min=core_um_range[0], max=core_um_range[1], step=0.25,
        description="core diam [um]:",
        readout_format=".2f",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="380px"),
    )
    w_zone = widgets.FloatSlider(
        value=round(start_zone, 1),
        min=zone_um_range[0], max=zone_um_range[1], step=10.0,
        description="zone length [um]:",
        readout_format=".0f",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="380px"),
    )
    w_na = widgets.FloatSlider(
        value=round(start_na, 2),
        min=na_range[0], max=na_range[1], step=0.05,
        description="objective NA:",
        readout_format=".2f",
        style={"description_width": "120px"},
        layout=widgets.Layout(width="380px"),
    )

    run_btn = widgets.Button(
        description="Update plots",
        button_style="primary",
        layout=widgets.Layout(width="150px", height="32px"),
    )
    out = widgets.Output()

    left_col = [w_ell, w_core, w_zone, w_na]
    right_widgets = []

    if method == "holographic":
        w_blaze = widgets.IntSlider(
            value=start_blaze,
            min=blaze_range[0], max=blaze_range[1], step=2,
            description="blaze period [px]:",
            style={"description_width": "120px"},
            layout=widgets.Layout(width="380px"),
        )
        right_widgets.append(w_blaze)
    else:
        w_blaze = None

    if method == "physical":
        w_stroke = widgets.Dropdown(
            options=[("ideal (continuous)", 0), ("64 levels", 64), ("128 levels", 128), ("256 levels", 256)],
            value=start_stroke if start_stroke in (0, 64, 128, 256) else 0,
            description="SLM2 levels:",
            style={"description_width": "120px"},
            layout=widgets.Layout(width="380px"),
        )
        right_widgets.append(w_stroke)
    else:
        w_stroke = None

    header = widgets.HTML(
        "<b>Quicklook — exploratory only (preset='fast', path='ideal')</b>"
        "<br><span style='color:#888;font-size:11px;'>"
        "Changes here do not update any saved outputs.  "
        "Click <i>Update plots</i> after adjusting sliders.</span>"
    )

    right_col = right_widgets + [run_btn]
    left_box = widgets.VBox(left_col)
    right_box = widgets.VBox(right_col)
    controls_row = widgets.HBox([left_box, right_box])

    def _collect_params() -> dict:
        p: dict = {
            "ell": w_ell.value,
            "target_core_diameter_um": w_core.value,
            "target_bessel_length_um": w_zone.value,
            "objective_NA": w_na.value,
        }
        if w_blaze is not None:
            p["blaze_period_px"] = w_blaze.value
        if w_stroke is not None:
            p["slm2_stroke_levels"] = w_stroke.value
        return p

    def _on_run(_btn=None):
        _run_and_plot(base, method, preset, _collect_params(), out)

    run_btn.on_click(_on_run)

    # Auto-run once on creation so the output area is not empty
    _on_run()

    return widgets.VBox([header, controls_row, out])


def _no_widgets_stub(message: str) -> Any:
    """Return a plain-text fallback when ipywidgets is not available."""
    try:
        from IPython.display import HTML
        return HTML(f"<pre style='color:red'>{message}</pre>")
    except ImportError:
        return message


__all__ = [
    "interactive_quicklook",
]
