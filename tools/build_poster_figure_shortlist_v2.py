"""Build a poster-specific figure shortlist from current Vortex-Bessel code.

This is intentionally NOT a poster compositor.  It exists to curate figures first.
The previous poster attempt over-valued breadth and included several weak figures.
This shortlist keeps only figures that communicate one clear scientific point at
poster scale, and regenerates two wider-code figures from the maintained model:

1. self-healing after a local obstruction;
2. propagation inside fused silica with/without ideal numerical interface correction.

Canonical presentation and q=20 inverse-reconstruction figures are copied without
alteration so measured/model provenance remains intact.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import bessel_twin_core as bt
import publication_diagnostics as pdiag
from vbb_study import vbb_studies
from vbb_study.config import MaterialConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "poster" / "figure_shortlist_v2"
DARK = "#070a0d"
FG = "#f1f4f6"
MUTED = "#aeb8c1"
CYAN = "#48d7cf"
GOLD = "#ffd166"
THERMAL = "inferno"


def norm(a: np.ndarray, scale: float | None = None) -> np.ndarray:
    arr = np.maximum(np.asarray(a, dtype=float), 0.0)
    denom = float(np.max(arr)) if scale is None else float(scale)
    return arr / max(denom, np.finfo(float).tiny)


def style_dark(ax: plt.Axes) -> None:
    ax.set_facecolor(DARK)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#59636d")
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)


def save(fig: plt.Figure, path: Path, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def crop_plane(plane: np.ndarray, full_grid: dict, crop_grid: dict) -> np.ndarray:
    x_full = np.asarray(full_grid["x"], float)
    x_crop = np.asarray(crop_grid["x"], float)
    ids = np.flatnonzero((x_full >= x_crop[0]) & (x_full <= x_crop[-1]))
    if ids.size < 4:
        return np.asarray(plane, float)
    return np.asarray(plane, float)[np.ix_(ids, ids)]


def build_self_healing(out: Path) -> Path:
    """Poster-quality before -> blocked -> reconstruction sequence."""
    cfg = bt.default_config("balanced")
    cfg = vbb_studies.beam_air_config(cfg)
    cfg = replace(cfg, target=replace(cfg.target, ell=0))
    design = bt.compute_design_from_config(cfg)

    # Obstruction deliberately blocks most of the central core but not the ring reservoir.
    radius = 0.90 * float(design.equivalent_l0_first_zero_radius_m)
    bundle = pdiag.build_self_healing_bundle(
        config=cfg,
        preset="balanced",
        path="ideal",
        case_id="poster_B0_self_healing",
        obstacle_kind="disk",
        obstacle_radius_m=radius,
        axial_points=81,
    )
    vol = bundle["obstructed_volume"]
    cgrid = vol["crop_grid"]
    x_um = np.asarray(cgrid["x"], float) / bt.um
    z_um = np.asarray(bundle["z_relative"], float) / bt.um
    ref = crop_plane(bundle["reference_plane"], bundle["grid"], cgrid)
    blocked = crop_plane(bundle["obstructed_plane"], bundle["grid"], cgrid)
    stack = np.asarray(vol["intensity_stack"], float)

    picks = [0, int(0.18 * (len(z_um) - 1)), int(0.42 * (len(z_um) - 1)), int(0.75 * (len(z_um) - 1))]
    planes = [ref, blocked] + [stack[i] for i in picks[1:]]
    labels = [
        "reference",
        "local obstruction",
        f"z = {z_um[picks[1]]:.0f} µm",
        f"z = {z_um[picks[2]]:.0f} µm",
        f"z = {z_um[picks[3]]:.0f} µm",
    ]
    shared = float(np.max(ref))
    extent = [x_um[0], x_um[-1], x_um[0], x_um[-1]]

    fig = plt.figure(figsize=(15.5, 7.0), facecolor=DARK)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.62], hspace=0.30, wspace=0.16)
    image = None
    for i, (plane, label) in enumerate(zip(planes, labels)):
        ax = fig.add_subplot(gs[0, i])
        style_dark(ax)
        image = ax.imshow(norm(plane, shared) ** 0.46, origin="lower", extent=extent, cmap=THERMAL, vmin=0, vmax=1)
        ax.set_title(label, fontsize=12, weight="bold")
        ax.set_xlabel("x (µm)")
        if i == 0:
            ax.set_ylabel("y (µm)")
        else:
            ax.set_yticklabels([])
        ax.grid(False)

    ax = fig.add_subplot(gs[1, :])
    style_dark(ax)
    ax.plot(z_um, bundle["peak_recovery"], color=CYAN, lw=2.6, label="peak recovery")
    ax.plot(z_um, bundle["onaxis_recovery"], color=GOLD, lw=2.2, ls="--", label="on-axis recovery")
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=":")
    ax.set_xlabel("distance after obstruction (µm)")
    ax.set_ylabel("obstructed / unobstructed")
    ax.set_ylim(0, min(1.45, max(1.15, 1.05 * float(np.nanmax(bundle["peak_recovery"])))))
    ax.grid(alpha=0.13)
    ax.legend(frameon=False, ncol=2, labelcolor=FG, loc="lower right")
    fig.suptitle("Self-healing of a Bessel field after a local obstruction", color=FG, fontsize=18, weight="bold", y=0.985)
    fig.text(0.5, 0.925, "Same intensity normalisation in every transverse panel", ha="center", color=MUTED, fontsize=10)
    return save(fig, out / "01_self_healing_sequence.png")


def build_interface_comparison(out: Path) -> Path:
    """Show sample-side propagation only; avoid the confusing stitched-air figure."""
    cfg = bt.default_config("balanced")
    cfg = replace(
        cfg,
        material=MaterialConfig.fused_silica(write_depth_m=300 * bt.um),
        target=replace(cfg.target, ell=1),
        apply_interface=True,
        correct_interface=False,
    )
    air = vbb_studies.build_beam_to_surface_result(cfg, path="ideal", z_values_m=None)
    surface = air["surface_field"]
    unc = vbb_studies.run_through_sample(surface, cfg, correct_interface=False)
    cor = vbb_studies.run_through_sample(surface, cfg, correct_interface=True)

    vu = unc.volume_result
    vc = cor.volume_result
    zu = np.asarray(vu["z"], float) / bt.um
    zc = np.asarray(vc["z"], float) / bt.um
    xu = np.asarray(vu["crop_grid"]["x"], float) / bt.um
    xc = np.asarray(vc["crop_grid"]["x"], float) / bt.um
    target = float(unc.metrics["target_write_depth_um"])
    iu = int(np.argmin(np.abs(zu - target)))
    ic = int(np.argmin(np.abs(zc - target)))
    pu = np.asarray(vu["intensity_stack"][iu], float)
    pc = np.asarray(vc["intensity_stack"][ic], float)
    xzu = np.asarray(vu["xz"], float)
    xzc = np.asarray(vc["xz"], float)
    shared_xy = max(float(np.max(pu)), float(np.max(pc)))
    shared_xz = max(float(np.max(xzu)), float(np.max(xzc)))

    fig = plt.figure(figsize=(13.8, 8.4), facecolor=DARK)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.18], hspace=0.22, wspace=0.14)
    panels = [
        (fig.add_subplot(gs[0, 0]), pu, "uncorrected interface", xu, zu, False),
        (fig.add_subplot(gs[0, 1]), pc, "ideal numerical correction diagnostic", xc, zc, False),
        (fig.add_subplot(gs[1, 0]), xzu, "uncorrected propagation", xu, zu, True),
        (fig.add_subplot(gs[1, 1]), xzc, "corrected-model propagation", xc, zc, True),
    ]
    for idx, (ax, arr, title, x, z, is_xz) in enumerate(panels):
        style_dark(ax)
        if is_xz:
            ax.imshow(norm(arr, shared_xz) ** 0.44, origin="lower", aspect="auto", extent=[z[0], z[-1], x[0], x[-1]], cmap=THERMAL, vmin=0, vmax=1)
            ax.axvline(target, color=GOLD, ls="--", lw=1.4)
            ax.set_xlabel("depth in fused silica (µm)")
            ax.set_ylabel("x (µm)" if idx == 2 else "")
        else:
            ax.imshow(norm(arr, shared_xy) ** 0.44, origin="lower", extent=[x[0], x[-1], x[0], x[-1]], cmap=THERMAL, vmin=0, vmax=1)
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)" if idx == 0 else "")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.grid(False)

    fig.suptitle("Model extension into the sample: field propagation across the air–silica interface", color=FG, fontsize=17, weight="bold", y=0.985)
    fig.text(0.5, 0.945, "Top: transverse field at the target write depth. Bottom: longitudinal propagation from the sample surface.", ha="center", color=MUTED, fontsize=10.5)
    fig.text(0.5, 0.018, "The correction branch is an ideal numerical diagnostic, not a hardware-validated correction.", ha="center", color="#ffb4ae", fontsize=10.5, weight="bold")
    return save(fig, out / "02_sample_interface_comparison.png")


def copy_curated(out: Path) -> dict[str, Path]:
    """Copy only strong existing figures; do not regenerate measured evidence."""
    files = {
        "03_computational_route.png": ROOT / "figures/presentation/01_computational_route_phase2j.png",
        "04_ideal_beam_family.png": ROOT / "figures/presentation/02_beam_profile_shaping_B0_V1_V3_thermal_tight.png",
        "05_axicon_decentre.png": ROOT / "figures/presentation/04_V1_axicon_decentre_fixed_lab_thermal_tight.png",
        "06_nonideal_apex.png": ROOT / "figures/presentation/05_V1_nonideal_tip_fixed_lab_thermal_tight.png",
        "07_retrieved_aberration_phase.png": ROOT / "figures/experimental/q20_aberration/reconstruction/annular_aberration_phase.png",
        "08_measured_fit_corrected_ideal.png": ROOT / "figures/experimental/q20_aberration/reconstruction/realigned_cartesian_xy_measured_fit_corrected_ideal.png",
        "09_single_mask_metrics_vs_z.png": ROOT / "figures/experimental/q20_aberration/single_mask/single_mask_metrics_vs_z.png",
        "10_measured_vs_corrected_3d.png": ROOT / "figures/experimental/q20_aberration/validation/measured_vs_corrected_3d_mesh.png",
    }
    copied = {}
    for name, src in files.items():
        if not src.exists():
            raise FileNotFoundError(src)
        dst = out / name
        shutil.copy2(src, dst)
        copied[name] = dst
    return copied


def contact_sheet(out: Path, paths: list[Path]) -> Path:
    fig = plt.figure(figsize=(18, 20), facecolor="#171b1f")
    gs = fig.add_gridspec(5, 2, hspace=0.20, wspace=0.10)
    for i, p in enumerate(paths[:10]):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_facecolor("#171b1f")
        img = plt.imread(p)
        ax.imshow(img)
        ax.set_title(p.stem.replace("_", " "), color="white", fontsize=12, weight="bold")
        ax.axis("off")
    fig.suptitle("Poster figure shortlist v2 — curated before layout", color="white", fontsize=22, weight="bold", y=0.995)
    fig.text(0.5, 0.006, "No sensitivity bar chart. No jagged V3 3-D surface. No stitched source-to-sample artefact.", ha="center", color="#ffb4ae", fontsize=12, weight="bold")
    return save(fig, out / "00_shortlist_contact_sheet.png", dpi=180)


def main() -> None:
    out = DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    generated = [build_self_healing(out), build_interface_comparison(out)]
    copied = copy_curated(out)
    ordered = generated + [copied[k] for k in copied]
    sheet = contact_sheet(out, ordered)
    print(sheet)
    for p in ordered:
        print(p)


if __name__ == "__main__":
    main()
