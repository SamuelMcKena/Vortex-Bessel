"""Poster error-study curation v2.

The v1 contact sheet showed that many physical perturbations become visually
indistinguishable if every case is reduced to the same final XY plane.  This
pass therefore shows the plane at which the perturbation enters the model next
to its downstream fixed-laboratory XZ signature.

Individual candidate figures are exported at 500 dpi and as vector PDF.  The
contact sheets are lower-resolution review aids only and are not final poster
assets.

A separate contact sheet collects existing tracked q=20 experimental inverse
figures so that the physical-error fitting result can later be joined to the
*actual* residual-phase correction work without mislabelling a model prediction
as a measured post-correction beam.
"""
from __future__ import annotations

from pathlib import Path
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import numpy as np
from PIL import Image, ImageOps, ImageDraw

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_system_error_sweeps import system_sweep_registry
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig, build_system_route

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "error_dictionary_v2"
EPS = np.finfo(float).tiny

BG = "#070a0d"
AXBG = "#0b0f14"
FG = "#f2f3f4"
MUTED = "#aab4be"
CYAN = "#4dd9d5"

THERMAL = LinearSegmentedColormap.from_list(
    "poster_thermal_v2",
    [
        (0.00, "#000000"),
        (0.10, "#090000"),
        (0.30, "#4d0000"),
        (0.52, "#b51d00"),
        (0.74, "#f56c00"),
        (0.90, "#ffc21a"),
        (1.00, "#fff5bd"),
    ],
    N=256,
)

Z = np.linspace(20e-3, 100e-3, 33)
XZ_COORD = np.linspace(-1.15e-3, 1.15e-3, 481)

# Broader than the final poster will need.  This is deliberately a shootout.
CASES = [
    ("beam_lateral_decentre_x", 500e-6, "Input beam decentre"),
    ("beam_radius_scale", 0.70, "Input beam radius"),
    ("beam_ellipticity", 1.30, "Input beam ellipticity"),
    ("slm1_hologram_offset_x", 200e-6, "SLM1 hologram registration"),
    ("fourf_iris_offset_x", 0.60e-3, "4F iris offset"),
    ("fourf_iris_radius_scale", 0.70, "4F iris opening"),
    ("fourf_lens1_despace", 10e-3, "4F lens 1 despace"),
    ("axicon_lateral_decentre_x", 500e-6, "Axicon decentre"),
    ("axicon_rigid_tilt_x", math.radians(0.50), "Axicon tilt"),
    ("axicon_round_tip", 20e-6, "Rounded axicon tip"),
    ("axicon_flat_tip", 100e-6, "Flat axicon tip"),
]

Q20_FILES = [
    ("figures/experimental/q20_aberration/reconstruction/polar_measured_fit_corrected.png",
     "Polar measured / fitted / model-corrected"),
    ("figures/experimental/q20_aberration/reconstruction/realigned_cartesian_xy_measured_fit_corrected_ideal.png",
     "Cartesian measured / fitted / model-corrected / ideal"),
    ("figures/experimental/q20_aberration/single_mask/single_z_double_confirmation_minus10.png",
     "Single-plane inverse / forward falsification test"),
    ("figures/experimental/q20_aberration/phase_error_recreation/phase_error_recreation_agreement_vs_z.png",
     "Recovered-phase recreation versus z"),
]


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(AXBG)
    for spine in ax.spines.values():
        spine.set_color("#56626e")
        spine.set_linewidth(0.7)
    ax.tick_params(colors=MUTED, labelsize=8.0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(FG)
    ax.grid(False)


def _save_final(fig: plt.Figure, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=500, bbox_inches="tight", facecolor=fig.get_facecolor())
    # Text/axes remain vector in the PDF; image arrays are embedded at their
    # native simulation resolution rather than being resampled to a screenshot.
    fig.savefig(pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return png, pdf


def _save_review(fig: plt.Figure, path: Path, dpi: int = 180) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def _norm(a: np.ndarray, peak: float | None = None) -> np.ndarray:
    arr = np.maximum(np.asarray(a, float), 0.0)
    p = float(np.max(arr)) if peak is None else float(peak)
    return arr / max(p, EPS)


def _format_value(family: str, value: float) -> str:
    if family in {"beam_lateral_decentre_x", "slm1_hologram_offset_x", "axicon_lateral_decentre_x"}:
        return f"{value*1e6:+.0f} µm"
    if family == "fourf_iris_offset_x":
        return f"{value*1e3:+.2f} mm"
    if family == "fourf_lens1_despace":
        return f"{value*1e3:+.0f} mm"
    if family == "axicon_rigid_tilt_x":
        return f"{math.degrees(value):+.2f}°"
    if family in {"axicon_round_tip", "axicon_flat_tip"}:
        return f"{value*1e6:.0f} µm"
    if family in {"beam_radius_scale", "beam_ellipticity", "fourf_iris_radius_scale"}:
        return f"{value:.2f} × nominal"
    return f"{value:g}"


def _propagation_xz(route: dict) -> np.ndarray:
    prop = build_fixed_support_spectrum(
        np.asarray(route["post_axicon"], np.complex128),
        dict(route["grid"]),
        wavelength_m=float(route["metadata"]["wavelength_m"]),
        z_max_m=float(Z[-1]),
        minimum_retained_spectral_power=0.995,
    )
    x_native = np.asarray(route["grid"]["x"], float)
    iy0 = int(np.argmin(np.abs(x_native)))
    rows = []
    for z in Z:
        field = native_field_at_z(prop, float(z))
        line = np.abs(np.asarray(field, np.complex128)[iy0, :])**2
        rows.append(np.interp(XZ_COORD, x_native, line, left=0.0, right=0.0))
    return np.asarray(rows, float).T


def _circle_pixels(ax: plt.Axes, centre_m: tuple[float,float], radius_m: float) -> None:
    # Coordinates on Fourier-plane plots are expressed relative to nominal +1.
    ax.add_patch(Circle((centre_m[0]*1e3, centre_m[1]*1e3), radius_m*1e3,
                        fill=False, ec="white", lw=1.0, ls="--", alpha=0.75))


def _component_panel(
    family: str,
    nominal: dict,
    route: dict,
) -> tuple[np.ndarray, list[float], str, str, dict]:
    """Return physically relevant component evidence and display metadata."""
    x = np.asarray(route["grid"]["x"], float)
    X = np.asarray(route["grid"]["X"], float)
    Y = np.asarray(route["grid"]["Y"], float)
    extra: dict = {}

    if family.startswith("beam_"):
        arr = np.abs(route["input_beam"])**2
        ref = np.abs(nominal["input_beam"])**2
        half = 3.2e-3
        ids = np.flatnonzero(np.abs(x) <= half)
        arr = arr[np.ix_(ids, ids)]
        ref = ref[np.ix_(ids, ids)]
        extent = [x[ids[0]]*1e3, x[ids[-1]]*1e3, x[ids[0]]*1e3, x[ids[-1]]*1e3]
        extra["reference_contour"] = _norm(ref, max(float(np.max(ref)), float(np.max(arr))))
        return _norm(arr, max(float(np.max(ref)), float(np.max(arr)))), extent, "Input beam at SLM1", "intensity", extra

    if family.startswith("slm1_"):
        # For a hologram registration error, the phase change on the panel itself
        # is much more direct than a nearly unchanged final transverse image.
        phase = np.angle(np.asarray(route["post_slm1"]) * np.conj(np.asarray(nominal["post_slm1"])))
        amp = np.abs(nominal["input_beam"])
        phase = np.where(amp >= 0.08*float(np.max(amp)), phase, np.nan)
        half = 2.6e-3
        ids = np.flatnonzero(np.abs(x) <= half)
        arr = phase[np.ix_(ids, ids)]
        extent = [x[ids[0]]*1e3, x[ids[-1]]*1e3, x[ids[0]]*1e3, x[ids[-1]]*1e3]
        return arr, extent, "Change in SLM1 phase pattern", "phase", extra

    if family.startswith("fourf_iris"):
        pre = np.asarray(route["fourier_plane_before_iris"], np.complex128)
        arr = np.abs(pre * np.asarray(route["fourier_iris_mask"], float))**2
        meta = route["metadata"]["fourf"]
        nominal_c = tuple(map(float, nominal["metadata"]["fourf"]["nominal_selected_order_centre_m"]))
        centre = tuple(map(float, meta["physical_iris_centre_m"]))
        radius = float(meta["iris_radius_m"])
        # Re-centre display coordinates on the nominal selected +1 order.
        xx = x - nominal_c[0]
        yy = x - nominal_c[1]
        half = max(1.2e-3, 2.0*radius)
        ix = np.flatnonzero(np.abs(xx) <= half)
        iy = np.flatnonzero(np.abs(yy) <= half)
        cropped = arr[np.ix_(iy, ix)]
        extent = [xx[ix[0]]*1e3, xx[ix[-1]]*1e3, yy[iy[0]]*1e3, yy[iy[-1]]*1e3]
        extra.update({
            "iris_circle": ((centre[0]-nominal_c[0], centre[1]-nominal_c[1]), radius),
            "nominal_iris_circle": ((0.0, 0.0), float(nominal["metadata"]["fourf"]["iris_radius_m"])),
        })
        return _norm(cropped), extent, "Selected +1 order at Fourier iris", "intensity", extra

    if family.startswith("fourf_"):
        arr = np.abs(route["fourier_plane_before_iris"])**2
        meta = route["metadata"]["fourf"]
        nominal_c = tuple(map(float, nominal["metadata"]["fourf"]["nominal_selected_order_centre_m"]))
        radius = float(nominal["metadata"]["fourf"]["iris_radius_m"])
        xx = x - nominal_c[0]
        yy = x - nominal_c[1]
        half = max(1.4e-3, 2.2*radius)
        ix = np.flatnonzero(np.abs(xx) <= half)
        iy = np.flatnonzero(np.abs(yy) <= half)
        cropped = arr[np.ix_(iy, ix)]
        extent = [xx[ix[0]]*1e3, xx[ix[-1]]*1e3, yy[iy[0]]*1e3, yy[iy[-1]]*1e3]
        extra["nominal_iris_circle"] = ((0.0, 0.0), radius)
        return _norm(cropped)**0.42, extent, "Fourier plane before the iris", "intensity", extra

    # Axicon families: show the change in the axicon phase transmission itself.
    def transmission(r: dict) -> np.ndarray:
        inc = np.asarray(r["field_on_axicon_plane"], np.complex128)
        out = np.asarray(r["post_axicon_local"], np.complex128)
        return out / np.where(np.abs(inc) > 1e-10, inc, 1.0)
    delta = np.angle(transmission(route) * np.conj(transmission(nominal)))
    inc = np.abs(route["field_on_axicon_plane"])
    delta = np.where(inc >= 0.06*float(np.max(inc)), delta, np.nan)
    half = 1.35e-3 if "tip" not in family else 0.45e-3
    ids = np.flatnonzero(np.abs(x) <= half)
    arr = delta[np.ix_(ids, ids)]
    extent = [x[ids[0]]*1e3, x[ids[-1]]*1e3, x[ids[0]]*1e3, x[ids[-1]]*1e3]
    return arr, extent, "Change in axicon phase transmission", "phase", extra


def build_candidates(out: Path, *, grid_n: int = 384) -> tuple[list[dict], list[Path]]:
    reg = system_sweep_registry()
    nominal = build_system_route("V1", grid_n=grid_n, config=SystemErrorConfig())
    nominal_xz = _propagation_xz(nominal)
    records: list[dict] = []
    png_paths: list[Path] = []
    for family, value, title in CASES:
        route = build_system_route("V1", grid_n=grid_n, config=reg[family]["builder"](float(value)))
        xz = _propagation_xz(route)
        component, comp_extent, comp_title, comp_kind, extra = _component_panel(family, nominal, route)
        xz_peak = max(float(np.max(nominal_xz)), float(np.max(xz)), EPS)
        nxz = _norm(nominal_xz, xz_peak)
        pxz = _norm(xz, xz_peak)
        xz_extent = [Z[0]*1e3, Z[-1]*1e3, XZ_COORD[0]*1e3, XZ_COORD[-1]*1e3]

        fig = plt.figure(figsize=(12.9, 4.7), facecolor=BG)
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.22, 1.22], wspace=0.20,
                              left=0.065, right=0.985, top=0.77, bottom=0.15)

        axc = fig.add_subplot(gs[0,0]); _style(axc)
        if comp_kind == "phase":
            im = axc.imshow(component, origin="lower", extent=comp_extent, cmap="twilight_shifted",
                             vmin=-np.pi, vmax=np.pi, interpolation="nearest", aspect="equal")
            cb = fig.colorbar(im, ax=axc, fraction=0.047, pad=0.03)
            cb.set_ticks([-np.pi, 0, np.pi], labels=["−π", "0", "π"])
            cb.ax.tick_params(colors=MUTED, labelsize=7.2)
            cb.outline.set_edgecolor("#56626e")
        else:
            axc.imshow(component**0.55, origin="lower", extent=comp_extent, cmap=THERMAL,
                       vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            if "reference_contour" in extra:
                ref = extra["reference_contour"]
                axc.contour(ref, levels=[np.exp(-2)], colors="white", linewidths=0.8,
                            alpha=0.65, extent=comp_extent, origin="lower")
        if "nominal_iris_circle" in extra:
            c, r = extra["nominal_iris_circle"]
            axc.add_patch(Circle((c[0]*1e3,c[1]*1e3), r*1e3, fill=False,
                                 ec="white", lw=0.9, ls="--", alpha=0.55))
        if "iris_circle" in extra:
            c, r = extra["iris_circle"]
            axc.add_patch(Circle((c[0]*1e3,c[1]*1e3), r*1e3, fill=False,
                                 ec=CYAN, lw=1.25, alpha=0.9))
        axc.set_title(comp_title, fontsize=10.5, weight="bold")
        axc.set_xlabel("x (mm)")
        axc.set_ylabel("y (mm)")

        for col, arr, ptitle in [(1,nxz,"Nominal propagation"),(2,pxz,"Perturbed propagation")]:
            ax = fig.add_subplot(gs[0,col]); _style(ax)
            ax.imshow(arr**0.48, origin="lower", aspect="auto", extent=xz_extent,
                      cmap=THERMAL, vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(ptitle, fontsize=10.5, weight="bold")
            ax.set_xlabel("z from axicon (mm)")
            if col == 1:
                ax.set_ylabel("x at fixed y = 0 (mm)")
            else:
                ax.tick_params(labelleft=False)

        fig.suptitle(f"{title}   ·   {_format_value(family, value)}",
                     color=FG, fontsize=16.5, weight="bold", y=0.965)
        fig.text(0.5, 0.855,
                 "Illustrative sensitivity case in the V1 forward model; perturbation magnitude is not a measured bench tolerance.",
                 color=MUTED, ha="center", fontsize=8.7)
        png, pdf = _save_final(fig, out / f"candidate_{family}")
        png_paths.append(png)
        records.append({
            "family": family,
            "display_name": title,
            "value": float(value),
            "display_value": _format_value(family, value),
            "units": str(reg[family].get("units", "")),
            "fidelity": str(reg[family].get("fidelity", "")),
            "component_plane": comp_title,
            "png_500dpi": str(png),
            "pdf_vector": str(pdf),
        })
    return records, png_paths


def build_candidate_contact(out: Path, pngs: list[Path], records: list[dict]) -> Path:
    cols = 2
    rows = int(math.ceil(len(pngs)/cols))
    fig = plt.figure(figsize=(18.0, 4.45*rows), facecolor="#15191d")
    gs = fig.add_gridspec(rows, cols, hspace=0.10, wspace=0.045,
                          left=0.02, right=0.98, top=0.955, bottom=0.02)
    for i, (p, rec) in enumerate(zip(pngs, records)):
        ax = fig.add_subplot(gs[i//cols, i%cols])
        ax.imshow(plt.imread(p))
        ax.set_title(rec["display_name"], color="white", fontsize=11.8, weight="bold", pad=3)
        ax.axis("off")
    for i in range(len(pngs), rows*cols):
        fig.add_subplot(gs[i//cols, i%cols]).axis("off")
    fig.suptitle("System-error figure shootout — component plane + downstream propagation",
                 color="white", fontsize=20, weight="bold", y=0.992)
    return _save_review(fig, out / "00_component_plane_error_contact_sheet.png", dpi=150)


def build_q20_contact(out: Path) -> tuple[Path, list[dict]]:
    """Review tracked q20 figures without promoting any one of them yet."""
    cards = []
    manifest = []
    thumb_w, thumb_h = 1500, 920
    for rel, title in Q20_FILES:
        path = ROOT / rel
        if not path.exists():
            manifest.append({"path": rel, "title": title, "status": "missing"})
            continue
        im = Image.open(path).convert("RGB")
        original_size = im.size
        fitted = ImageOps.contain(im, (thumb_w, thumb_h), method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h+90), (20,24,28))
        x0 = (thumb_w-fitted.width)//2
        y0 = (thumb_h-fitted.height)//2
        canvas.paste(fitted, (x0,y0))
        draw = ImageDraw.Draw(canvas)
        draw.text((24, thumb_h+25), title, fill=(240,240,240))
        cards.append(canvas)
        manifest.append({"path": rel, "title": title, "status": "tracked", "pixel_size": list(original_size)})
    cols=2; rows=int(math.ceil(len(cards)/cols)) if cards else 1
    sheet = Image.new("RGB", (cols*thumb_w, rows*(thumb_h+90)+120), (18,22,26))
    draw = ImageDraw.Draw(sheet)
    draw.text((35,35), "Existing q=20 inverse/correction figures — visual audit only", fill=(250,250,250))
    for i, card in enumerate(cards):
        sheet.paste(card, ((i%cols)*thumb_w, 120+(i//cols)*(thumb_h+90)))
    path = out / "01_q20_existing_figure_audit.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92, subsampling=0)
    return path, manifest


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records, pngs = build_candidates(OUT)
    contact = build_candidate_contact(OUT, pngs, records)
    q20_contact, q20 = build_q20_contact(OUT)
    manifest = {
        "outcome": "POSTER-ERROR-DICTIONARY-V2",
        "individual_export": {"png_dpi": 500, "vector_pdf": True},
        "contact_sheets_are_review_only": True,
        "component_error_contact_sheet": str(contact),
        "q20_existing_figure_audit": str(q20_contact),
        "candidates": records,
        "q20_candidates": q20,
        "next_step": "visually select the strongest 3-4 physical-error candidates, then compose the physical-fit result with one scientifically defensible q20 residual-phase panel",
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
