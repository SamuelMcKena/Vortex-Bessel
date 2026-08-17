"""Build a complete *nominal* SLM2 command preview.

This is deliberately not a hardware export: panel orientation, beam centre/radius,
camera-to-SLM mapping, z-to-annulus mapping and the 1030-nm greyscale LUT remain
uncalibrated.  It exists to show the correct composition of vortex + blaze + residual.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image


def build_slm2_complete_preview(correction_path, output_dir, *, nx=1920, ny=1080,
                                pixel_pitch_um=8.0, ell_slm2=-10,
                                blaze_period_px=20.0, beam_radius_mm=2.0,
                                correction_gain=0.20, centre_x_px=None,
                                centre_y_px=None, filename_tag="SLM2"):
    correction_path, output_dir = Path(correction_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cx = (nx-1)/2 if centre_x_px is None else float(centre_x_px)
    cy = (ny-1)/2 if centre_y_px is None else float(centre_y_px)
    yy, xx = np.indices((ny, nx), dtype=float)
    dx, dy = xx-cx, yy-cy
    theta = np.arctan2(dy, dx)
    rho = np.hypot(dx, dy)/(beam_radius_mm*1000/pixel_pitch_um)

    residual_src = np.load(correction_path)
    src_ny, src_nx = residual_src.shape
    sx = (rho*np.cos(theta)+1)*(src_nx-1)/2
    sy = (rho*np.sin(theta)+1)*(src_ny-1)/2
    source_filled = np.nan_to_num(residual_src, nan=0.0)
    residual = ndimage.map_coordinates(source_filled, [sy, sx], order=1,
                                       mode="constant", cval=0.0)
    residual[rho > 1] = 0.0

    vortex = ell_slm2*theta
    carrier = 2*np.pi*xx/blaze_period_px
    base_phase = np.mod(vortex+carrier, 2*np.pi)
    complete_phase = np.mod(vortex+carrier+correction_gain*residual, 2*np.pi)
    # Linear conversion is for visualization only. A measured panel LUT must replace it.
    base_gray = np.rint(base_phase/(2*np.pi)*255).astype(np.uint8)
    complete_gray = np.rint(complete_phase/(2*np.pi)*255).astype(np.uint8)

    stem = f"NOMINAL_PREVIEW_NOT_FOR_DISPLAY_{filename_tag}"
    np.save(output_dir/f"{stem}_phase_rad.npy", complete_phase.astype(np.float32))
    np.save(output_dir/f"{stem}_linear_gray.npy", complete_gray)
    np.save(output_dir/f"{stem}_mapped_correction_rad.npy",
            np.mod(correction_gain*residual, 2*np.pi).astype(np.float32))
    # Native panel exports: these are exact 1920x1080 rasters, independent of
    # Matplotlib figure DPI. The 16-bit versions preserve phase detail for
    # inspection/archival; hardware upload still requires the measured 8-bit LUT.
    Image.fromarray(complete_gray, mode="L").save(
        output_dir/f"{stem}_linear_gray_full_resolution.png")
    complete_u16 = np.rint(complete_phase/(2*np.pi)*65535).astype(np.uint16)
    correction_u16 = np.rint(np.mod(correction_gain*residual, 2*np.pi) /
                              (2*np.pi)*65535).astype(np.uint16)
    Image.fromarray(complete_u16, mode="I;16").save(
        output_dir/f"{stem}_phase_uint16_full_resolution.png")
    Image.fromarray(correction_u16, mode="I;16").save(
        output_dir/f"{stem}_correction_component_uint16_full_resolution.png")

    zoom = int(np.ceil(1.25*beam_radius_mm*1000/pixel_pitch_um))
    ys = slice(max(0, int(cy)-zoom), min(ny, int(cy)+zoom+1))
    xs = slice(max(0, int(cx)-zoom), min(nx, int(cx)+zoom+1))
    fig, axs = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    vortex_label = f"{ell_slm2:+d} vortex + " if ell_slm2 else ""
    panels = ((base_gray, f"SLM2 baseline: {vortex_label}20 px blaze"),
              (complete_gray, f"Complete nominal command, correction gain={correction_gain:.2f}"),
              (np.mod(correction_gain*residual, 2*np.pi), "Added mapped modal residual (wrapped rad)"))
    for col, (array, title) in enumerate(panels):
        cmap = "gray" if col < 2 else "twilight"
        vmax = 255 if col < 2 else 2*np.pi
        im = axs[0, col].imshow(array, origin="upper", cmap=cmap, vmin=0, vmax=vmax,
                                interpolation="nearest", aspect="equal")
        axs[0, col].set(title=title, xlabel="SLM2 x pixel", ylabel="SLM2 y pixel")
        axs[0, col].add_patch(plt.Circle((cx, cy), beam_radius_mm*1000/pixel_pitch_um,
                                        fill=False, color="cyan", lw=1.0))
        axs[1, col].imshow(array[ys, xs], origin="upper", cmap=cmap, vmin=0, vmax=vmax,
                           interpolation="nearest", aspect="equal")
        axs[1, col].set(title="Central illuminated-region zoom", xlabel="local x pixel",
                        ylabel="local y pixel")
        fig.colorbar(im, ax=axs[:, col], shrink=.75,
                     label="linear grayscale" if col < 2 else "wrapped phase (rad)")
    fig.suptitle("COMPLETE NOMINAL SLM2 PHASE COMMAND PREVIEW — NOT HARDWARE READY\n"
                 "linear grayscale only; measured 1030-nm LUT and coordinate calibration missing",
                 fontsize=15)
    fig.savefig(output_dir/f"{stem}_overview.png", dpi=400, bbox_inches="tight")
    fig.savefig(output_dir/f"{stem}_overview.pdf", dpi=400, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "status": "NOMINAL_PREVIEW_NOT_FOR_DISPLAY",
        "composition": "wrap(ell_slm2*theta + 2*pi*x/blaze_period_px + gain*mapped_residual)",
        "resolution_px": [nx, ny], "pixel_pitch_um": pixel_pitch_um,
        "ell_slm2": ell_slm2, "blaze_period_px": blaze_period_px,
        "nominal_beam_radius_mm": beam_radius_mm, "correction_gain": correction_gain,
        "centre_px": [cx, cy], "linear_gray_visualization_only": True,
        "native_resolution_exports": {
            "float_phase_rad": f"{stem}_phase_rad.npy",
            "float_mapped_correction_rad": f"{stem}_mapped_correction_rad.npy",
            "linear_8bit_png": f"{stem}_linear_gray_full_resolution.png",
            "phase_16bit_png": f"{stem}_phase_uint16_full_resolution.png",
            "correction_component_16bit_png": f"{stem}_correction_component_uint16_full_resolution.png",
            "vector_overview_pdf": f"{stem}_overview.pdf",
        },
        "missing_before_hardware": ["measured SLM2 1030-nm LUT/phase stroke",
            "SLM2 display rotation/parity", "measured beam centre and radius on SLM2",
            "camera-to-SLM2 transform", "physical z-to-input-annulus map",
            "experimental single-mask full-stack validation"],
    }
    (output_dir/f"{stem}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return complete_phase, complete_gray, manifest


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out = here/"outputs"/"slm_closed_loop_alignment"/"modal_q20"/"slm2_preview"
    phase, gray, manifest = build_slm2_complete_preview(
        here/"outputs"/"slm_closed_loop_alignment"/"modal_q20"/
        "UNCALIBRATED_DO_NOT_APPLY_q20_modal_correction.npy", out)
    print(json.dumps(manifest, indent=2))
