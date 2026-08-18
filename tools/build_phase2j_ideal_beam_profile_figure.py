"""High-resolution Phase 2J renderer for the ideal B0/V1/V3 presentation figure.

This is presentation-only. The optical route, fixed-support propagator and
fixed-laboratory coordinates are unchanged. The figure uses the shared
``phase2j_thermal`` palette and a high native simulation grid.

Presentation annotations explicitly connect the transverse row to the
z=60 mm slice of the longitudinal propagation maps. They do not alter the
simulated fields or any numerical normalisation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

import presentation_phase2j_style as style
from vbb_study.digital_twin.vortex_continuous_propagation import (
    build_fixed_plane_longitudinal_map,
    build_fixed_support_spectrum,
    native_field_at_z,
)
from vbb_study.digital_twin.vortex_system_route import build_system_route


Z_REF_M = 60.0e-3
Z_VALUES_M = np.linspace(5.0e-3, 120.0e-3, 72)
TRANSVERSE_HALFWIDTH_M = 0.18e-3
LONGITUDINAL_COORD_M = np.linspace(-0.18e-3, 0.18e-3, 481)


def _propagator(route: Mapping[str, Any]):
    return build_fixed_support_spectrum(
        np.asarray(route["post_axicon"], dtype=np.complex128),
        dict(route["grid"]),
        wavelength_m=float(route["metadata"]["wavelength_m"]),
        z_max_m=float(Z_VALUES_M[-1]),
        minimum_retained_spectral_power=0.995,
    )


def _xy_at_z(route: Mapping[str, Any], z_m: float = Z_REF_M):
    prop = _propagator(route)
    field = native_field_at_z(prop, float(z_m))
    return np.asarray(field, dtype=np.complex128), prop


def _longitudinal(route: Mapping[str, Any], label: str):
    prop = _propagator(route)
    mapped = build_fixed_plane_longitudinal_map(
        prop,
        z_values_m=Z_VALUES_M,
        x_coordinates_m=LONGITUDINAL_COORD_M,
        y_coordinates_m=LONGITUDINAL_COORD_M,
        fixed_x_m=0.0,
        fixed_y_m=0.0,
        source_label=label,
    )
    return mapped, prop


def _fixed_lab_crop(intensity: np.ndarray, grid: Mapping[str, Any]):
    x = np.asarray(grid["x"], dtype=float)
    ids = np.flatnonzero(np.abs(x) <= TRANSVERSE_HALFWIDTH_M)
    if ids.size < 70:
        raise RuntimeError(
            f"Phase 2J ideal crop has only {ids.size} native samples; "
            "increase --grid-n rather than hiding low native resolution with DPI."
        )
    crop = np.asarray(intensity)[np.ix_(ids, ids)]
    extent = [
        x[ids[0]] * 1e3,
        x[ids[-1]] * 1e3,
        x[ids[0]] * 1e3,
        x[ids[-1]] * 1e3,
    ]
    return crop, extent, int(ids.size)


def build_figure(output_dir: Path, grid_n: int) -> Path:
    if grid_n < 2048:
        raise ValueError("Phase 2J ideal presentation figure requires grid_n >= 2048")
    style.validate_palette_has_no_cool_segment()

    # The current integrated route keeps all other beamline settings fixed;
    # these three cases differ only by the programmed SLM1 vortex charge.
    cases = (
        ("B0", "B0 — ℓ=0\nbright-core Bessel"),
        ("V1", "V1 — ℓ=1\nvortex–Bessel"),
        ("V3", "V3 — ℓ=3\nvortex–Bessel"),
    )

    # Reserve an explicit header band so the slide annotations never collide
    # with the top-row panel titles.
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 7.8), constrained_layout=False)
    style.style_fig(fig)
    fig.subplots_adjust(
        left=0.058,
        right=0.985,
        bottom=0.080,
        top=0.805,
        wspace=0.085,
        hspace=0.315,
    )

    retained: list[float] = []
    native_crop_samples: list[int] = []

    for col, (case_id, title) in enumerate(cases):
        route = build_system_route(case_id, grid_n=int(grid_n))
        field, _ = _xy_at_z(route)
        intensity = np.abs(field) ** 2
        crop, extent, nsamp = _fixed_lab_crop(intensity, route["grid"])
        native_crop_samples.append(nsamp)
        style.draw_xy(
            axes[0, col],
            crop,
            extent,
            title,
            ylabel=(col == 0),
        )

        mapped, prop = _longitudinal(route, f"phase2j-ideal-{case_id}")
        retained.append(float(prop.retained_spectral_power_fraction))
        style.draw_xz(
            axes[1, col],
            np.asarray(mapped.xz_intensity, dtype=float),
            LONGITUDINAL_COORD_M,
            Z_VALUES_M,
            ylabel=(col == 0),
            z_ref_m=Z_REF_M,
        )
        axes[1, col].set_title(
            f"{case_id} — fixed-lab x–z propagation",
            fontsize=12.0,
            weight="bold",
            pad=7,
        )

    fig.suptitle(
        "Beam profile shaping — ideal simulated outputs",
        color=style.TEXT,
        fontsize=18,
        weight="bold",
        y=0.972,
    )
    fig.text(
        0.5,
        0.923,
        "Transverse intensity at z = 60 mm",
        ha="center",
        va="center",
        color=style.TEXT,
        fontsize=11.3,
        weight="bold",
    )
    fig.text(
        0.5,
        0.888,
        r"$\phi(r,\theta)=\phi_{\rm axicon}(r)+\ell\theta$   •   only $\ell$ varies: 0 → 1 → 3",
        ha="center",
        va="center",
        color=style.MUTED,
        fontsize=10.1,
    )
    fig.text(
        0.5,
        0.030,
        "Dashed line in each x–z map: z = 60 mm transverse slice shown above",
        ha="center",
        va="center",
        color=style.MUTED,
        fontsize=9.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "02_beam_profile_shaping_B0_V1_V3_thermal_tight.png"
    style.save(fig, out)

    manifest = output_dir / "02_beam_profile_shaping_B0_V1_V3_thermal_tight.txt"
    manifest.write_text(
        "\n".join(
            [
                "PHASE2J-PRESENTATION-VISUAL-REFINEMENT-V4",
                f"grid_n={grid_n}",
                f"colormap={style.CMAP_NAME}",
                f"palette_hex={','.join(style.THERMAL_HEX)}",
                f"transverse_halfwidth_mm={TRANSVERSE_HALFWIDTH_M*1e3:.6f}",
                f"transverse_reference_z_mm={Z_REF_M*1e3:.6f}",
                f"longitudinal_halfwidth_mm={abs(LONGITUDINAL_COORD_M[0])*1e3:.6f}",
                f"z_samples={len(Z_VALUES_M)}",
                f"native_crop_samples={native_crop_samples}",
                f"fixed_support_retained_power_fraction={retained}",
                f"display_interpolation={style.DISPLAY_INTERPOLATION}_only_for_rendering",
                "longitudinal_coordinates=fixed_lab_no_per_z_recentering",
                "normalisation=per_case_peak_for_morphology_only",
                "beam_family_control=programmed_SLM1_vortex_charge_only",
                "presentation_annotation=z60_transverse_slice_explicitly_linked_to_longitudinal_maps",
                "physics_change=false",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-n", type=int, default=2048)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/presentation_phase2j"),
    )
    args = parser.parse_args()
    print(build_figure(args.output_dir, args.grid_n))


if __name__ == "__main__":
    main()
