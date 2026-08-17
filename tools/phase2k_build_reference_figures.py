"""Build Phase 2K truth-reference figures from corrected analytic equations.

These are validation/reference assets, not bench predictions and not thesis
hero figures.  They exist to make the corrected mathematics visually auditable
before any lab-route output is regenerated.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vbb_study.equations.scalar_bessel import (
    axicon_cone_angle_exact_rad,
    bessel_gauss_field,
    transverse_wavevector_from_axicon,
)


WAVELENGTH_M = 1029.0e-9
N_MEDIUM = 1.0
KR_M_INV = 0.75e6
WAIST_M = 16.0e-6
ELL_CASES = ((0, "B0"), (1, "V1"), (3, "V3"))


def _normalise(values: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=float), 0.0)
    peak = float(np.max(arr))
    return arr / max(peak, np.finfo(float).tiny)


def _independent_fresnel_fft(field: np.ndarray, dx_m: float, z_m: float, k_m_inv: float) -> np.ndarray:
    n = int(field.shape[0])
    fx = np.fft.fftfreq(n, d=float(dx_m))
    kx = 2.0 * np.pi * fx
    kx_grid, ky_grid = np.meshgrid(kx, kx, indexing="xy")
    transfer = np.exp(-1j * (kx_grid**2 + ky_grid**2) * float(z_m) / (2.0 * float(k_m_inv)))
    return np.fft.ifft2(np.fft.fft2(field) * transfer)


def build_family_figure(out: Path) -> dict[str, object]:
    axis = np.linspace(-18.0e-6, 18.0e-6, 401)
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    R = np.hypot(X, Y)
    Phi = np.arctan2(Y, X)
    z_values = np.linspace(-120.0e-6, 120.0e-6, 401)
    x_line = axis
    phi_line = np.zeros_like(x_line)
    r_line = np.abs(x_line)

    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4), constrained_layout=True)
    case_rows: list[dict[str, object]] = []
    for column, (ell, case_id) in enumerate(ELL_CASES):
        field0 = bessel_gauss_field(
            R,
            Phi,
            ell=ell,
            kr_m_inv=KR_M_INV,
            waist_m=WAIST_M,
            z_m=0.0,
        )
        intensity0 = _normalise(np.abs(field0) ** 2)
        axes[0, column].imshow(
            intensity0,
            origin="lower",
            extent=[axis[0] * 1e6, axis[-1] * 1e6, axis[0] * 1e6, axis[-1] * 1e6],
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axes[0, column].set_title(f"{case_id}: ell={ell}, z=0")
        axes[0, column].set_xlabel("x (um)")
        if column == 0:
            axes[0, column].set_ylabel("y (um)")

        xz = np.empty((z_values.size, x_line.size), dtype=float)
        for iz, z in enumerate(z_values):
            line = bessel_gauss_field(
                r_line,
                phi_line,
                ell=ell,
                kr_m_inv=KR_M_INV,
                waist_m=WAIST_M,
                z_m=float(z),
                wavelength0_m=WAVELENGTH_M,
                n_medium=N_MEDIUM,
            )
            xz[iz] = np.abs(line) ** 2
        xz = _normalise(xz)
        axes[1, column].imshow(
            xz,
            origin="lower",
            extent=[x_line[0] * 1e6, x_line[-1] * 1e6, z_values[0] * 1e6, z_values[-1] * 1e6],
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="auto",
        )
        axes[1, column].set_xlabel("x (um)")
        if column == 0:
            axes[1, column].set_ylabel("z (um)")
        axes[1, column].set_title("analytic finite BG propagation")
        case_rows.append({"case_id": case_id, "ell": ell})

    fig.suptitle(
        "Phase 2K analytic Bessel-Gauss reference family\n"
        "finite-energy paraxial solution; morphology normalised per case",
        fontsize=14,
    )
    path = out / "01_corrected_bessel_gauss_reference_family.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return {"path": path.name, "cases": case_rows}


def build_independent_propagation_check(out: Path) -> dict[str, object]:
    n = 512
    dx = 0.30e-6
    axis = (np.arange(n, dtype=float) - n // 2) * dx
    X, Y = np.meshgrid(axis, axis, indexing="xy")
    R = np.hypot(X, Y)
    Phi = np.arctan2(Y, X)
    ell = 3
    z = 22.0e-6
    k = 2.0 * np.pi * N_MEDIUM / WAVELENGTH_M
    initial = bessel_gauss_field(
        R,
        Phi,
        ell=ell,
        kr_m_inv=KR_M_INV,
        waist_m=WAIST_M,
        z_m=0.0,
    )
    numeric = _independent_fresnel_fft(initial, dx, z, k)
    analytic = bessel_gauss_field(
        R,
        Phi,
        ell=ell,
        kr_m_inv=KR_M_INV,
        waist_m=WAIST_M,
        z_m=z,
        wavelength0_m=WAVELENGTH_M,
        n_medium=N_MEDIUM,
    )
    mask = np.abs(analytic) >= 1.0e-5 * float(np.max(np.abs(analytic)))
    rel_l2 = float(np.linalg.norm((numeric - analytic)[mask]) / np.linalg.norm(analytic[mask]))

    crop = np.flatnonzero(np.abs(axis) <= 18.0e-6)
    A = _normalise(np.abs(analytic[np.ix_(crop, crop)]) ** 2)
    N = _normalise(np.abs(numeric[np.ix_(crop, crop)]) ** 2)
    residual = np.abs(N - A)
    extent = [axis[crop[0]] * 1e6, axis[crop[-1]] * 1e6, axis[crop[0]] * 1e6, axis[crop[-1]] * 1e6]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), constrained_layout=True)
    for ax, data, title in zip(
        axes,
        (A, N, residual),
        ("analytic finite BG", "independent FFT Fresnel", "|normalised intensity difference|"),
    ):
        image = ax.imshow(
            data,
            origin="lower",
            extent=extent,
            cmap="inferno" if "difference" not in title else "magma",
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x (um)")
        fig.colorbar(image, ax=ax, shrink=0.8)
    axes[0].set_ylabel("y (um)")
    fig.suptitle(f"V3 propagation truth check at z={z*1e6:.1f} um; complex-field relative L2={rel_l2:.3e}")
    path = out / "02_bessel_gauss_analytic_vs_independent_fresnel.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return {"path": path.name, "complex_field_relative_L2": rel_l2, "z_m": z, "ell": ell}


def build_axicon_reference(out: Path) -> dict[str, object]:
    n_axicon = 1.458
    n_external = 1.0
    k0 = 2.0 * np.pi / WAVELENGTH_M
    gamma_deg = np.linspace(0.1, 25.0, 500)
    exact_kr = np.empty_like(gamma_deg)
    thin_kr = np.empty_like(gamma_deg)
    exact_deflection = np.empty_like(gamma_deg)
    for index, angle_deg in enumerate(gamma_deg):
        gamma = np.deg2rad(float(angle_deg))
        theta = axicon_cone_angle_exact_rad(n_axicon, n_external, gamma)
        exact_deflection[index] = np.rad2deg(theta)
        exact_kr[index] = transverse_wavevector_from_axicon(
            k0, n_axicon, n_external, gamma, mode="snell_exact"
        )
        thin_kr[index] = transverse_wavevector_from_axicon(
            k0, n_axicon, n_external, gamma, mode="tan"
        )
    rel = (thin_kr - exact_kr) / exact_kr

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    axes[0].plot(gamma_deg, exact_deflection, label="exact Snell deflection")
    axes[0].set_xlabel("axicon base / surface-normal tilt gamma (deg)")
    axes[0].set_ylabel("external cone deflection theta (deg)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(gamma_deg, 100.0 * rel)
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_xlabel("axicon base / surface-normal tilt gamma (deg)")
    axes[1].set_ylabel("thin-phase k_r error versus exact Snell (%)")
    axes[1].grid(alpha=0.25)
    fig.suptitle(
        "Refractive axicon reference: exact Snell geometry versus thin phase-screen approximation\n"
        "angle convention is base/surface-normal tilt; not a manufacturer apex-angle assumption"
    )
    path = out / "03_axicon_snell_vs_thin_phase_reference.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)

    checkpoints = {}
    for angle in (2.0, 10.0, 20.0):
        i = int(np.argmin(np.abs(gamma_deg - angle)))
        checkpoints[str(angle)] = {
            "exact_deflection_deg": float(exact_deflection[i]),
            "thin_phase_relative_kr_error": float(rel[i]),
        }
    return {"path": path.name, "checkpoints": checkpoints}


def main() -> None:
    out = Path("outputs/validation/phase2k_reference_figures")
    out.mkdir(parents=True, exist_ok=True)
    items = [
        build_family_figure(out),
        build_independent_propagation_check(out),
        build_axicon_reference(out),
    ]
    manifest = {
        "outcome": "PHASE2K-MATHEMATICAL-REFERENCE-FIGURES",
        "claim_boundary": (
            "Analytic/numerical truth-reference assets only. They are not calibrated bench predictions, "
            "experimental images, material-response predictions or thesis-selection figures."
        ),
        "common_parameters": {
            "wavelength_m": WAVELENGTH_M,
            "n_medium": N_MEDIUM,
            "kr_m_inv": KR_M_INV,
            "waist_m": WAIST_M,
        },
        "references": {
            "scalar_bessel": "Durnin, JOSA A 4, 651-654 (1987), DOI 10.1364/JOSAA.4.000651",
            "bessel_gauss": "Gori, Guattari & Padovani, Optics Communications 64, 491-495 (1987), DOI 10.1016/0030-4018(87)90276-8",
            "axicon": "McLeod, JOSA 44, 592-597 (1954), DOI 10.1364/JOSA.44.000592",
        },
        "figures": items,
    }
    (out / "phase2k_reference_figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
