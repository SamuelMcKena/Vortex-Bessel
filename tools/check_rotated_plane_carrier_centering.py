from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_explicit_4f import explicit_4f_relay
from vbb_study.digital_twin.vortex_rotated_plane import _spline_uniform_complex, rotation_matrix
from vbb_study.digital_twin.vortex_system_route import build_system_route
from vbb_study.equations.fields import fft2c, ifft2c

EPS = np.finfo(float).tiny
TWOPI = 2.0 * np.pi


def power(a: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(a, dtype=complex)) ** 2))


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=complex).ravel()
    bb = np.asarray(b, dtype=complex).ravel()
    return float(abs(np.vdot(aa, bb)) / max(np.linalg.norm(aa) * np.linalg.norm(bb), EPS))


def intensity_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.abs(np.asarray(a, dtype=complex)) ** 2
    bb = np.abs(np.asarray(b, dtype=complex)) ** 2
    aa = aa.ravel() - float(np.mean(aa))
    bb = bb.ravel() - float(np.mean(bb))
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), EPS))


def spectral_centroid(field: np.ndarray, grid: dict) -> tuple[float, float]:
    s = fft2c(field)
    w = np.abs(s) ** 2
    total = max(float(np.sum(w)), EPS)
    return (
        float(np.sum(w * np.asarray(grid['FX'])) / total),
        float(np.sum(w * np.asarray(grid['FY'])) / total),
    )


def centered_rotate(
    field: np.ndarray,
    grid: dict,
    *,
    wavelength_m: float,
    tilt_x_rad: float,
    tilt_y_rad: float,
    source_center_cpm: tuple[float, float],
    inverse: bool,
    interpolation_order: int,
) -> tuple[np.ndarray, dict]:
    R = rotation_matrix(float(tilt_x_rad), float(tilt_y_rad))
    if inverse:
        R = R.T

    inv_lam = 1.0 / float(wavelength_m)
    fsx, fsy = map(float, source_center_cpm)
    fsz = math.sqrt(max(inv_lam * inv_lam - fsx * fsx - fsy * fsy, 0.0))
    k_src_c = np.asarray([fsx, fsy, fsz], dtype=float)
    k_dst_c = R.T @ k_src_c
    fdx, fdy = float(k_dst_c[0]), float(k_dst_c[1])

    X = np.asarray(grid['X'], dtype=float)
    Y = np.asarray(grid['Y'], dtype=float)
    baseband = np.asarray(field, dtype=complex) * np.exp(-1j * TWOPI * (fsx * X + fsy * Y))
    spectrum = fft2c(baseband)

    foff = np.asarray(grid['FX'], dtype=float)
    goff = np.asarray(grid['FY'], dtype=float)
    fd_abs = fdx + foff
    gd_abs = fdy + goff
    dest_sq = fd_abs * fd_abs + gd_abs * gd_abs
    valid_dest = dest_sq < inv_lam * inv_lam
    hd_abs = np.sqrt(np.maximum(inv_lam * inv_lam - dest_sq, 0.0))

    fx_src_abs = R[0, 0] * fd_abs + R[0, 1] * gd_abs + R[0, 2] * hd_abs
    fy_src_abs = R[1, 0] * fd_abs + R[1, 1] * gd_abs + R[1, 2] * hd_abs
    fz_src_abs = R[2, 0] * fd_abs + R[2, 1] * gd_abs + R[2, 2] * hd_abs
    valid = valid_dest & (fz_src_abs > 0.0)

    axis = np.asarray(grid.get('fx', grid['FX'][0]), dtype=float)
    sampled = _spline_uniform_complex(
        spectrum,
        axis,
        fx_src_abs - fsx,
        fy_src_abs - fsy,
        order=int(interpolation_order),
    )
    jac = np.zeros_like(hd_abs)
    jac[valid] = np.abs(fz_src_abs[valid]) / np.maximum(hd_abs[valid], EPS)
    out_spectrum = np.where(valid, sampled * jac, 0.0)
    envelope = ifft2c(out_spectrum)
    output = envelope * np.exp(1j * TWOPI * (fdx * X + fdy * Y))
    ratio = float(np.sum(np.abs(out_spectrum) ** 2) / max(np.sum(np.abs(spectrum) ** 2), EPS))
    return np.asarray(output), {
        'source_center_cpm': [fsx, fsy],
        'destination_center_cpm': [fdx, fdy],
        'spectral_power_ratio': ratio,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid-n', type=int, default=1536)
    ap.add_argument('--case', default='V1')
    ap.add_argument('--output', type=Path, default=Path('outputs/validation/rotated_plane_carrier_centering.json'))
    args = ap.parse_args()

    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, 'wavelength_m'))
    carrier = float(hardware_value(manifest, 'carrier_frequency_cpm'))
    f4f = float(hardware_value(manifest, 'fourf_focal_length_m'))
    iris_radius = float(hardware_value(manifest, 'fourier_iris_radius_m'))
    route = build_system_route(args.case, grid_n=args.grid_n)
    grid = route['grid']
    relay = explicit_4f_relay(
        route['post_slm2'], grid,
        wavelength_m=wavelength,
        nominal_focal_length_m=f4f,
        nominal_iris_radius_m=iris_radius,
        nominal_carrier_cpm=carrier,
    )

    rows = []
    for field_name in ('pre_lens1', 'pre_lens2'):
        field = np.asarray(relay[field_name])
        measured_center = spectral_centroid(field, grid)
        for center_label, center in (
            ('nominal_carrier', (carrier, 0.0)),
            ('spectral_centroid', measured_center),
        ):
            for axis in ('x', 'y'):
                for deg in (0.25, 0.5):
                    tx = math.radians(deg) if axis == 'x' else 0.0
                    ty = math.radians(deg) if axis == 'y' else 0.0
                    for order in (3, 5):
                        tilted, m1 = centered_rotate(
                            field, grid,
                            wavelength_m=wavelength,
                            tilt_x_rad=tx,
                            tilt_y_rad=ty,
                            source_center_cpm=center,
                            inverse=False,
                            interpolation_order=order,
                        )
                        dest_center = tuple(m1['destination_center_cpm'])
                        returned, m2 = centered_rotate(
                            tilted, grid,
                            wavelength_m=wavelength,
                            tilt_x_rad=tx,
                            tilt_y_rad=ty,
                            source_center_cpm=dest_center,
                            inverse=True,
                            interpolation_order=order,
                        )
                        rows.append({
                            'field': field_name,
                            'center_label': center_label,
                            'measured_source_center_cpm': list(measured_center),
                            'axis': axis,
                            'tilt_deg': deg,
                            'order': order,
                            'forward_spectral_power_ratio': m1['spectral_power_ratio'],
                            'inverse_spectral_power_ratio': m2['spectral_power_ratio'],
                            'roundtrip_power_ratio': power(returned) / power(field),
                            'roundtrip_field_overlap': overlap(field, returned),
                            'roundtrip_intensity_correlation': intensity_corr(field, returned),
                            'destination_center_cpm': m1['destination_center_cpm'],
                        })
    payload = {'outcome': 'ROTATED-PLANE-CARRIER-CENTERING', 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
