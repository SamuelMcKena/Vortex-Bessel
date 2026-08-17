from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from vbb_study.digital_twin.phase2a_contracts import canonical_hardware_manifest, hardware_value
from vbb_study.digital_twin.vortex_explicit_4f import explicit_4f_relay
from vbb_study.digital_twin.vortex_rotated_plane import lab_to_tilted_plane, tilted_to_lab_plane
from vbb_study.digital_twin.vortex_system_route import build_system_route


def power(a: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(a, dtype=complex)) ** 2))


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=complex).ravel()
    bb = np.asarray(b, dtype=complex).ravel()
    return float(abs(np.vdot(aa, bb)) / max(np.linalg.norm(aa) * np.linalg.norm(bb), np.finfo(float).tiny))


def intensity_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.abs(np.asarray(a, dtype=complex)) ** 2
    bb = np.abs(np.asarray(b, dtype=complex)) ** 2
    aa = aa.ravel() - float(np.mean(aa))
    bb = bb.ravel() - float(np.mean(bb))
    return float(np.dot(aa, bb) / max(np.linalg.norm(aa) * np.linalg.norm(bb), np.finfo(float).tiny))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid-n', type=int, default=1536)
    ap.add_argument('--case', default='V1')
    ap.add_argument('--output', type=Path, default=Path('outputs/validation/rotated_plane_actual_fields.json'))
    args = ap.parse_args()

    route = build_system_route(args.case, grid_n=args.grid_n)
    grid = route['grid']
    manifest = canonical_hardware_manifest()
    wavelength = float(hardware_value(manifest, 'wavelength_m'))
    f4f = float(hardware_value(manifest, 'fourf_focal_length_m'))
    iris_radius = float(hardware_value(manifest, 'fourier_iris_radius_m'))
    carrier = float(hardware_value(manifest, 'carrier_frequency_cpm'))
    relay = explicit_4f_relay(
        route['post_slm2'], grid,
        wavelength_m=wavelength,
        nominal_focal_length_m=f4f,
        nominal_iris_radius_m=iris_radius,
        nominal_carrier_cpm=carrier,
    )

    fields = {
        'pre_lens1': np.asarray(relay['pre_lens1']),
        'pre_lens2': np.asarray(relay['pre_lens2']),
    }
    rows = []
    for field_name, field in fields.items():
        p0 = power(field)
        for axis in ('x', 'y'):
            for deg in (0.25, 0.5):
                tx = math.radians(deg) if axis == 'x' else 0.0
                ty = math.radians(deg) if axis == 'y' else 0.0
                for order in (1, 3, 5):
                    tilted, m1 = lab_to_tilted_plane(
                        field, grid, wavelength_m=wavelength,
                        tilt_x_rad=tx, tilt_y_rad=ty,
                        interpolation_order=order,
                    )
                    returned, m2 = tilted_to_lab_plane(
                        tilted, grid, wavelength_m=wavelength,
                        tilt_x_rad=tx, tilt_y_rad=ty,
                        interpolation_order=order,
                    )
                    rows.append({
                        'case': args.case,
                        'grid_n': args.grid_n,
                        'field': field_name,
                        'axis': axis,
                        'tilt_deg': deg,
                        'order': order,
                        'forward_spectral_power_ratio': float(m1['spectral_power_ratio']),
                        'inverse_spectral_power_ratio': float(m2['spectral_power_ratio']),
                        'roundtrip_power_ratio': power(returned) / p0,
                        'roundtrip_field_overlap': overlap(field, returned),
                        'roundtrip_intensity_correlation': intensity_corr(field, returned),
                    })
    payload = {'outcome': 'ROTATED-PLANE-ACTUAL-FIELD-CONVERGENCE', 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
