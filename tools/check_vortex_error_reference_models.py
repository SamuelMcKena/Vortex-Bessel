from __future__ import annotations

import json
import math

from vbb_study.digital_twin.vortex_error_reference_models import (
    error_model_fidelity_registry,
    first_order_iris_geometry,
    rounded_tip_modulation_period_m,
    snell_axicon_geometry,
)


def main() -> None:
    wavelength = 1029e-9
    carrier = 6250.0
    iris_radius = 2500.0
    focal_length = 0.300

    pointing = {}
    for mrad in (-1.0, -0.5, 0.0, 0.5, 1.0):
        result = first_order_iris_geometry(
            wavelength_m=wavelength,
            carrier_cpm=carrier,
            iris_radius_cpm=iris_radius,
            focal_length_m=focal_length,
            input_angle_x_rad=mrad * 1e-3,
        )
        pointing[f"{mrad:+.1f}_mrad"] = result

    axicons = {}
    for deg in (2.0, 20.0):
        geometry = snell_axicon_geometry(
            base_angle_rad=math.radians(deg),
            refractive_index=1.458,
            external_index=1.0,
        )
        axicons[f"{deg:g}_deg"] = {
            **geometry.__dict__,
            "deflection_deg": math.degrees(geometry.deflection_rad),
            "round_tip_reference_period_m": rounded_tip_modulation_period_m(
                wavelength_in_medium_m=wavelength,
                cone_angle_rad=geometry.deflection_rad,
            ),
        }

    payload = {
        "pointing": pointing,
        "axicons": axicons,
        "fidelity_registry": error_model_fidelity_registry(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
