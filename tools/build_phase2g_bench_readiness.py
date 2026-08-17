from __future__ import annotations

import argparse
import json
from pathlib import Path

from vbb_study.calibration.schema import CalibrationBundle, canonical_calibration_template
from vbb_study.calibration.validation import calibration_readiness_for_claim, validate_calibration_bundle
from vbb_study.vector_arm_config import SLMPanelConfig


PHASE2G_GATES = (
    "slm_phase_fidelity",
    "fourier_filter_geometry",
    "wavefront_correction",
    "refractive_axicon_tilt",
    "objective_sample_vector_field",
    "vector_case1_hardware",
    "vector_analyzer_spots",
    "segmented_vector_hexagon",
    "full_stokes_vector",
    "absolute_dimensions",
    "absolute_fluence",
    "spatiotemporal_field",
    "experimental_agreement",
    "material_response",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2G bench-calibration readiness evidence.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/phase2g/bench_readiness.json"),
    )
    parser.add_argument(
        "--template-output",
        type=Path,
        default=Path("calibration/templates/phase2g_bench_calibration_template.json"),
    )
    args = parser.parse_args()

    template = canonical_calibration_template()
    bundle = CalibrationBundle(template)
    validation = validate_calibration_bundle(bundle)
    panel = SLMPanelConfig()
    readiness = {}
    for gate in PHASE2G_GATES:
        row = calibration_readiness_for_claim(bundle, gate)
        readiness[gate] = {
            "ready": bool(row.ready),
            "status": row.status,
            "missing_measurements": list(row.missing_measurements),
            "non_calibrated_measurements": list(row.non_calibrated_measurements),
            "satisfied_measurements": list(row.satisfied_measurements),
        }

    report = {
        "outcome": "PHASE2G-CALIBRATION-READY-CODE-MEASUREMENTS-PENDING",
        "schema_version": template["schema_version"],
        "template_valid": bool(validation.valid_schema),
        "template_errors": list(validation.errors),
        "known_physical_bench_contract": {
            "slm_make": template["slm"]["make"],
            "slm_model": template["slm"]["model"],
            "slm_resolution_px": template["slm"]["active_resolution"],
            "slm_pixel_pitch_m": template["slm"]["pixel_pitch_m"],
            "slm_fill_factor": template["slm"]["fill_factor"],
            "slm_phase_bits": template["slm"]["phase_bits"],
            "carrier_period_px": int(template["slm"]["carrier_period_px"]),
            "carrier_lp_per_mm": float(template["slm"]["carrier_lp_per_mm"]),
            "carrier_period_from_config_px": float(panel.carrier_period_px),
            "wavelength_m": template["laser"]["wavelength_m"]["value"],
            "camera_model": template["camera"]["model"],
        },
        "vector_study_contract": {
            "cylindrical_vector_atlas": {
                "states": ["radial", "azimuthal"],
                "ell_values": [1, 3],
                "linear_analyzer_angles_deg": [0, 45, 90, 135],
                "expected_petal_counts": {"ell_1": 2, "ell_3": 6},
                "direct_linear_analyzer_observables": ["S0", "S1", "S2", "psi"],
                "S3_policy": "blocked without calibrated QWP/full-Stokes analyzer",
            },
            "segmented_vector_hexagon": {
                "n_pairs": 3,
                "sector_family": "alternating radial/azimuthal six-sector",
                "route_kept_separate_from_case1": True,
                "calibration_requires": ["both SLM LUTs", "input polarization", "HWP/QWP state", "4F", "camera"],
            },
        },
        "readiness": readiness,
        "physics_components_available": [
            "audited scalar/vector free-space propagation",
            "physical SLM pixelation/quantisation/fill-factor/order filtering",
            "measured LUT inversion and greyscale export",
            "Shack-Hartmann Southwell least-squares OPD reconstruction",
            "topology-preserving additive SLM correction",
            "explicit 4F selected-order propagation",
            "explicit two-surface refractive axicon reference for rigid tilt",
            "vector Debye/Richards-Wolf objective focusing",
            "spectral vector Fresnel sample interface",
            "cylindrical-vector 0/45/90/135 analyzer comparison",
            "camera-calibrated 2|ell|-petal count/orientation/modulation metrics",
            "segmented radial/azimuthal six-sector vector-hexagon study",
            "linear ultrafast spectral/time reconstruction",
            "fluence/peak-intensity/discrete scan exposure",
            "camera-calibrated simulation/experiment comparison",
            "calibration-bound empirical material-response interface",
        ],
        "absolute_claim_policy": (
            "Code readiness is not measurement readiness. A gate remains blocked until the corresponding "
            "laboratory values are present with calibrated provenance. Vector analyzer morphology and "
            "segmented-vector hardware are independently gated."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.template_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.template_output.write_text(json.dumps(template, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
