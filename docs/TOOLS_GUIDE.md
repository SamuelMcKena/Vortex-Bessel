# Specialist tools

`run_study.py` is the normal entry point for notebook workflows.  The `tools/` directory contains lower-level reproducibility, benchmark, error-study and figure-building scripts retained from the audited codebase.

Use these when you specifically want to regenerate a diagnostic or validate a numerical route rather than run the whole study.

## Beam/system-error work

- `tools/audit_vortex_system_error_signatures.py` — audit the simulated signatures produced by controlled vortex/system errors.
- `tools/check_system_error_propagation_reference.py` — independent/reference checks for system-error propagation.
- `tools/check_vortex_error_reference_models.py` — reference-model checks for vortex error cases.
- `tools/check_vortex_explicit_4f_parity.py` — explicit 4F/parity consistency checks.
- `tools/check_rotated_plane_actual_fields.py` and `check_rotated_plane_carrier_centering.py` — rotated-plane/carrier diagnostics.

## Axicon propagation and non-idealities

- `tools/check_axicon_continuous_propagation_benchmark.py` — continuous axial-propagation benchmark.
- `tools/check_axicon_tip_benchmark.py` — non-ideal/tip-related benchmark.
- `tools/check_axicon_oblique_benchmark.py` and `check_axicon_oblique_wave_benchmark_v2.py` — oblique/tilt reference checks.
- `tools/check_phase2b_longitudinal_continuity.py` — verifies that longitudinal views use a genuine fixed physical propagation frame.

The measured inverse/correction workflow itself is not in `tools/`; it lives under `notebooks/experimental/axicon_aberration_correction/` because it uses local measurement data.

## Presentation/figure regeneration

- `tools/build_phase2i_presentation_figures.py` — presentation figure suite from the Phase 2I evidence path.
- `tools/build_phase2j_ideal_beam_profile_figure.py` — high-resolution ideal beam-family figure.
- `tools/build_phase2j_presentation_suite.py` — refined presentation figure set.
- `tools/build_presentation_extended_evidence.py` and `build_presentation_extended_evidence_v2.py` — extended evidence figures.
- `tools/presentation_phase2j_style.py` — shared presentation plotting style.

These are retained as specialist figure builders.  They are not a second competing physics implementation; they should import/use the current `vbb_study` model paths.

## Validation, evidence and provenance

- `tools/build_vortex_report_freeze.py` — builds the report evidence/freeze metadata.
- `tools/audit_phase2b_propagation.py` — propagation audit.
- `tools/phase2k_output_truth_audit.py` and `phase2k_output_semantic_audit.py` — Phase 2K output truth/semantic checks.
- `tools/inventory_repo.py` and Phase 2K inventory helpers — provenance and repository inventory utilities.
- `tools/check_phase2h_vector_refractive_reference.py` — vector/refractive reference validation.
- `tools/check_phase2i_experimental_readiness.py` — checks whether the required calibration/evidence state exists for experimental claims.

## Recommended usage

Run a tool from the repository root, for example:

```powershell
python tools\check_axicon_tip_benchmark.py
python tools\audit_vortex_system_error_signatures.py
python tools\build_phase2j_ideal_beam_profile_figure.py
```

If a specialist script produces a scientific figure, retain its associated numeric/provenance output rather than treating a rendered image alone as quantitative evidence.

For the main study sequence, use `python run_study.py --list` instead.