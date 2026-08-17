# Vortex Report Evidence Index

This index is the report-first route through G0, B0, V1 and V3. The authoritative row-level mapping
is `VORTEX_CLAIM_TO_EVIDENCE.csv`; the frozen copy and file hashes are under
`outputs/validation/report_freeze/`.

## Case Definitions

| Case | Definition | Primary source | Primary data |
|---|---|---|---|
| G0 | Charge-zero Gaussian control | `vbb_study/digital_twin/phase2a_canonical.py` | `outputs/validation/phase2a/canonical_case_summary.csv` |
| B0 | Charge-zero bright-core Bessel reference | same | same |
| V1 | Charge-one vortex Bessel | same plus Phase 2C winding metric | `outputs/validation/phase2c/phase2c_objective_benchmark.csv` |
| V3 | Charge-three vortex Bessel | same plus Phase 2C winding metric | same |

Analytic scalar definitions and metric conventions are in `vbb_study/equations/scalar_bessel.py`,
`vbb_study/equations/fields.py`, and `vbb_study/equations/metrics.py`.

## Target and Physical Routes

Phase 2A defines the analytic target control, ideal optical route, realistic fixed-bench route,
mild-error diagnostic, and deliberately degraded diagnostic. The report must retain those exact
labels. The repaired physical-axicon vortex contract is in `vbb_study/vbb_axicon.py`; the holographic
phase route is in `vbb_study/equations/holography.py`.

Older Stage 6 phase-mask and charge-sweep artifacts remain useful diagnostics, but they predate the
full Phase 1/1R mapping reconciliation. They are not the primary quantitative fixed-bench evidence.
No preview PNG is a calibrated SLM mask.

## Propagation and Topology

- Scalar propagation and the 5% plane-power gate: Phase 1 and Phase 2A.
- Physical-route vortex preservation: Phase 1/1R plus the two `test_slm2_preserve_vortex*` suites.
- V1/V3 winding, ring radius, dark-core radius, side lobes and focal comparison: Phase 2C objective
  benchmark.
- Axial behaviour and conserved-power views: Phase 2A summary and Phase 2B x-z/y-z figures.
- Debye quadrature and canonical-grid convergence: Phase 2C convergence outputs.

Winding is a complex-field phase measurement on the identified dominant transverse component. An
annular intensity or dark core alone is not evidence of topological charge.

## Lab Realism and Energy

The canonical hardware definition is
`outputs/validation/phase2a/canonical_hardware_manifest.json`. SLM models, selected-order efficiency,
pupil capture and the unified power ledger are governed by Phase 2A. Each filtered route contains one
`simulated_selected_first_order` factor. Absolute pulse energy and measured throughput remain blocked.

## Scalar Versus Vector Reference

Phase 2C supports scalar broad morphology for G0/B0. V1/V3 preserve charge and ring radius across the
comparison, but quantitative peak-location and component claims use vector Debye. The bounded scalar
normal-incidence interface comparison is acceptable only in its tested scope; component-resolved or
changed-interface work uses vector Fresnel.

## Figure Families

- Phase 2B `02_xy_planes/`: transverse evolution and SAS focus views.
- Phase 2B `03_xz_yz_slices/`: axial context and plane-power diagnostics.
- Phase 2B `04_profiles/`: native radial/angular/centre-line profiles.
- Phase 2B `05_3d_maps/`: fixed transverse-plane intensity surfaces for B0, V1 and V3.
- Phase 2C `objective/`: scalar FFT versus vector Debye.
- Phase 2C `interface/`: scalar versus spectral vector Fresnel.
- Phase 2C `profiles/` and `components/`: solver-specific focal metrics and components.

Interpolation in these figures is display-only; report metrics come from native arrays. See
`VORTEX_FIGURE_AND_TABLE_PLAN.md` for main/supplementary/diagnostic classifications.

## Limitations

Absolute dimensions, absolute fluence, LUT-calibrated masks, and experimental agreement require real
calibration. The current repository makes no nonlinear material-modification, accumulation, thermal,
or ablation prediction. Software tests establish implementation and numerical contracts only.
