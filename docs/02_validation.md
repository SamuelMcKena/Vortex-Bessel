# Validation Record

This is the canonical validation note for the publication study. It is based on
the former `PHYSICS_AUDIT.md` file and the Stage G
`vbb_validation.run_validation_suite()` run. Re-run `NB_validation.ipynb` or the
validation stage of the publication runner to refresh the numbers.

Validation checks are reported by the validation pipeline when the relevant
stage is run. They are not a promise that every notebook execution re-runs every
physics check unless that stage is included in the requested run.

## Current Validation Table

| # | group | check | status | value | tolerance |
|---|---|---|---|---|---|
| 1 | propagator | bl_asm_energy_conservation | PASS | 2.6e-16 | <= 1e-10 relative drift |
| 2 | propagator | gaussian_waist_growth | PASS | 4.7e-4 | <= 2e-3 max relative waist error |
| 3 | true_bessel_gauss | ring_radius_invariant_inside_0p8_zmax | PASS | 0.023 | <= 0.03 relative ring-radius error |
| 4 | true_bessel_gauss | peak_halfmax_break_before_geometric_zmax | PASS | 0.62 z/zmax | 0.35-1.05 z/zmax |
| 5 | true_bessel_gauss | measured_zone_scales_with_eq5_zmax | PASS | 0.61 mean ratio | all in [0.45, 0.80], spread <= 0.05 |
| 6 | radial_metrics | ring_radius_matches_jprime_zero_across_ell_kr | PASS | 0.020 | <= 0.03 max relative error |
| 7-12 | sas | self-check suite | PASS | n/a | `bt.run_sas_self_checks` tolerances |
| 13 | sas | sas_vs_bl_asm_rel_l2_notebook04_cases | PASS | 0.149 | <= 0.18 relative L2 |
| 14 | sas | sas_retained_window_fraction_notebook04_cases | PASS | 0.900 | >= 0.90 |
| 15 | propagator | energy_conservation parametric API | PASS | 2.8e-16 | <= 1e-10 |
| 16 | propagator | check_gaussian_diffraction_parametric | PASS | 4.7e-4 | <= 0.03 |
| 17 | true_bessel_gauss | bg_invariance_ring_radius | PASS | 3.2e-16 | <= 0.05 CV |
| 18 | true_bessel_gauss | bg_invariance_peak_intensity | PASS | 0.194 | <= 0.25 CV |
| 19 | radial_metrics | ring_radius_vs_kr_ell3 parametric API | PASS | 0.0017 | <= 0.05 |
| 20 | convergence | resolution_convergence_bessel_zone_um | PASS | 0.000 | <= 0.05 |
| 21 | convergence | resolution_convergence_ring_radius_um | PASS | 0.0055 | <= 0.05 |

## Validation Notes

### Peak-Intensity Invariance

`bg_invariance_peak_intensity` passes with a coefficient of variation near
0.19. This is expected for a finite Bessel-Gauss beam. The FWHM zone is defined
from the axial range where the peak intensity remains above half its maximum,
so the Gaussian envelope naturally varies inside that zone. The ring-radius
invariance check confirms that the beam geometry remains Bessel-Gauss-like
inside the zone.

Experimental implication: axial intensity variation can produce
writing-depth-dependent feature size when the fluence margin above threshold is
small.

### Resolution Convergence

The convergence check uses dense axial sampling, a robust high-percentile peak
estimate, and interpolated half-maximum crossings. The current Stage G run
reports matching coarse/fine Bessel-zone estimates and a separately converged
ring radius.

## Not Tested By This Suite

- Interface refraction is checked by the through-sample stages when those are
  run, including power-continuity diagnostics.
- First-order isolation efficiency is checked per holographic case where that
  route is used.
- Phase quantization effects are reported through sampling and phase-validity
  labels.
- Threshold, fluence, capsule, and modification predictions are not calibrated
  material validation. They are planning proxies until experimental data are
  fitted.
