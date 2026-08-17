# Stage 8C.1 Integrated Beam-to-Write Cockpit MVP

Stage 8C.1 turns the Stage 8B/8C backend diagnostics into one editable
source-to-sample cockpit. It remains an optical, energy-accounting, fluence, and
exposure-bookkeeping layer only.

It does not predict material modification, absorption, ablation, cracks, voids,
waveguides, calibrated index change, nonlinear propagation, or thermal
accumulation.

## Foundation Repairs

- `compute_energy_ledger` accepts a real `LaserSource` and no longer creates a
  false hardcoded source record. If source metadata are not provided, the ledger
  source is `None`.
- `scale_intensity_to_fluence_j_cm2` rejects NaN/inf intensity, sampling, and
  pulse energy inputs.
- `OpticalFieldPlane` now carries `x_um` and `y_um` coordinate arrays in the
  `intensity[y, x]` convention. Legacy callers that supply only `dx_um` and
  `dy_um` get centered coordinates using `(n - 1) / 2`.
- Stack extraction uses `crop_grid["y"]` when present. If it must fall back to
  `y=x`, metadata records `assumed_y_equals_x=True`.
- Reconstructed coordinates use the centered pixel convention:
  `(np.arange(n) - (n - 1) / 2.0) * dx_um`.
- `FluenceStackResult` exposes raw transverse integrals and normalized raw
  captured-power fraction by z. The per-plane normalized energy is still
  conserved, but raw captured-power drift is now the diagnostic quantity.
- Peak diagnostics include global peak, central ROI peak, target-depth peak,
  sample-surface peak, crop-edge warning, captured-power drift, and selected
  display-plane reason.

## Cockpit Flow

The user-facing notebook is:

`notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb`

It contains one top editable control cell covering:

- laser source
- pre-SLM conditioning
- telescope / beam expander
- SLM1 vortex phase
- SLM2 / axicon route
- first-order filtering
- relay optics
- objective and pupil
- sample interface
- in-sample propagation
- field-to-fluence display
- exposure bookkeeping
- disabled future physics toggles

The notebook sections are ordered as:

1. purpose and claim boundary
2. user controls
3. stage-by-stage lab realism report
4. experiment request summary
5. energy ledger
6. lab realism / hardware feasibility panel
7. real optical field acquisition
8. field-to-fluence scaling
9. exposure bookkeeping
10. integrated cockpit dashboard figure
11. interpretation panel
12. disabled future physics panels

## Lab Realism Report

`vbb_study.digital_twin.lab_realism_controls` defines:

- `LabStageControl`
- `LabStageResult`
- `LabRealismReport`

Every required beam-path stage reports editable inputs, computed outputs,
warnings, missing metrics, model status, status level, and handoff to the next
stage. Missing lab-realism quantities are explicitly shown as
`not available from current engine` or `future control / not implemented`.

Allowed status levels are:

- `pass`
- `caution`
- `fail`
- `diagnostic_only`
- `disabled_future`
- `missing`

## Dashboard Diagnostics

`vbb_study.digital_twin.cockpit_dashboard` provides:

- `compute_peak_location_diagnostics`
- `choose_display_plane`
- `build_warning_flags`
- `build_cockpit_summary`
- `plot_integrated_cockpit_dashboard`
- `make_interpretation_text`

The display plane defaults to the target-depth plane. If the global optical
peak is at or near a crop boundary, it remains visible as a warning diagnostic
but is not silently promoted to the headline plane.

The integrated dashboard shows:

- experiment request
- energy ledger
- exposure bookkeeping
- lab realism feasibility flags
- selected-plane XY intensity
- selected-plane XY fluence
- central ROI fluence zoom
- XZ fluence with sample surface, target-depth, selected-z, and global-peak markers
- peak fluence versus z
- raw captured-power drift versus z
- warnings and caveats
- disabled future physics panel

All saved figures carry:

- `final_export_allowed=False`
- `figure_status=diagnostic_allowed`

## Claim Boundary

Stage 8C.1 outputs are:

- optical predictions
- energy-accounting predictions
- fluence predictions
- exposure bookkeeping
- diagnostic previews

They are not material-response outputs and are not final experimental claims.

