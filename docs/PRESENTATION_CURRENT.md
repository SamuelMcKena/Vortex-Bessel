# Current presentation figure state

**Updated:** 2026-08-14

This is the canonical pointer for the current first-year presentation figure set. Future coding / ChatGPT sessions should use these repository-generated assets rather than historical presentation figures.

## Phase 2J visual standard — corrected thermal pass

The first Phase 2J standardisation pass still looked too much like the older presentation output because Matplotlib `inferno` retains purple low-intensity tones and the framing was not visually tight enough. That pass is superseded.

The current policy is now explicit:

- every **intensity** heatmap uses the custom `phase2j_thermal` palette: black -> deep red -> red -> orange -> amber -> yellow;
- the thermal palette contains no blue, cyan, green or purple segment;
- phase maps remain cyclic because phase is not an intensity quantity;
- signed residual maps remain diverging because residual sign must remain visible;
- main presentation simulations use native model grid `N=2048`;
- high-resolution export uses 480 dpi;
- display interpolation is `lanczos` only after the numerical field is calculated;
- main longitudinal maps use 72 z samples;
- x-z evidence stays in fixed laboratory coordinates with no per-z recentring;
- tighter crops are presentation framing only and never alter the simulated field;
- comparative error figures preserve common nominal normalisation.

Shared style:

`tools/presentation_phase2j_style.py`

Master renderer:

`tools/build_phase2j_presentation_suite.py`

GitHub Actions creates and uploads:

`outputs/figures/presentation_phase2j/00_phase2j_visual_audit_contact_sheet.jpg`

so the exact generated suite is visually inspected before promotion.

## Canonical live-presentation figures

### 1. Computational model / numerical route

`outputs/figures/presentation_phase2j/01_computational_route_phase2j.png`

### 2. Ideal beam profile shaping

`outputs/figures/presentation_phase2j/02_beam_profile_shaping_B0_V1_V3_thermal_tight.png`

Dedicated renderer:

`tools/build_phase2j_ideal_beam_profile_figure.py`

Current settings: N=2048, +/-0.18 mm transverse crop, +/-0.18 mm longitudinal crop, 72 z samples.

### 3. Moving towards the real experimental system

`outputs/figures/presentation_phase2j/08_V1_real_error_fingerprints_thermal_tight.png`

Nominal V1 is compared with representative input pointing, SLM1 registration and 4F iris-offset errors using the same forward model and fixed-laboratory propagation convention.

### 4. Axicon lateral decentre

`outputs/figures/presentation_phase2j/04_V1_axicon_decentre_fixed_lab_thermal_tight.png`

Physical perturbations remain -500 µm / aligned / +500 µm. Presentation framing is tightened while fixed-lab coordinates and common nominal normalisation are retained.

### 5. Non-ideal axicon tip

`outputs/figures/presentation_phase2j/05_V1_nonideal_tip_fixed_lab_thermal_tight.png`

Physical cases remain ideal sharp / 200 µm / 800 µm radial rounding and continue to use the existing tip-resolution gate.

### 6. Experiment / simulation closure

`outputs/figures/presentation_phase2j/06_simulation_experiment_closure_phase2j.png`

### Supporting figures

Tip-avoidance planning proxy:

`outputs/figures/presentation_phase2j/09_tip_avoidance_planning_proxy_thermal.png`

Synthetic z-stack inverse recovery:

`outputs/figures/presentation_phase2j/10_synthetic_zstack_inverse_recovery_thermal.png`

Master provenance / metrics manifest:

`outputs/figures/presentation_phase2j/presentation_phase2j_manifest.json`

## Scientific boundaries retained

1. Presentation styling does not alter optical fields, propagation physics or calibration assumptions.
2. Illustrative perturbation magnitudes are not automatically measured bench tolerances.
3. The tip-avoidance figure remains a planning proxy unless a calibrated phase-only implementation is demonstrated.
4. The z-stack inverse figure remains synthetic model-to-model validation until real calibrated camera data are fitted.
5. Correction remains an additive residual-wavefront correction problem rather than conjugation of the complete structured field.

## Rule for future presentation edits

Use repository-generated canonical Phase 2J figures only. Do not substitute locally approximated, AI-generated, cached historical or superseded `*_inferno*` figures when a current repository renderer exists. Presentation visual changes must be implemented in the shared Phase 2J style/master renderer and visually checked from the GitHub Actions artifact before promotion to `main`.
