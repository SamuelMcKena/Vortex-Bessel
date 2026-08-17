# Stage 8C.3R — Component-Plane Reality Audit

Status: **audit of the pre-existing Stage 8C.3 / 8C.3C / 8C.3D implementation, prior to
the Component-Plane Reality Reset.**

Scope: optical / fluence diagnostics only. `final_export_allowed=False`. No material
response, absorbed energy, dose, plasma, ablation, voids, refractive-index change,
nonlinear propagation, thermal accumulation, or calibrated material prediction is
introduced or claimed anywhere in this stage.

---

## 1. Headline finding

**Every Stage 8C.3 "active perturbation" is applied to the already-propagated optical
*intensity* stack (`OpticalFieldStack.intensity_zyx`), not to a complex field at the
optical plane where the perturbation physically lives.**

The single function that applies them,
[`apply_lab_perturbations_to_stack`](../vbb_study/digital_twin/lab_perturbations.py)
(`lab_perturbations.py:590`), receives the post-engine intensity stack `I = stack.intensity_zyx`
and edits it with image-space operations: transverse interpolation shifts
(`_shift_stack_xy`), per-plane intensity walk-off (`_tilt_walkoff`), multiplicative
"gain" envelopes (`_elliptical_envelope`, `_phase_centre_degrade`,
`_apply_zernike_like_distortions`), binary masks (`_circle_mask`, `_rect_mask`), and
intensity blends (`_zero_order_component`). None of these re-propagates a field.

Two structural tells confirm this is a diagnostic image layer, not physics:

1. **Self-downgrade.** `lab_perturbations.py:466-473` rewrites *every* control that was
   registered `physics_active` to `diagnostic_active`, appending the note
   *"Stage 8C.3 applies this as a post-engine diagnostic stack transform"*. The module
   already knows it is not physics-active.
2. **Per-plane re-normalisation.** `_match_plane_integrals` (`lab_perturbations.py:994`)
   rescales each z-plane of the perturbed stack back to the *baseline* plane integral
   after several "perturbations" (ellipticity, relative misregistration, quantisation,
   phase noise, Zernike). This silently restores any energy the operation removed and
   directly violates the Stage 8C.3R energy rule *"do not renormalise each perturbed z
   plane to the same pre-clipping pulse energy."*

### Why the screenshots look unphysical

- **Hard semicircular / straight black cut-offs** (XY & ROI): produced by binary
  `_circle_mask` / `_rect_mask` multiplied directly onto a *propagated* plane (only when
  `enable_post_engine_spatial_clipping` is set), or by interpolation `left=0/right=0`
  fill from `_shift_*`. A real aperture clips the field **before** propagation; the
  output then shows diffraction rings, never a literal hard disc edge at the sample.
- **Abrupt rectangular XZ discontinuities**: `_tilt_walkoff` and `_shift_stack_z`
  translate whole planes by interpolation with zero-fill, leaving box-edged voids.
- **Larger local peak despite "clipping"**: `_match_plane_integrals` re-injects removed
  energy, and `_phase_centre_degrade` adds a hand-tuned positive `contaminated_core`
  term, so a "severe" case can show a brighter peak than baseline with no honest
  throughput penalty.

This is a *diagnostic visual transform*, not a `physics_active` model. The reset below
builds a genuine component-plane path that perturbs a complex field **before**
propagation, reusing the locked angular-spectrum propagator
(`vbb_study/equations/propagation.py`) and phase formulas
(`vbb_study/equations/holography.py`), without modifying the locked engine.

---

## 2. Control-by-control audit table

`field_is_repropagated_after_application` = was a field re-propagated after the control
was applied? For all pre-existing 8C.3 controls the answer is **no** — the operation is
on the final intensity stack.

`accepted_as_physical_model` = accepted as a `physics_active` component-plane model in
the 8C.3R reset? Where "no", the column says whether 8C.3R re-implements it physically
(`reimplemented`), keeps it as `warning_only`, or leaves it `display_only`.

| control_name | current_classification (pre-8C.3R) | intended_physical_plane | actual_implementation_stage (pre-8C.3R) | field_is_repropagated_after_application | energy_conserving_or_loss_accounted | accepted_as_physical_model | reason |
|---|---|---|---|---|---|---|---|
| enable_beam_decentre / beam_decentre_x/y_um | diagnostic_active (was physics_active) | input_complex_field | post_propagation_stack_transform (`_shift_stack_xy`) | no | translation only; no loss | no → **reimplemented** (8C.3R applies decentre to entrance amplitude) | decentre is an input-plane amplitude shift; shifting the output stack cannot show vignetting against a fixed downstream mask |
| enable_beam_tilt / beam_tilt_x/y_mrad | diagnostic_active | input_complex_field | post_propagation_stack_transform (`_tilt_walkoff` per-plane shift) | no | none | no → **reimplemented** (phase ramp `exp[i(kx0 x+ky0 y)]` before propagation) | a tilt is a pupil-plane phase ramp; the walk-off must emerge from propagation, not from translating output planes |
| enable_beam_ellipticity / beam_radius_x/y_um / beam_rotation_deg | diagnostic_active | input_complex_field | post_propagation_stack_transform (multiply + `_match_plane_integrals`) | no | re-normalised (energy restored) | no → **reimplemented** (elliptical entrance amplitude) | applied to entrance amplitude in 8C.3R |
| enable_input_aperture / input_aperture_radius/decentre | diagnostic_active | input_complex_field | post_propagation diagnostic, audit-only mask unless `enable_post_engine_spatial_clipping` | no | throughput audited but not applied to field | no → **reimplemented** (hard aperture on entrance field; loss propagates) | aperture clips the entrance field; diffraction appears downstream |
| enable_slm_phase_centre_offset / slm_phase_centre_offset_x/y | diagnostic_active | SLM_phase_mask_generation | post_propagation (folded into `_phase_centre_degrade` / 0.20× common translation) | no | re-normalised | no → **reimplemented** (common phase-mask centre offset) | belongs in mask generation |
| enable_vortex_centre_offset / vortex_centre_offset_x/y | diagnostic_active | SLM_phase_mask_generation | post_propagation (`_phase_centre_degrade` heuristic gain) | no | re-normalised | no → **reimplemented** (independent `phi_vortex` centre) | vortex centre is independent in 8C.3R: `ell*atan2(y-yv,x-xv)` |
| enable_axicon_centre_offset / axicon_centre_offset_x/y | diagnostic_active | SLM_phase_mask_generation | post_propagation (`_phase_centre_degrade` heuristic gain) | no | re-normalised | no → **reimplemented** (independent `phi_axicon` centre) | axicon centre is independent in 8C.3R: `-k_r*sqrt((x-xa)^2+(y-ya)^2)` |
| enable_physical_axicon_misalignment / apex_offset / tilt | diagnostic_active | SLM_phase_mask_generation + pre-propagation | post_propagation (axicon-centre + `_tilt_walkoff`) | no | none | partial → **reimplemented** (apex offset = axicon centre; apex tilt = phase ramp) | mapped onto physical mask/ramp perturbations |
| physical_axicon_angle_error_deg | warning_only | SLM_phase_mask_generation | warning only | no | n/a | warning_only | cone-angle retune needs route-level k_r change; flagged, not modelled |
| enable_axicon_apex_defect / radius | warning_only | SLM_phase_mask_generation | warning only | no | n/a | warning_only | apex micro-defect needs sub-resolution modelling |
| enable_slm_phase_quantisation / slm_phase_levels | diagnostic_active | SLM_phase_mask_generation | post_propagation (`_quantisation_modulation` gain + renorm) | no | re-normalised | no → **reimplemented** (quantise the mask phase before propagation) | quantisation is a phase operation on the mask |
| enable_slm_phase_noise / rms / seed | diagnostic_active | SLM_phase_mask_generation | post_propagation (seeded intensity gain + renorm) | no | re-normalised | no → **reimplemented** (seeded phase noise added to mask) | phase noise lives on the mask phase |
| enable_slm_pixelation / enable_slm_fill_factor / slm_fill_factor | diagnostic_active | SLM_amplitude_or_active_area | post_propagation (gain / scalar) | no | fill-factor scalar only | partial → **reimplemented** (fill-factor amplitude before propagation) | amplitude mask at SLM |
| enable_slm_dead_pixels / fraction / seed | diagnostic_active | SLM_amplitude_or_active_area | post_propagation (seeded amplitude mask on stack) | no | loss measured on stack | no → **reimplemented** (seeded amplitude dropout on the SLM field) | amplitude defect at SLM |
| enable_slm_active_area / width / height / decentre | diagnostic_active | SLM_amplitude_or_active_area | post_propagation (`_rect_mask`, audit-only unless post-engine clip) | no | throughput audited | no → **reimplemented** (rectangular amplitude clip on SLM field) | finite device aperture clips field, then diffracts |
| enable_slm_rotation / enable_mask_rotation / *_deg | diagnostic_active (implemented=False) | SLM_phase_mask_generation | warning only | no | n/a | warning_only | mask resampling/rotation not implemented |
| enable_zero_order_leakage / zero_order_leakage_fraction / mode | diagnostic_active | SLM_phase_mask_generation (order content) | post_propagation (blend with gaussian `_zero_order_component`) | no | blended on intensity | no → **reimplemented** (coherent unmodulated carrier added before propagation; fills core on focus) | zero order is a real field component, not an intensity blend |
| enable_unwanted_order_leakage / fraction / k-shift | diagnostic_active | Fourier_plane | post_propagation (shifted intensity ghost) | no | blended | warning_only (no genuine Fourier plane) | needs an explicit 4F transform; not represented by the direct-propagation engine |
| enable_first_order_filter(+decentre/clipping) | warning_only | Fourier_plane | warning only | no | n/a | warning_only / future_not_implemented | the direct-propagation engine has no explicit Fourier plane to mask; honestly retained as warning-only |
| enable_relay_magnification_error / decentre / tilt / aperture | diagnostic_active | relay_plane | post_propagation (magnify/shift/walkoff/mask on stack) | no | aperture audited only | warning_only | no relay imaging plane is modelled in this engine; retained as warning-only (was silently "physics_active") |
| enable_pupil_clipping / pupil_radius / decentre / fill_target | diagnostic_active | objective_pupil_plane | post_propagation (`_circle_mask`, audit-only unless post-engine clip) | no | throughput audited | no → **reimplemented** (circular pupil mask on the field before the focusing propagation) | hard pupil edge applied to the field; downstream rings emerge from propagation |
| enable_zernike_aberrations / zernike_*_waves | diagnostic_active | objective_pupil_plane | post_propagation (`_apply_zernike_like_distortions` gain + renorm) | no | re-normalised | no → **reimplemented** (true Zernike *phase* `exp(i·2π·Σ cⱼZⱼ)` on the pupil field) | aberrations are pupil phase, not an intensity "gain" polynomial |
| enable_defocus / focus_offset_um, enable_focus_depth_error | geometry_active | pre-propagation / sample_geometry | post_propagation (`_shift_stack_z` interpolation) | no | none | reimplemented as defocus (Zernike) where a phase is meaningful; axial z-marker stays geometry | a pure z-shift of the stack is geometry, not a new focus field |
| enable_sample_tilt / surface_offset / refractive_index_error / thickness_limit | geometry_active | sample_geometry_only | geometry/metadata only | no | n/a | geometry_only (unchanged) | honestly geometry-only until an interface model consumes them |
| enable_interface_reflection / pulse_energy_jitter / repetition_rate_error / pulse_duration_error / average_power_limit | energy_active | energy ledger | energy ledger/bookkeeping | no | accounted in ledger | energy_only (unchanged) | legitimate energy bookkeeping; not a field perturbation |
| enable_pointing_jitter / stage_position_jitter / focus_drift | warning_only | metadata/ensemble | warning only | no | n/a | warning_only (unchanged) | needs an ensemble model |
| enable_camera_crop / detector_noise / display_autoscale | diagnostic_active | display_only | post_propagation display ops | no | n/a | **display_only / diagnostic_visual_transform** (explicitly labelled) | display/detector operations; never `physics_active` |

---

## 3. Classification summary

**Diagnostic visual transforms (NOT physics-active).** The whole pre-existing
`apply_lab_perturbations_to_stack` path, including `compute_misalignment_sensitivity_sweep`
and `plot_misalignment_sensitivity_sweep` in `active_realism_metrics.py`, operates on the
propagated intensity stack. It is retained for backward compatibility and regression
coverage, but is labelled **`diagnostic_visual_transform`** and must not be presented as
`physics_active`.

**Re-implemented physically in 8C.3R** (perturb the complex field *before* propagation,
re-propagate, account energy honestly): input decentre, ellipticity, rotation, input
aperture, beam tilt; SLM vortex-centre and axicon-centre offsets (independent), common
phase-mask offset, phase quantisation, phase noise, dead pixels, fill factor, SLM active
area; pupil clipping/decentre; Zernike defocus/astig/coma/spherical; zero-order leakage.

**Honestly warning-only in 8C.3R** (the direct-propagation engine has no explicit 4F or
relay imaging plane to act on): first-order filter radius/decentre/clipping, unwanted
diffraction-order leakage, relay magnification/decentre/tilt/aperture, physical-axicon
cone-angle error, axicon apex defect, SLM/mask rotation, pointing/stage/focus jitter.

**Geometry-only** (unchanged): sample tilt, surface offset, focus-depth marker,
refractive-index uncertainty, sample thickness.

**Energy/exposure bookkeeping** (unchanged): pulse-energy jitter, repetition-rate and
pulse-duration error, interface reflection, average-power limit.

**Display-only** (unchanged, explicitly labelled): camera crop, detector noise,
display autoscale.

---

## 4. What the reset adds

`vbb_study/digital_twin/component_plane_states.py` — explicit per-plane state
(complex field, coords, energy before/after, transmitted fraction, applied components,
warnings).

`vbb_study/digital_twin/component_plane_pipeline.py` — the physical path
`InputFieldState → SLMFieldState → PupilPlaneState → SampleEntranceState →
PropagatedFieldStack → FluenceStackResult`, built on the locked
`make_xy_grid` / band-limited angular-spectrum propagator and the locked vortex/axicon
phase formulas. Perturbations are applied at their physical plane to the **complex
field**; the Bessel zone, vignetting, diffraction rings, walk-off and core contamination
then emerge from genuine propagation.

`vbb_study/digital_twin/component_plane_metrics.py` — energy throughput ledger,
commanded-vs-actual axis tracking with a z-trajectory fit, translation-vs-deformation
classification, the seven required scenarios, and the
`stage8c3r_component_plane_reality_preview.png` figure.

Energy discipline: passive clipping reduces the transmitted energy carried into the
fluence scaling; no per-plane re-normalisation back to the pre-clip energy is performed;
the peak-to-energy ratio is reported so a genuine reshaping peak rise is distinguishable
from restored energy.
