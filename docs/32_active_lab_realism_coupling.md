# Stage 8C.3 Active Lab-Realism Coupling

Stage 8C.3 makes the lab-realism controls visible in downstream optical diagnostics where that is physically feasible without changing the locked propagation engine. The output remains diagnostic optical field / optical fluence only. It is no material response, not absorbed energy, not a dose model, not a damage prediction, not a void prediction, not a waveguide prediction, and not an ablation prediction.

All saved figures remain stamped with `final_export_allowed=False` and `figure_status=diagnostic_allowed`.

## Control Classes

Each lab-realism control is classified and declares affected outputs.

- `physics_active`: reserved for future engine-level controls consumed before propagation. Stage 8C.3 post-engine stack/image transforms are not labelled `physics_active`.
- `energy_active`: changes energy ledger, pulse energy, repetition rate, interface transmission, fluence scaling, or exposure bookkeeping.
- `geometry_active`: changes sample, focus, pupil, or exposure geometry. Some geometry controls move markers while leaving the field unchanged, and that is stated in the control row.
- `diagnostic_active`: changes selected plane, ROI, display/crop/noise diagnostics, warnings, metadata, or Stage 8C.3 post-engine optical/fluence stack diagnostics.
- `warning_only`: accepted as an experimental control but not coupled to the field in Stage 8C.3; dashboard warnings flag it if enabled.
- `metadata_only`: recorded for traceability but not an active scalar optical primitive.
- `future_not_implemented`: intentionally disabled future physics.

Affected-output labels are restricted to:

`field`, `phase`, `amplitude`, `angular_spectrum`, `energy_ledger`, `fourier_filter`, `pupil`, `sample_geometry`, `fluence`, `exposure_bookkeeping`, `warnings`, `metadata`, and `future_stage`.

## Active Perturbations And Physical Placement

The active perturbation pass is implemented in `vbb_study/digital_twin/lab_perturbations.py`. It operates on the canonical `OpticalFieldStack` after the locked field engine has produced a baseline stack.

Every classified row now reports both `physical_placement` and `implementation_stage`. The placement tells where the perturbation belongs in a real optical model; the implementation stage tells what Stage 8C.3 actually does today. If a control is implemented as a post-propagation diagnostic stack transform, it is `diagnostic_active`, not `physics_active`.

- Beam decentre belongs at the input complex field before phase-mask interaction; Stage 8C.3 applies a diagnostic transverse shift.
- Beam tilt belongs as an input complex-field phase ramp before propagation; Stage 8C.3 records the phase-ramp equivalent and visualises z-dependent walk-off.
- Beam ellipticity applies an anisotropic amplitude envelope.
- Input, relay, SLM active-area, and pupil apertures clip the amplitude and report clipped power.
- Vortex centre offset belongs in vortex phase-mask generation only; axicon centre offset belongs in axicon phase-mask generation only. Relative vortex/axicon offset degrades phase-mask registration and ring symmetry.
- Co-shifted vortex and axicon centres are treated as common-mode translation/steering diagnostics and must not be used as the primary "misalignment ruins the beam" case.
- Physical axicon apex offset maps to the same centre-offset perturbation; physical tilt maps to the same steering proxy.
- SLM phase quantisation, pixelation, fill factor, seeded dead pixels, seeded phase noise, and active-area clipping perturb amplitude / phase proxies.
- Zero-order leakage adds a residual unmodulated component and fills the dark core.
- Unwanted order leakage adds a shifted ghost order.
- Relay magnification, decentre, tilt, and aperture controls perturb the downstream field.
- Low-order Zernike terms apply documented scalar intensity distortions for defocus, astigmatism, coma, spherical, and trefoil effects.
- Defocus and focus-depth error shift the axial response relative to the selected plane.

These are deliberate diagnostic perturbations on top of the locked optical result. They do not modify scalar/vector propagation equations, axicon physics, characterization locks, or validation baselines.

## Energy And Exposure Coupling

Energy-active controls use the existing Stage 8B/8C bookkeeping path:

- Pulse-energy jitter applies a deterministic seeded sample to pulse energy before optics.
- Repetition-rate error changes average power and pulse spacing.
- Pulse-duration error changes peak-intensity estimates.
- Interface reflection can use the existing Fresnel estimate.
- Average-power-limit enable controls whether power-limit warnings are active.
- Scan speed continues to change pulse spacing and pulses per spot.

## Warning-Only And Metadata-Only Controls

The following remain honest report-only controls in Stage 8C.3:

- Field-active first-order Fourier filtering is not implemented outside the engine. `enable_first_order_filter_decentre` and `enable_first_order_filter_clipping` therefore raise dashboard/report warnings when enabled.
- Physical axicon angle error is warning-only because cone-angle retuning belongs in the route engine.
- Axicon apex defect is warning-only.
- SLM/mask rotation is reported but not resampled.
- Pointing jitter, stage-position jitter, and focus drift need ensemble support and are warning-only.
- Polarisation and true vectorial energy-flow controls remain metadata-only in scalar mode.

Future material controls stay `future_not_implemented`: material response, threshold proxies, dose accumulation, nonlinear proxy, thermal proxy, microscope proxy, and calibrated prediction.

## Poynting Vector Boundary

In the current scalar optical model, direct Poynting-vector editing is not a primitive active input. Beam direction changes are represented by input beam tilt / angular-spectrum phase ramp. Polarisation and true vectorial energy-flow effects require a vector engine and are metadata-only unless the vector path is active.

The scalar tilt proxy is recorded as:

```text
E(x,y) exp[i(kx0 x + ky0 y)]
```

For intensity-only stacks, the dashboard visualises the corresponding z-dependent walk-off and records the phase-ramp slope in metadata.

## Degradation Metrics

`vbb_study/digital_twin/active_realism_metrics.py` computes:

`centroid_x_um`, `centroid_y_um`, `peak_x_um`, `peak_y_um`, `peak_z_um`, `ring_circularity_score`, `azimuthal_uniformity_score`, `side_lobe_imbalance`, `core_fill_fraction`, `central_darkness_contrast`, `peak_fluence_change_fraction`, `target_depth_peak_fluence`, `central_roi_peak_fluence`, `captured_power_drift_fraction`, `pupil_clipped_power_fraction`, `first_order_selected_fraction`, `symmetry_score`, `baseline_similarity_score`, `unregistered_similarity_score`, `registered_similarity_score`, `centroid_shift_um`, `translation_dominated_boolean`, and `residual_shape_deformation_score`.

Additional trajectory-span metrics are included for steering diagnostics.

The registration rule is:

- if unregistered similarity is low but registered similarity recovers near one, the perturbation is mainly translation or steering
- residual shape deformation is `1 - registered_similarity_score`
- translated beams are not called badly degraded unless residual deformation remains after recentering

Expected responses:

- beam decentre increases centroid offset
- beam tilt increases trajectory/projection span and records phase-ramp metadata
- vortex/axicon offset reduces symmetry or azimuthal uniformity
- zero-order leakage increases core fill
- pupil clipping increases clipped-power fraction
- defocus shifts peak z or changes peak fluence
- coma reduces symmetry or increases imbalance

## Baseline-Vs-Perturbed Interpretation

The notebook section `Baseline vs Perturbed Misalignment Sanity Check` compares:

- baseline XY fluence
- perturbed XY fluence
- difference XY map
- baseline XZ fluence
- perturbed XZ fluence
- difference XZ map
- metric delta table

The baseline and perturbed panels use identical axes and identical colour scales. This is essential: autoscaling must not hide degradation.

Preview path:

```text
outputs/figures/digital_twin/stage8c3_baseline_vs_perturbed_preview.png
```

## Stage 8C.3D Conservation and Axis Diagnostics

Stage 8C.3D keeps the Stage 8C.3C translation-versus-deformation repair and adds explicit passive throughput conservation, commanded-axis tracking, and field-of-view/crop diagnostics. The primary preview is saved as:

```text
outputs/figures/digital_twin/stage8c3d_conservation_axis_diagnostics_preview.png
```

Passive apertures, finite pupils, SLM active-area masks, fill-factor losses, and dead-pixel masks now reduce the pulse energy used for the perturbed fluence stack. Lost energy is not silently put back. The displayed audit includes input pulse energy, energy before perturbation, energy after passive loss, transmitted fraction, peak-to-total-energy ratio, and the renormalisation factor applied to the fluence scaling.

Stage 8C.3D deliberately does not multiply upstream passive aperture masks through already-propagated z-planes in the headline preview. That older diagnostic shortcut made harsh straight XZ cutoffs that looked like propagated physics but were only post-engine image-space clipping. Passive clipping is now rendered as an energy-throughput audit unless `enable_post_engine_spatial_clipping` is explicitly set for artifact investigation.

The saved headline preview uses the low-order-aberration scenario because it gives a smooth optical-field diagnostic without making passive clipping look propagated. Clipping-heavy cases remain in the scenario audit table and energy ledger until a real upstream aperture/pupil propagation path is implemented.

The commanded-axis diagnostic reports the commanded optical axis at x=0, y=0 against the actual ring/core centre and brightest point in the selected plane. It also fits x_ring(z)=ax z+bx and y_ring(z)=ay z+by to report surface intercept, target-plane offset, steering angle, field-of-view margin, out-of-frame fraction, and crop-edge energy fraction.

The sweep layout uses four columns:

1. aligned baseline
2. mild perturbation
3. severe perturbation
4. severe-minus-baseline difference / degradation summary

Rows show XY fluence, XZ fluence, central ROI/core fluence, and metric cards. Baseline, mild, and severe panels share colour scales within each row, so autoscaling cannot hide degradation. The figure carries status badges for diagnostic-only output, no material response, and shared colour scales.

The scenario registry includes:

- Scenario A: relative vortex/axicon misregistration, with vortex shifted and axicon fixed.
- Scenario A reverse: axicon shifted and vortex fixed.
- Scenario B: input beam decentre relative to fixed phase masks plus finite SLM active area.
- Scenario C: beam tilt plus finite pupil.
- Scenario D: objective pupil decentre and clipping.
- Scenario E: low-order aberrations: coma, astigmatism, defocus, and spherical aberration.
- Scenario F: zero-order leakage.
- Scenario G: combined diagnostic lab stress test.
- Translation diagnostic: co-shifted vortex and axicon, reported as translation-dominated when registration recovers the similarity.

Metric cards include centroid shift, ring-axis offset, fitted target-axis offset, steering angle, peak shift, passive-output energy, transmitted fraction, peak-to-energy ratio, azimuthal uniformity, core fill fraction, field-of-view margin, out-of-frame fraction, unregistered similarity, registered similarity, translation-dominated status, and residual shape deformation. The similarity metric is a normalized-shape similarity based on the relative L2 difference of normalized stacks, and the registered variant recenters the perturbed stack before comparing.

This is still a diagnostic stress visualizer over canonical optical stacks, not a full pre-propagation component engine. Component controls continue to report intended `physical_placement` and actual post-engine diagnostic implementation stage.

## Still Unmodelled

Stage 8C.3 does not implement material response, plasma dynamics, ablation, cracking, voids, waveguide prediction, refractive-index prediction, nonlinear propagation, thermal accumulation, calibrated modification, or simulated microscopy. Those remain future-stage work requiring separate physics and calibration.
