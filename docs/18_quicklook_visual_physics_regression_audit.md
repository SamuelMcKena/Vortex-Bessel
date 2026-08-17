# Stage 8.7D Quicklook Visual-Physics Regression Audit

Stage 8.7C is not accepted as locked. This audit treats the quicklook notebook as a diagnostic workflow that must either show physically recognisable structures or fail loudly. It does not approve final publication/export cleanup.

## Summary

The ideal preview still looks sensible because it uses the trusted scalar ideal path and a true central crop of the propagated Bessel/vortex field. The lab-realistic and four-condition quicklook failures come from a combination of bad default visual settings and a real first-order carrier/filter geometry failure when the notebook uses a long blaze period such as `blaze_period_px=32`.

The lab-realistic path is not a toy or placeholder field: it calls the locked `bt.run_case(..., path="realistic")` beam-to-surface route, which uses the holographic SLM field, first-order isolation, objective pupil, focus-to-focal-plane transform, and propagation volume. The failure is therefore a quicklook configuration/diagnostic failure, not evidence that a simplified lab field was substituted.

The most important finding is that the air-side beam-to-surface wrapper designs the conical spectrum in air. For the default ell=3, 3 um / 150 um target, the first-order cone radius is about `5.03 lp/mm` in the SLM/Fourier plane. A `32 px` blaze has carrier `3.906 lp/mm`, which is inside that cone radius. The engine correctly marks this geometry invalid and clips the effective filter radius to `3.711 lp/mm`; the selected fraction collapses to about `5.7e-4` at the old coarse `grid_size=192`, reproducing the observed `~0.000608` failure.

## Evidence Snapshot

Using the current locked engine through `quicklook.make_twin_config()` and `bt.run_case(..., path="realistic")`:

| case | grid | blaze period | carrier lp/mm | cone radius lp/mm | geometry valid | selected fraction |
|---|---:|---:|---:|---:|---|---:|
| old notebook-like failure | 192 | 32 | 3.906 | 5.025 | false | 0.0005687 |
| current JSON default | 256 | 20 | 6.250 | 5.025 | true | 0.4391 |
| balanced visual probe | 384 | 20 | 6.250 | 5.025 | true | 0.7478 |

The `first_order_filter_radius=2.5` notebook alias resolves to `first_order_filter_radius_lpmm=2.5`. In the air-side lab route, the engine expands the effective stop to the recommended cone-containing radius, about `5.147 lp/mm`, when the carrier leaves enough room. That unit handling is correct, but the alias is ambiguous for a user-facing quicklook control.

## Required Questions

1. Why does the ideal preview still show a sensible ring while lab-realistic quicklook plots do not?

   The ideal preview uses the trusted scalar target/axicon propagation path without holographic carrier isolation. It is sampled on the focal/sample grid and centrally cropped, so the ell>0 ring remains recognisable. The lab-realistic route must pass a wide conical spectrum around the blaze carrier. With invalid carrier/filter geometry, almost no useful first-order content reaches the focal propagation, producing clipped or drifting artefacts.

2. Is quicklook using the same trusted propagation/build functions as the locked scalar/lab-realism notebooks?

   Yes. Quicklook calls `bt.run_case()` for ideal and realistic previews. The realistic case goes through `vbb_studies.build_beam_to_surface_result()` and the holographic route, including `build_realistic_slm_field()`, `first_order_filter_geometry()`, `isolate_first_order()`, `focus_to_focal_plane()`, and `propagate_volume()`.

3. Is quicklook accidentally using a simplified placeholder field for lab-realistic previews?

   No. The lab-realistic preview is not a placeholder. The failure is caused by invalid or marginal quicklook defaults and too-coarse visual sampling, not by a toy lab field.

4. Is the first-order filter centred on the correct diffraction order?

   In the trusted engine, yes. `isolate_first_order()` centers the filter at `slm.carrier_cpm`, i.e. the +1 blaze carrier. Raw inspection for `blaze_period_px=20` shows the mask spans the +1 carrier region. The bad plots occur when the carrier itself is too close to zero order to contain the conical spectrum cleanly.

5. Are the first-order filter radius units correct?

   Internally yes: `first_order_filter_radius_lpmm` is line pairs/cycles per millimetre in the SLM/Fourier spatial-frequency plane. The user-facing alias `first_order_filter_radius` is ambiguous and should be replaced or clearly deprecated in notebook usage.

6. Is `first_order_filter_radius=2.5` interpreted correctly?

   It is interpreted as `2.5 lp/mm`. For the air-side route this is smaller than the conical spectrum radius, so the engine recommends and uses an expanded effective radius of about `5.147 lp/mm` when the blaze carrier leaves enough room. The ambiguity is naming, not unit conversion.

7. Why is `first_order_selected_fraction` approximately `0.000608`?

   It is reproduced by the old notebook-like settings: `grid_size=192`, `axial_points=13`, and `blaze_period_px=32`. The carrier is `3.906 lp/mm`, below the required `~5.147 lp/mm` effective stop radius. The engine flags the geometry invalid and clips the stop, leaving a tiny selected fraction (`0.0005687` in the audit probe).

8. Is the blaze period/order location correctly mapped to the filter plane?

   The mapping is correct: carrier frequency is `1 / (blaze_period_px * slm_pixel_pitch)`, converted to lp/mm. The quicklook problem is that a `32 px` blaze places the carrier too close to zero order for the air-side conical spectrum.

9. Is the objective pupil clipping too aggressive?

   The objective pupil can degrade the lab-realistic field, but the primary observed collapse is upstream in the carrier/filter geometry. With a valid carrier, selected fraction improves substantially. Pupil clipping should still be reported as a sanity metric because it can make a degraded lab field physically unusable.

10. Is the active SLM aperture / pupil / Fourier crop causing the crescent/blob artefacts?

   The artefacts are consistent with invalid first-order isolation plus fixed central cropping of a damaged/off-centred focal field. Active aperture and pupil effects can add asymmetry, but they are not the first cause of the near-zero selected fraction.

11. Are x/y/z axes using consistent physical units?

   The engine reports SLM/pupil coordinates in metres, focal/sample coordinates in metres, and quicklook display converts to mm or um. The units are consistent in the trusted functions. The confusing part is not units but fixed-origin cropping after a failed or shifted lab route.

12. Are quicklook XY/XZ arrays cropped around the true beam centre or around a fixed origin?

   Currently they are cropped around the fixed grid origin for display. This is acceptable for well-centred ideal fields but can hide or clip a lab field whose energy is shifted by order/filter failure. Quicklook needs a measured-centre crop or a loud failure label when the offset is too large.

13. Is the apparent beam drift physical, or a plotting/cropping/order-centering bug?

   For the failed `32 px` lab route, the drift is a diagnostic artefact of invalid order/filter geometry and fixed-origin cropping. For valid routes, small asymmetries may be physical/modelled lab realism, but quicklook must separate those from failed carrier/filter cases.

14. Is the display interpolation hiding numerical under-sampling?

   Yes. Bilinear interpolation makes coarse `192 x 13` or `256 x 17` maps appear smoother but does not recover rings or axial structure. Visual sanity metrics must be computed on raw arrays, and display interpolation must remain labelled as display-only.

15. Which plots are physically meaningful, and which are currently artefacts?

   The ideal Bessel/vortex previews are physically meaningful diagnostic previews. Conical propagated previews are meaningful when shown separately from the true Bessel-Gauss target. Lab-realistic previews are meaningful only when first-order geometry, selected fraction, beam centering, and basic beam-shape sanity pass. Four-condition lab-realistic comparisons using a failed carrier/filter setup are artefacts and must not be shown as predictions. Material proxy maps are meaningful only after upstream optical sanity passes; otherwise they are just failure diagnostics.

## Required Fix Direction

- Use a known-good visual default that leaves enough carrier margin for the air-side conical spectrum.
- Make `first_order_filter_radius_lpmm` explicit in notebook/config; keep the old alias only for backward compatibility.
- Increase visual sampling for default notebook plots.
- Add raw-array sanity metrics for centre suppression, ringness, radial symmetry, XZ structure, beam-centre offset, and first-order selection.
- Auto-centre display crops on measured beam centre when the optical sanity passes; fail loudly when the offset or first-order geometry is implausible.
- Split true target, conical propagation, lab-realistic, through-sample, and material proxy sections.
- Do not show failed lab-realistic or material-proxy plots as valid beam predictions.
