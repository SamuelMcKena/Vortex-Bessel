# Materials Application Notes

The materials layer connects optical intensity fields to planning quantities
such as fluence maps, above-threshold masks, capsule-like feature geometry, and
shortlist tables. It is useful for comparing cases and designing experiments,
but it is not a calibrated material-response model.

## Status

Taxonomy labels:

- `beam_family`: `material_capsule`
- `model_level`: `material_proxy`
- `generation_method`: inherited from the optical case
- `hardware_status`: inherited from the optical case
- `material_model_status`: `planning_proxy`
- `qa_status`: `exploratory` or `validation_pipeline_checked`, depending on
  which stage has been run

## Planning Proxy Warning

This branch contains CrZnSe-specific proxy assumptions and is not validated
for fused-silica TGV, waveguide, welding, or modification predictions.

Material modification, threshold, incubation, capsule, and
fluence-threshold outputs are planning proxies. They are not calibrated
predictions of ablation, melt, void formation, refractive-index change,
cracking, stress, or permanent feature geometry.

These material-facing results are planning proxies unless explicitly marked
experimentally_calibrated. A thresholded fluence map is not a calibrated
prediction of ablation, void formation, refractive-index change, or weld
success.

Line-fluence XZ maps are diagnostic planning visualisations unless explicitly
generated from an energy-conserving 3D deposition model.

Quantitative claims require experimental calibration for the material, pulse
duration, wavelength, repetition rate, scan strategy, focusing stack, and
measurement protocol. Until that calibration exists, the outputs should be read
as relative design signals rather than ground-truth outcomes.

## What The Layer Does

- Converts simulated intensity into fluence using the pulse-energy normalization
  documented in `01_conventions.md`.
- Applies a simple incubated-threshold proxy to explore pulse-count trends.
- Builds above-threshold masks and geometric summaries for comparing candidate
  cases.
- Reports capsule and teardrop-style descriptors for feature-shape planning.
- Exports figures and CSV tables under `Publication_Study/outputs/`.

## What The Layer Does Not Do

- It does not solve heat diffusion, plasma dynamics, nonlinear absorption,
  stress, cracking, melt flow, redeposition, or index-change kinetics.
- It does not infer a real Cr:ZnSe threshold from first principles.
- It does not guarantee that an above-threshold mask maps to a written feature.
- It does not replace calibration shots, microscopy, profilometry, or
  spectroscopic validation.

## Calibration Boundary

When experimental data become available, the threshold and incubation constants
should be fitted with uncertainty ranges and stored with the run manifest. At
that point the `material_model_status` can move from `planning_proxy` to
`experimentally_calibrated` for the specific material and process window only.
