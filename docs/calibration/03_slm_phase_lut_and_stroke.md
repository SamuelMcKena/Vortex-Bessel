# 03 SLM Phase LUT and Stroke

## Purpose

Measure the wavelength-specific drive-to-phase response, usable stroke, panel orientation, and date
for each PLUTO panel without assuming an exact `2*pi` stroke.

## Required Equipment

An approved interferometric phase-calibration arrangement, polarisation diagnostics, stable imaging,
the two identified SLM panels, and the laboratory's authorised laser-safety controls.

## Beam-Safe Setup

Follow the formal laser safety procedure and the approved interferometer risk assessment. Use
enclosures, attenuation and engineering controls. Do not derive alignment actions from this document.

## Measurement Sequence

Verify physical panel identity and LC-director orientation. Sweep all required drive codes for each
panel, acquire at least three independent phase sweeps, unwrap phase using a documented method, and
retain source interferograms and hashes. Derive a monotonic inverse LUT only over the validated range.

## Equation And Units

For measured fringe displacement `delta_x` and fringe period `p`, use the validated interferometer
geometry to obtain

```text
delta_phi = 2*pi*delta_x/p
```

Phase and standard uncertainty are in radians; wavelength is in metres; drive code is integer.

## Repeats And Uncertainty

Use at least three full sweeps per panel. Include fringe-localisation, repeatability, drift, unwrap,
fit residual and wavelength uncertainty. Keep panel results separate.

## Acceptance Criteria

Panel identity and orientation verified; no unresolved unwrap jumps; usable monotonic LUT segment;
phase residual and stroke uncertainty reported; source evidence checksummed and calibration dated.

## Output Format

Populate `calibration/templates/slm_phase_lut_template.csv`; set `slm.phase_lut_path`,
`phase_stroke_rad`, `phase_stroke_uncertainty_rad`, `panel_orientation_verified`, and
`calibration_date` in the JSON bundle.

## Code Mapping And Claims

These values unlock calibrated SLM phase fidelity. They improve route fidelity but do not alone
calibrate dimensions, energy, or fluence.
