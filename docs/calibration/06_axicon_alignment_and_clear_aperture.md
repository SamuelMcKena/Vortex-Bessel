# 06 Axicon Geometry and Clear Aperture

## Purpose

Verify axicon identity, base angle, refractive index provenance, usable clear aperture and centred
pupil overlap for Bessel-zone and clipping predictions.

## Required Equipment

Manufacturer metrology/certificate, calibrated mechanical or image scale, approved pupil imaging,
Stage 9A records, and the laboratory's authorised safety controls.

## Beam-Safe Setup

Follow the formal safety procedure and approved engineering controls. Geometry may be recorded with
the laser off where possible. Any illuminated observation must use the authorised enclosed low-power
diagnostic route; this is not an alignment instruction.

## Measurement Sequence

Record serial identity and certificate values. Measure usable aperture on two axes, document edge
criteria, and acquire at least five pupil-overlap observations under the approved setup. Record
decentring without prescribing mirror moves. Retain scale and hashes.

## Equation And Units

The bounded geometric estimate used for uncertainty checking is

```text
z_B = beam_radius / ((n_axicon - 1) * tan(base_angle))
```

Lengths are metres, base angle degrees in the bundle and radians inside trigonometric evaluation,
and refractive index is dimensionless.

## Repeats And Uncertainty

Use at least five aperture/overlap readings. Include scale, edge selection, repeatability, certificate
uncertainty, wavelength dependence of index and base-angle uncertainty.

## Acceptance Criteria

Identity and wavelength applicability documented; finite clear aperture with no unexplained clipping;
measured beam footprint lies inside the accepted aperture with uncertainty margin; no value inferred
from an unscaled screenshot.

## Output Format

Record evidence through Stage 9A and map accepted values to `axicon.base_angle_deg`,
`axicon.clear_aperture_m`, and `axicon.refractive_index` in the JSON bundle.

## Code Mapping And Claims

These values support Bessel-zone and pupil-clipping uncertainty. They do not establish sample damage
or nonlinear material response.
