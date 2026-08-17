# 04 Fourier Plane, Carrier Order and Iris

## Purpose

Measure effective 4F focal length, selected `+1` order position, and iris centre/radius at the Fourier
plane using the corrected wavelength-bearing Fourier geometry.

## Required Equipment

A calibrated position measurement or camera, approved low-power order observation, iris readback,
known carrier masks, Stage 9A capture records, and authorised safety controls.

## Beam-Safe Setup

Follow the formal safety procedure and approved enclosed diagnostic route. Use rated attenuation and
engineering controls; this procedure does not authorise direct viewing or open-beam alignment.

## Measurement Sequence

Display at least three known carrier frequencies spanning the operating point. Measure zero and `+1`
order positions, repeat each measurement at least five times, fit displacement versus carrier, and
record iris centre/radius independently. Preserve images, plane identity, scale and hashes.

## Equation And Units

```text
x_plus1 = wavelength * f_4F * carrier_frequency
f_4F = slope(x_plus1 versus carrier_frequency) / wavelength
```

Positions, focal length and iris radius are in metres; carrier frequency is cycles per metre.

## Repeats And Uncertainty

Use at least five positions per carrier and three carrier values. Propagate position scale,
localisation, regression slope, wavelength and repeatability uncertainty.

## Acceptance Criteria

Linear carrier-displacement fit with no unresolved order ambiguity; operating order inside the
measured iris with documented zero-order clearance; finite uncertainty for focal length, order
position and iris radius.

## Output Format

Use `calibration/templates/objective_relay_template.csv` for the fitted focal result and record
accepted values under `fourier_filter.*` in the JSON bundle. Source captures remain under Stage 9A.

## Code Mapping And Claims

These fields validate carrier-order geometry and first-order position uncertainty. The measured 4F
throughput remains a separate entry in `energy_transmission_template.csv`.
