# 01 Beam Radius on the SLM

## Purpose

Measure the incident `1/e^2` intensity radii `wx` and `wy` at the defined SLM plane, with traceable
scale and uncertainty. A screenshot without calibrated scale is not acceptable.

## Required Equipment

An approved beam profiler or calibrated camera/relay, suitable attenuation and diagnostics, the
Stage 9A acquisition package, and the laboratory's authorised laser-safety controls.

## Beam-Safe Setup

Follow the formal laser safety procedure and approved risk assessment. Use engineering controls,
enclosures, rated attenuation, and authorised personnel. Do not use this procedure as an open-beam
alignment instruction.

## Measurement Sequence

Acquire background and unsaturated beam images at the SLM reference plane. Record scale, exposure,
attenuation, plane identity, and checksums. Acquire at least five independent frames after stable
operation. Fit each frame and retain residual and saturation diagnostics.

## Equation And Units

Fit SI-coordinate intensity to

```text
I(x,y) = I0 exp(-2[((x-x0)/wx)^2 + ((y-y0)/wy)^2]) + background
```

Report `wx`, `wy`, their standard uncertainties and covariance in metres; also report residual RMS,
fit crop in pixels, and saturation fraction.

## Repeats And Uncertainty

Use at least five independent captures. Combine fit covariance with repeatability and calibrated
pixel-scale uncertainty; state whether the reported value is a standard uncertainty.

## Acceptance Criteria

No saturated fit pixels; stable centre and radii; no structured residual that invalidates the
elliptical Gaussian model; complete plane/scale/checksum provenance. A failed model fit remains a
measured diagnostic and does not unlock the claim.

## Output Format

Populate `calibration/templates/beam_radius_measurement_template.csv` and then map the accepted
mean and uncertainty into `laser.beam_radius_on_slm_m` in the JSON bundle.

## Code Mapping And Claims

The value enters `vbb_study.calibration.schema` and the Bessel-zone uncertainty calculation. It
contributes to pupil-fill and Bessel-zone readiness; it does not alone calibrate sample dimensions.
