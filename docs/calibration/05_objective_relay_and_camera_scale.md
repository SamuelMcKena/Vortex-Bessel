# 05 Objective, Relay and Camera Scale

## Purpose

Measure effective objective/relay geometry and object-plane camera scale needed for absolute focal
dimensions and image-to-model comparison.

## Required Equipment

A traceable calibration target or displacement standard, camera, objective/relay hardware,
appropriate enclosed illumination and attenuation, Stage 9A capture tools, and authorised safety
controls.

## Beam-Safe Setup

Follow the formal laser safety procedure and use an approved target-imaging configuration. Use
engineering controls and authorised personnel. Do not perform open-beam alignment from this text.

## Measurement Sequence

Record target images at several known spacings and positions. Measure objective focal geometry,
effective pupil radius, pupil fill, relay magnification, camera rotation and centre using documented
plane identities. Acquire at least five independent images per setting and retain raw hashes.

## Equation And Units

```text
scale_object = known_spacing_m / delta_pixels
u(scale)^2 = (u(spacing)/delta_pixels)^2
             + (spacing*u(delta_pixels)/delta_pixels^2)^2
magnification = camera_pixel_pitch_m / scale_object
```

Lengths are in metres, scale in metres per pixel, angles in degrees, and NA/magnification are
dimensionless.

## Repeats And Uncertainty

Use at least five captures at three target positions. Include target certificate, localisation,
distortion, repeatability, pixel-pitch and regression uncertainty. Report spatial nonuniformity.

## Acceptance Criteria

No unresolved distortion over the analysis crop; independent scale estimates agree within combined
uncertainty; objective pupil and fill are measured at the governed plane; plane mapping is explicit.

## Output Format

Populate `camera_scale_template.csv` and `objective_relay_template.csv`; map accepted values to
`camera.*`, `objective.*`, and `relay.magnification` in the JSON bundle.

## Code Mapping And Claims

Together with wavelength, these measurements unlock calibrated transverse dimensions and H1 focal
detail. Absolute fluence additionally requires calibrated energy and transmission.
