# Quantitative beam metrics

All metrics consume the untouched `CameraFrame.data` matrix. A robust border
background and noise estimate is applied to a separate float64 analysis copy.
Display colour, log scale, gamma, zoom and overlays never enter the metric engine.

## Generic metrics

- background estimate;
- total signal / power proxy;
- peak;
- saturation fraction against the declared full scale;
- edge-power fraction and clipping flag;
- optional image correlation;
- repeat count, centre RMS scatter and power coefficient of variation.

## Central / q=0 metrics

- intensity centroid;
- x/y and principal-axis second-moment widths;
- x/y and principal FWHM equivalents;
- ellipticity, principal-axis direction and a covariance symmetry score;
- radial profile.

These moment/FWHM values describe the measured distribution. They are not a
claim that every Bessel or clipped beam is exactly Gaussian.

## Vortex / annular metrics

- gradient-symmetry annulus centre with explicit centroid fallback warning;
- principal-ring radius from the radial profile;
- ring eccentricity;
- azimuthal coefficient of variation and ring-uniformity transform;
- dark-core mean relative to principal-ring mean;
- ring-power fraction;
- radial profile and optional normalised radial-profile L1 error;
- generic power, saturation and clipping metrics.

The family is selected from `ExperimentState.effective_vortex_charge()`. Any
nonzero effective charge uses annular analysis; q=20 has no special metric branch.

## Coordinates and units

Single-frame centres are `(y, x)` in native camera pixels. Beam walk converts
pixel displacement per millimetre using the configured `pixel_size_um`. A pixel
size copied from a datasheet is still calibration evidence that must be recorded;
it is not silently inferred from an image.

## Current limitations

- The annular centre assumes enough rotational gradient structure is visible.
- Saturation uses the configured numeric full scale; camera processing/export
  must be fixed and documented.
- Camera aliasing, optical asymmetry, pointing and detector artifacts remain
  separate hypotheses. A better-looking rendered ring is not a metric.
- Acceptance criteria belong to a recipe/calibration, not to the generic engine.
