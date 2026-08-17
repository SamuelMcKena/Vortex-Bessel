# Stage 8C.3R.2 Annular Axis Tracking and Free-Space Sensitivity Lock

Stage 8C.3R.2 is a free-space reference-plane optical diagnostic lock. The
model is still `n = 1.0`, post-objective/intended sample entrance in air, with
`final_export_allowed=False`. It does not introduce a material interface,
writing model, dose model, plasma model, thermal model, calibrated tolerance, 3D
visualisation, or GUI work.

## Why Raw Peak Fails

The vortex/Bessel reference field is annular. Its brightest pixel is an
azimuthally degenerate sample of the bright ring, not the physical beam axis. A
clean symmetric run can move the raw maximum to a different angular point on the
same ring when the field of view or sampling changes. That movement is real as a
display diagnostic, but it is not beam steering and must not drive FOV
reliability, tilt validation, or headline axis metrics.

The required raw-peak status label is:

```text
not_a_primary_axis_metric_for_annular_fields
```

## Estimator Hierarchy

`vbb_study.digital_twin.annular_axis_tracking.estimate_annular_axis` exposes all
candidate centres for each transverse plane:

- commanded axis
- fitted annular ring centre, ring radius, ring-fit quality, ring circularity,
  and azimuthal uniformity
- fitted dark-core centre, core-fit quality, core-fill fraction, and central
  darkness contrast
- central ROI intensity centroid
- optional phase-singularity estimate when a complex transverse field is passed
- raw brightest pixel, labelled diagnostic-only

The selected `beam_axis_*` estimate follows this hierarchy:

1. fitted annular ring centre
2. fitted dark-core centre
3. central ROI intensity centroid
4. phase singularity when a valid complex phase/null estimate is supplied
5. no reliable axis if the above fail

The raw brightest pixel is never selected as the primary annular axis.

## Confidence And Reliability

The estimator reports `beam_axis_method`, `beam_axis_fit_quality`, and
`beam_axis_reliability`. Good annular planes report `reliable`; deformed or
crop-limited planes report `caution`, `caution_crop_limited`, or
`axis_estimate_unreliable`. The trajectory fit rejects weak, crop-limited, or
unreliable planes instead of manufacturing a precise axis.

`track_axis_trajectory` fits `x_axis(z)` and `y_axis(z)` from the selected
annular-axis estimate and reports:

- `axis_intercept_at_z0_x_um`, `axis_intercept_at_z0_y_um`
- `reference_plane_axis_error_um`
- `beam_steering_angle_x_mrad`, `beam_steering_angle_y_mrad`
- `trajectory_fit_quality`
- `valid_z_fit_range_um`
- `valid_plane_fraction`

Beam tilt validation compares the measured fitted-axis slope with the free-space
relation `dx/dz = kx/kz`, `dy/dz = ky/kz`.

## FOV Convergence

FOV convergence now reports both primary and diagnostic differences:

- `ring_centre_difference_um`
- `core_centre_difference_um`
- `axis_trajectory_difference_um`
- `axis_error_difference_um`
- `raw_peak_position_difference_um`

Reliability decisions use ring/core convergence, trajectory convergence,
peak-fluence convergence, captured-power drift, FOV margin, and out-of-frame
fraction. Raw peak movement is reported but cannot make a clean annular field
look shifted. Deliberately undersized FOV cases are labelled
`invalid_out_of_frame`.

## Diagnostic Response Curves

The response-curve framework sweeps ten free-space perturbation families from a
central configuration:

- vortex-centre offset with axicon fixed
- axicon-centre offset with vortex fixed
- input beam decentre relative to fixed active area
- input beam tilt
- pupil decentre/clipping
- defocus
- astigmatism
- coma
- spherical aberration
- zero-order leakage fraction

Each row carries transmitted fraction, reference-plane pulse energy, beam-axis
error, azimuthal uniformity, ring fit/circularity, residual shape deformation,
core contamination/fill, central darkness contrast, peak fluence per
reference-plane energy, and numerical reliability.

All response plots and tables are labelled:

```text
Diagnostic sensitivity sweep. Not an experimentally measured laboratory tolerance.
```

These curves are diagnostic rankings in the free-space optical model. They are
not acceptance criteria, calibration tolerances, or material-response predictions.

## Warning-Only And Future Controls

The direct component-plane engine still does not represent a 4F Fourier filter
plane, relay imaging plane, mask rotation resampling, physical axicon apex
defect, stochastic pointing/stage jitter, focus drift ensembles, material
interfaces, dose accumulation, nonlinear propagation, thermal response,
microscope response, or calibrated writing prediction. Those controls remain
warning-only or future-stage.

Stage 8D remains out of scope until the C3R.2 diagnostic lock is accepted.
