# Stage 9A.1B Downstream-Focus Carrier and Stop Characterisation

The installed laboratory camera is at a downstream final-focus/output plane, not
at the physical Fourier plane. Stage 9A.1B therefore adds a second calibration
mode, `downstream_focus_empirical`, while preserving the existing
`direct_fourier_plane_access` mode for later temporary access at or conjugate to
the Fourier plane.

## Why Downstream And Fourier-Plane Measurements Differ

Direct Fourier-plane access can observe zero/+1/-1 order positions in camera
pixels and can later support physical Fourier-plane calibration after scale and
geometry are known. The installed downstream camera sees the result of the
complete route after the stop and all downstream optics. It can show how the
final output changes with carrier and stop settings, but it cannot identify
physical order coordinates or stop radius in Fourier-plane units.

## What The Installed Camera Can Establish

- final output centroid in camera pixels;
- final output morphology;
- relative transmitted intensity;
- saturation and clipping flags;
- empirical sensitivity to carrier settings;
- empirical sensitivity to stop x/y/radius settings;
- repeatable usable operating points.

## What Requires Temporary Fourier-Plane Access

- direct Fourier-plane order positions;
- physical Fourier-plane x/y coordinates;
- direct stop radius in Fourier-plane mm;
- order-power fractions at the stop;
- physical 4F readiness marked ready from measured Fourier-plane geometry.

Direct Fourier-plane mapping requires a temporary diagnostic method at or
conjugate to the Fourier plane. The installed downstream camera supports
empirical carrier-and-stop response characterisation only.

## Evidence Storage

The downstream package records `calibration_mode`, camera-plane relationship,
Fourier-stop state, axicon state, downstream optics state, carrier command
cycles, SLM mask IDs, exposure/gain, filters, energy setting if available, and
operator notes. Unknown values may remain unknown, but their status is recorded.

The output summary is labelled with:

```text
empirical_downstream_operating_point
not_direct_fourier_plane_calibration
not_physical_4f_model_validation
```

## Physical-4F Readiness Impact

Downstream empirical evidence supports practical operating-point selection,
repeatability assessment, and later comparison against a physical 4F model. It
does not by itself support `physical_fourier_plane_coordinate_calibrated` or
`physical_4f_readiness_ready`.

Warning: With the installed downstream camera, this experiment characterises the final response to carrier and stop settings. It does not directly measure Fourier-plane order positions or by itself calibrate the physical 4F coordinate system.
