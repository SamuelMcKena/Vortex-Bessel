# Beam-walk measurement

For repeated observations at physical z coordinates, `fit_beam_walk()` averages
each plane and fits:

\[
x(z)=x_0+s_x z, \qquad y(z)=y_0+s_y z.
\]

It reports:

- every frame centre;
- per-plane mean and repeat standard deviation;
- intercepts and x/y slopes in pixels per millimetre;
- x/y and total small-angle slopes in mrad using camera µm/pixel;
- direction;
- x/y fit R²;
- repeat scatter;
- mean ring radius and power versus z when available.

Because `(px/mm) × (µm/px) = µm/mm`, that numeric slope equals mrad in the
small-angle convention.

## Required scientific label

Until camera travel has been independently calibrated parallel to the desired
optical axis, the result is:

> measured beam-camera relative propagation mismatch

It is not automatically an absolute laser angle. When the calibration registry
contains a valid camera-travel-axis calibration, the report can label the slope
against the calibrated optical axis.

## q=0 versus vortex comparison

`compare_beam_walk()` calculates the vector difference between q=0 and vortex
slopes using a recipe-configured materiality threshold:

- similar slopes: primarily common post-axicon or measurement geometry;
- materially different slopes: possible vortex, SLM or asymmetric contribution.

This is a diagnostic classification, not unique causal proof.

## Interfaces

- Core: `labcontrol.beam_walk`
- General CLI: `run_labcontrol_cli.py analyse beam-walk ...`
- Existing BMG convenience tool: `tools/analyze_beam_walk.py`; it now calls the
  same reusable centre estimator and fitter.
- GUI: formal captures appear in the Sessions comparison page; once two physical
  z planes exist, the dedicated propagation view plots x centre, y centre and
  ring radius versus z and displays the fit label/total slope.
