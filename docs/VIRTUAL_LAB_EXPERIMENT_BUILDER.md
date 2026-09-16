# Virtual Lab experiment builder

This extends the existing Virtual Lab / blind-correction workflow into a configurable offline experiment rather than a fixed seeded demo.

## What can be perturbed together

The **Perturbation Builder** can apply multiple hidden faults simultaneously:

- input Gaussian X/Y radius scale (ellipticity)
- input beam X/Y decentre
- input beam X/Y pointing
- finite X/Y wavefront curvature
- hidden low-order defocus, astigmatism, coma and spherical terms
- physical-axicon X/Y decentre

All of these values belong to the virtual bench truth. They are not passed to the optimiser. The optimiser receives only intensity images, public z positions and the SLM commands that it deliberately issued.

The nominal incident 1/e field-amplitude radius is also editable. This remains a model input, not a measured bench calibration, until bound experimentally.

## Measurement plan

The operator can define an explicit comma-separated z plan as before, or build an evenly spaced plan from:

- z start
- z stop
- number of distinct z planes
- repeats per plane

The GUI reports the expected numerical-camera frame count before a correction run and enforces a maximum frame budget for auto-convergence.

## SLM correction authority

The SLM-only correction basis is operator-selectable. The default set is:

- tip/tilt X and Y
- defocus
- astigmatism X and XY
- coma X and Y
- spherical

Tip/tilt is intentionally included before mechanical Alignment Assist. Some apparent alignment / beam-walk errors can be reduced by programmable first-order phase steering and should be tested with the SLMs before declaring a physical move necessary.

This does **not** imply that every mechanical fault is uniquely or perfectly phase-correctable. For example, a displaced axicon may be only partially compensated by upstream phase control. The purpose of the experiment is to measure that correction envelope.

## Auto-converge SLM-only

**Start auto-converge SLM-only** repeatedly runs the existing sensorless, candidate-measurement, fresh-verification and rollback logic. Each completed cycle begins from the already accepted correction state, then the probe amplitude is reduced for the next cycle.

The run stops when one of the following occurs:

- fractional objective improvement stays below the configured threshold for the configured patience count
- maximum correction cycles are reached
- maximum camera-frame budget would be exceeded
- the operator cancels

Mechanical Alignment Assist is never enabled automatically by this workflow.

## Suggested offline experiment

1. Select **VIRTUAL LAB**.
2. Configure the intended SLM beam state (for example the vortex charge and any nominal shaping terms).
3. In **Perturbation Builder**, enable several faults at once; for example:
   - beam X offset = +80 µm
   - pointing Y = -0.08 mrad
   - astigmatism X = +0.15 waves
   - coma Y = -0.10 waves
   - axicon X decentre = +35 µm
4. Apply the perturbations.
5. Choose the z range, number of planes and repeats.
6. Leave tip/tilt enabled in SLM correction authority if you want the SLMs to attempt alignment-like compensation.
7. Capture a baseline z stack if desired.
8. Press **Start auto-converge SLM-only**.
9. Watch the same virtual camera viewer update as candidate corrections are applied and re-measured.
10. Inspect the objective history and accepted coefficients.
11. Only after the SLM-only plateau, optionally run **Alignment Assist** to quantify the residual benefit of a physical axicon move.
12. Reveal truth only for post-run validation.

## Claim boundary

This remains an offline control-space / algorithm-development model. It does not convert currently unmeasured SLM spacing, pinhole axial position, axicon angle convention or physical camera z reference into bench truth. Simulated frames remain clearly classified as simulated.
