# Stage 9B.0.2 First Bench Screen Package

Stage 9B.0.2 creates the first controlled bench-screen handoff package. It
does not change the Stage 9B.0/9B.0.1 nominal ranking, does not add optical
physics, and does not process raw camera images.

## Why This Exists

The nominal atlas is useful only after the bench records a practical baseline
carrier/stop state. This stage therefore combines:

- Stage 9A.1B downstream carrier/stop capture families D0-D5;
- Stage 9B.0.1 command-mask candidate outputs;
- operator templates for unknown bench values;
- raw-data placeholders that keep camera files immutable.

## Default Candidate Screen

The first-screen schedule is limited to:

- `gaussian_reference`
- `vortex_ell_1`
- `vortex_ell_2`

`vortex_ell_3` and `vortex_ell_4` are optional later extensions only. They are
not part of the default first-screen capture plan.

## Operator-Set Unknowns

The package requires actual SLM IDs, display orientation, phase-LUT/wavelength
setting, carrier orientation, lens/pinhole/camera/axicon/downstream optics
state, exposure/gain, filters, power/energy where available, date, operator,
and run notes. Unknown values are permitted only when explicitly recorded as
`unknown_recorded`.

## Carrier And Camera Boundary

The Fourier-plane stop selects a region of the ideal continuous-ramp spectrum. It is not a simulation of physical pixelated-SLM zero-order leakage, discrete diffraction-order power fractions, or measured selected-order purity.

The downstream camera capture is empirical. It does not directly calibrate the
Fourier plane, selected-order purity, physical order efficiency, or physical 4F
readiness.

## Raw Data Boundary

Raw camera files are immutable source evidence. Save them under the run raw directory, do not modify them in place, do not hash or process them in Stage 9B.0.2, and do not commit raw images by default.

## Unsupported

No pixelated-SLM order physics, physical 4F readiness, camera model, inverse
correction, Zernike fitting, phase diversity, AI, material response, plasma,
thermal, damage, or fused-silica prediction is enabled.
