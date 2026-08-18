# Axicon vortex–Bessel z-scan correction

This directory contains the experimental BeamGage z-scan analysis and the q=20
aberration-retrieval/correction pipeline.

## Authoritative correction path

For aberration correction, use:

- `miao_full_retrieval.py` — full Miao-style inverse physics.
- `run_q20_miao_retrieval.py` — 18-plane × 4-repeat BMG runner.
- `q20_hardware_calibration_template.json` — bench calibration fields that must be measured, not guessed.
- `iterative_correction_controller.py` — compatibility shim that now routes to `iterative_correction_controller_v2.py`.
- `iterative_correction_controller_v2.py` — low-gain closed-loop trial/acceptance controller.

The legacy normalized-z correction file
`UNCALIBRATED_DO_NOT_APPLY_q20_modal_correction.npy` is no longer consumed by
the controller.

## What changed physically

The retrieval now follows the two parts of the Bessel phase-retrieval method in
Miao et al., *Optics Express* **30**, 11360–11371 (2022):

1. **Each measured z plane gets its own optimized transverse wavenumber**
   `k_perp_opt(z)`. A global ring-derived value is used only as an optimizer seed.
2. **The complex Bessel modal coefficients are then fitted with increasing
   aberration order** until the cost stops improving or reaches the threshold.

For the deliberately programmed vortex charge `q`, the aberration Fourier order
is represented by `m`, with Bessel order `n = m - q`; the ideal q-th order Bessel
term is `m=0`. The inverse angular series therefore reconstructs the incident
annular field with the programmed `q*theta` removed. The target vortex is never
put into the aberration correction.

The stationary-phase mapping used for each fitted plane is

```text
rho_z = z * k_perp_opt / k
```

and the radial phase error is recovered from

```text
d psi_rho / d rho = k_perp_nominal - k_perp_opt.
```

The radial gradient is integrated across the sampled annuli and added to the
non-axisymmetric angular residual. Wrapped phases are interpolated through unit
phasors, not directly as numbers.

## Coordinate handling

The new BMG loader **does not recenter every z plane**. It removes only
repeat-to-repeat acquisition jitter at a given z and crops all planes in the same
raw camera coordinate system. Genuine beam translation/pointing along the scan
is therefore retained for the inverse problem.

## Conjugate ambiguity

Intensity-only Bessel retrieval has the `U` / `U*` ambiguity described by Miao
et al. The conjugate solution corresponds to a 180-degree-rotated input
intensity. The new code will only choose a branch when an independent annular
input-intensity reference is supplied. Otherwise the branch is marked
`unresolved` and SLM output is blocked.

A known-sign SLM perturbation plus a second capture can be used as an equivalent
experimental branch check if a direct input-plane intensity reference is not
available.

## Hardware gates

`run_q20_miao_retrieval.py` will fit the local per-plane quantities even when
hardware calibration is incomplete, but it will not create a trial-ready SLM2
map until all of the following are known:

- absolute distance from the axicon/input reference to relative `z=0`;
- conjugate-branch choice from an independent reference/known-sign test;
- measured input-plane → SLM2 scale;
- SLM2 rotation, parity and beam centre;
- SLM2 phase response/LUT at 1030 nm.

The transform to SLM2 uses a measured end-to-end `input_plane_m_per_slm2_pixel`
so the code does not silently assume a nominal SLM pixel pitch or a relay
magnification.

After those gates are satisfied, the v2 controller can propose only a **low-gain
trial**. That proposal is not accepted from the model. A new, identically sampled
18×4 BMG z-stack must be captured and pass the experimental before/after gates.

## Calibration file

Copy:

```text
q20_hardware_calibration_template.json
```

to:

```text
q20_hardware_calibration.json
```

and fill only measured values. `null` values intentionally block the hardware
path.

## Running the full retrieval

With the raw 72 BMG files in `z-scan 2 1010` beside the script:

```powershell
python run_q20_miao_retrieval.py
```

The runner writes under `outputs/miao_full_q20/`:

- `per_plane_retrieval.csv` — per-plane `k_perp_opt`, adaptive modal order and fit metrics;
- `frame_qc_preserved_coordinates.csv` — raw camera core positions and repeat QC;
- `rho_sampled_m.npy` — sampled input-annulus radii when absolute z is calibrated;
- `radial_phase_gradient_rad_per_m.npy`;
- `radial_phase_rad.npy`;
- `angular_phase_rows_rad.npy`;
- `retrieved_full_residual_phase_input_plane_rad.npy`;
- `conjugate_correction_input_plane_rad.npy`;
- `slm2_correction_phase_rad.npy` only after branch + coordinate calibration;
- `correction_manifest.json` — authoritative readiness/blocker state.

## Presentation / diagnostic path

`rebuild_q20_presentation_from_bmg.py`, `q20_phase_physics.py` and
`single_transverse_phase_forward_test.py` remain useful for measured XZ/YZ and
forward-model presentation diagnostics. They are not the hardware correction
pipeline.

`q20_modal_analysis.py` remains a legacy/diagnostic modal analysis. Its
normalized z-order phase rendering must not be promoted as an SLM correction.

## Tests

`tests/test_miao_full_retrieval.py` checks the per-plane k-perp optimization,
stationary-phase radius mapping, radial-gradient recovery, q-vortex exclusion,
conjugate-branch logic, phasor interpolation and the removal of the legacy map
from the active controller. The q20 CI workflow compiles the full pipeline and
runs these tests together with the earlier phase-physics tests.

## Reference

B. Miao, L. Feder, J. E. Shrock, and H. M. Milchberg, “Phase front retrieval and
correction of Bessel beams,” *Optics Express* **30**(7), 11360–11371 (2022),
DOI: 10.1364/OE.454796.
