# Axicon vortex–Bessel z-scan correction

This directory contains the experimental BeamGage z-scan analysis and the q=20
aberration-retrieval/correction pipeline.

## Authoritative correction path

For aberration correction, use:

- `miao_full_retrieval.py` — full Miao-style inverse physics.
- `run_q20_miao_retrieval.py` — 18-plane × 4-repeat BMG runner.
- `q20_hardware_calibration_template.json` — bench calibration fields that must be measured, not guessed.
- `iterative_correction_controller.py` — compatibility shim that routes to `iterative_correction_controller_v2.py`.
- `iterative_correction_controller_v2.py` — native-SLM2 low-gain closed-loop trial/acceptance controller.

The legacy normalized-z correction file
`UNCALIBRATED_DO_NOT_APPLY_q20_modal_correction.npy` is no longer consumed by
the controller, and the calibrated SLM2 map is no longer passed through the old
normalized-coordinate preview remapper.

## What changed physically

The retrieval follows the Bessel phase-retrieval structure of Miao et al.,
*Optics Express* **30**, 11360–11371 (2022):

1. **Each measured z plane gets its own optimized transverse wavenumber**
   `k_perp_opt(z)`. A global ring-derived value is only an optimizer seed.
2. **The complex Bessel modal coefficients are fitted with increasing aberration
   order** until the cost reaches its threshold or further modes stop improving it.

For deliberately programmed vortex charge `q`, aberration Fourier order is `m`
and Bessel order is `n = m - q`; the ideal q-th order component is `m=0`.
Reconstructing the annular input field therefore factors the programmed
`q*theta` out of the residual. The target vortex is never placed in the
aberration correction.

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
phasors, not directly as wrapped numbers.

## Coordinate handling

The BMG loader **does not recenter every z plane**. It removes only
repeat-to-repeat acquisition jitter at a given z and crops all planes in one raw
camera coordinate system, retaining genuine beam walk/pointing.

For a trial-ready retrieval, `camera_optical_axis_yx_px` must be measured in raw
camera coordinates. Until then the median observed beam core is only a diagnostic
axis estimate and hardware application stays blocked.

## Conjugate ambiguity

Intensity-only Bessel retrieval has a `U` / `U*` ambiguity. The code chooses a
branch only when an independent annular input-intensity reference is supplied.
Otherwise the branch is `unresolved` and SLM output is blocked.

A known-sign SLM perturbation plus a second capture can serve as an equivalent
experimental branch test if a direct input-plane reference is unavailable.

## Input-plane to SLM2 physics

A reconstructed input/axicon-plane phase is **not automatically an SLM2 phase**.
The simple geometric mapper is allowed only when the experiment confirms that
SLM2 is conjugate to the reconstructed input plane. The calibration file therefore
contains:

```text
slm2_is_conjugate_to_input_plane
```

- If `true`, the measured input-plane metres-per-SLM2-pixel, centre, rotation and
  parity define the geometric map.
- If `false`, a scale/rotation/parity remap is physically insufficient. The
  complex field must instead be propagated/back-propagated through the measured
  relay to the SLM2 plane. That non-conjugate relay solver is intentionally not
  guessed by the current code, so the hardware path remains blocked.
- If `null`, conjugacy has not been established and the path is also blocked.

## Hardware gates

`run_q20_miao_retrieval.py` may fit local per-plane quantities while calibration
is incomplete, but it will not create a trial-ready SLM2 map until all relevant
items are known:

- absolute distance from the axicon/input reference to relative `z=0`;
- intended/calibrated `k_perp_nominal_m_inv`;
- raw-camera optical axis;
- conjugate-branch choice from an independent reference/known-sign test;
- confirmation that SLM2 is conjugate to the reconstructed plane **or** a measured
  relay propagation model for a non-conjugate plane;
- if conjugate: measured input-plane → SLM2 scale, centre, rotation and parity;
- SLM2 phase response/LUT at 1030 nm.

The transform to SLM2 uses a measured end-to-end `input_plane_m_per_slm2_pixel`;
it does not silently assume nominal SLM pixel pitch or relay magnification.

## Native SLM2 correction layer

When the full retrieval is calibrated and the geometric mapping is valid,
`slm2_correction_phase_rad.npy` is already in **native SLM2 pixel coordinates**.
The v2 controller keeps that coordinate system unchanged. It does not send the
map through `slm2_complete_mask_preview.py` or any second radial remapping.

The low-gain candidate is saved as a signed phase array in radians. It is **not a
greyscale bitmap**. The lab GUI should add this correction layer to the existing
programmed SLM2 phase, wrap the combined phase once, then apply the independently
measured 1030-nm LUT through the normal SLM driver.

After a low-gain trial, a new identically sampled 18×4 BMG z-stack must be
captured. The model cannot accept its own proposed correction.

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
- `rho_sampled_m.npy` — sampled input-annulus radii once absolute z is calibrated;
- `radial_phase_gradient_rad_per_m.npy`;
- `radial_phase_rad.npy`;
- `angular_phase_rows_rad.npy`;
- `retrieved_full_residual_phase_input_plane_rad.npy`;
- `conjugate_correction_input_plane_rad.npy`;
- `slm2_correction_phase_rad.npy` only after branch + valid SLM2-plane mapping;
- `correction_manifest.json` — authoritative readiness/blocker state.

## Presentation / diagnostic path

`rebuild_q20_presentation_from_bmg.py`, `q20_phase_physics.py` and
`single_transverse_phase_forward_test.py` remain useful for measured XZ/YZ and
forward-model presentation diagnostics. They are not the hardware correction
pipeline.

`q20_modal_analysis.py` remains legacy/diagnostic. Its normalized z-order phase
rendering must not be promoted as an SLM correction.

`slm2_complete_mask_preview.py` is also a legacy/nominal visualizer for the old
normalized-coordinate workflow. It is not used by the calibrated v2 controller.

## Tests

`tests/test_miao_full_retrieval.py` checks per-plane k-perp optimization,
stationary-phase radius mapping, radial-gradient recovery, q-vortex exclusion,
conjugate-branch logic, phasor interpolation, native-SLM2 phase preservation,
conjugacy/axis gates, and removal of the legacy correction/remapping path.
The q20 CI workflow compiles the full pipeline and runs these tests together with
the earlier phase-physics tests.

## Reference

B. Miao, L. Feder, J. E. Shrock, and H. M. Milchberg, “Phase front retrieval and
correction of Bessel beams,” *Optics Express* **30**(7), 11360–11371 (2022),
DOI: 10.1364/OE.454796.
