# Axicon vortex–Bessel z-scan correction

This directory contains the experimental BeamGage z-scan analysis and the q=20
aberration-retrieval/correction pipeline.

## Authoritative correction path

For aberration correction, use:

- `miao_full_retrieval.py` — Miao-style annular inverse retrieval plus the explicit stationary-phase radial-gradient extension used here.
- `run_q20_miao_retrieval.py` — provenance-checked 18-plane × 4-repeat BMG runner.
- `q20_hardware_calibration_template.json` — bench calibration fields that must be measured, never guessed.
- `q20_experimental_acceptance_metrics.py` — independent measured before/after beam-quality metrics.
- `iterative_correction_controller.py` — compatibility shim to `iterative_correction_controller_v2.py`.
- `iterative_correction_controller_v2.py` — native-SLM2 low-gain trial and experimental acceptance controller.

The legacy normalized-z correction file
`UNCALIBRATED_DO_NOT_APPLY_q20_modal_correction.npy` is no longer consumed by
the controller, and a calibrated SLM2 map is never passed through the old
normalized-coordinate preview remapper.

## Bessel-specific inverse model

The angular retrieval follows the structure of Miao et al., *Optics Express*
**30**, 11360–11371 (2022): each measured focal-line plane samples an input
annulus, `k_perp` is optimized before the complex modal coefficients, and the
number of fitted modes is increased until the cost stops improving or reaches a
threshold.

For deliberately programmed vortex charge `q`, this code indexes the incident
aberration by Fourier order `m` and uses Bessel order

```text
n = m - q
```

so the ideal q-th order Bessel term is `m=0`. The reconstructed residual therefore
excludes the programmed `q*theta`: the desired vortex is beam structure, not an
aberration to remove.

The paper gives the stationary-phase annulus mapping. This project additionally
uses the same stationary-phase condition to recover an axisymmetric radial phase
gradient from the fitted plane-by-plane transverse spatial frequency:

```text
rho_z = z * k_perp_opt / k

d psi_rho / d rho = k_perp_nominal - k_perp_opt
```

The second relation is an explicit extension/derivation used by this pipeline;
it should not be described as a directly quoted algorithmic step from Miao et
al. It follows by adding a slowly varying radial residual phase `psi_rho(rho)`
to the stationary-phase exponent and differentiating with respect to `rho`.
Its absolute piston is irrelevant; the gradient is numerically integrated across
the sampled annuli.

The total sampled residual is then the radial term plus the non-axisymmetric
angular residual. Wrapped phase is interpolated through complex unit phasors,
not directly across the `-pi/pi` discontinuity.

## Coordinate handling and stage runout

The BMG loader does **not** recenter one z plane onto another. It registers only
the four repeated captures within the same z position and keeps the complete
scan in one raw-camera coordinate system, preserving genuine beam walk/pointing.

A moving camera/stage can itself have lateral runout, so the preferred calibration
is:

```text
camera_optical_axis_yx_px_by_z
```

containing 18 raw-sensor `[y,x]` optical-axis coordinates measured with an
aligned reference beam at the same 18 stage positions. This lets the inverse
model subtract mechanical camera motion without erasing real Bessel-beam motion.

A single `camera_optical_axis_yx_px` is accepted only if
`camera_optical_axis_single_value_valid_for_all_z=true` has been experimentally
verified. Otherwise the median observed vortex core is diagnostic only and the
hardware path stays blocked.

## Intensity-only conjugate ambiguity

The Bessel intensity inverse problem has a direct/conjugate ambiguity. The code
retrieves the direct-q mathematical branch and only declares the physical branch
resolved when an independent annular input-intensity reference supports either
the direct orientation or the 180-degree-rotated conjugate orientation.

A deliberately known-sign SLM perturbation followed by another capture can be
used as an equivalent experimental branch test if a direct input-plane reference
is unavailable. Until the branch is resolved, no SLM correction trial is allowed.

## Input plane to SLM2

A reconstructed input/axicon-plane phase is **not automatically an SLM2 phase**.
The simple geometric mapper is allowed only when the experiment confirms:

```text
slm2_is_conjugate_to_input_plane = true
```

If true, the measured input-plane metres-per-SLM2-pixel, SLM2 beam centre,
rotation and x/y parity define the map. If false, scale/rotation/parity is
physically insufficient: the complex field must be propagated/back-propagated
through the measured relay. That non-conjugate relay is intentionally not guessed
from nominal drawings or unvalidated 4F values, so application remains blocked
until its real geometry is known.

## Hardware gates

`run_q20_miao_retrieval.py` may fit local per-plane quantities before all bench
calibration exists, but it will not produce a trial-ready SLM2 map until the
relevant measurements are present:

- absolute distance from the axicon/input reference to relative `z=0`;
- intended/calibrated `k_perp_nominal_m_inv`;
- optical axis versus z-stage position;
- direct/conjugate branch resolution;
- confirmation of SLM2 conjugacy or a measured non-conjugate relay model;
- if conjugate: input-plane → SLM2 scale, centre, rotation and parity;
- SLM2 phase response/LUT at 1030 nm.

The code deliberately does not substitute the fitted aberrated `k_perp` for the
nominal target and does not silently assume SLM pixel pitch, relay magnification,
orientation or phase stroke.

## Native SLM2 correction layer

When the full retrieval is calibrated and a valid SLM2 mapping exists,
`slm2_correction_phase_rad.npy` is already in **native SLM2 pixel coordinates**.
The v2 controller keeps that coordinate system unchanged.

A low-gain candidate is saved as a signed phase array in radians. It is **not a
greyscale bitmap**. The lab GUI should add this phase layer to the existing SLM2
programmed phase, wrap the combined phase once, then encode it using the measured
1030-nm LUT through the normal driver.

The default trial is deliberately conservative (normally 5%, with allowed gains
capped at 20%). A model prediction cannot accept its own correction.

## Independent experimental acceptance

Run `q20_experimental_acceptance_metrics.py` on the baseline BMG stack and again
on the newly captured post-mask stack. It compares measured intensity against the
same fixed analytical `J_q(k_perp_nominal r)^2` target and does not consume the
retrieved phase map.

The principal-ring uniformity metric is a true **azimuthal** coefficient of
variation: the same narrow radial samples are averaged at every theta before the
CV is calculated. This avoids falsely treating the ideal radial Bessel variation
inside a thick annulus as angular non-uniformity.

Each metrics CSV carries the fixed target/calibration provenance and a SHA-256
hash of the complete BMG dataset. The controller requires:

- exactly matching z planes;
- identical `q`, nominal `k_perp`, wavelength, pixel pitch, ROI and axis-calibration provenance;
- **different** BMG SHA-256 fingerprints before and after.

The same camera stack therefore cannot accidentally pass as a new experiment.
The current acceptance gates require improved median correlation and RMSE,
≥5% reduction in median azimuthal ring CV, and no material degradation of the
dark vortex core.

## Calibration file

Copy:

```text
q20_hardware_calibration_template.json
```

to:

```text
q20_hardware_calibration.json
```

and fill only measured values. `null` values intentionally block hardware use.

## Running

With all 72 raw BMG files in `z-scan 2 1010` beside the scripts:

```powershell
python run_q20_miao_retrieval.py
```

Once a calibration file exists, baseline acceptance metrics can be generated with:

```powershell
python q20_experimental_acceptance_metrics.py
```

The full retrieval output includes:

- `per_plane_retrieval.csv` — per-plane optimized `k_perp`, adaptive modal order and fit quality;
- `frame_qc_preserved_coordinates.csv` — raw core positions, repeat shifts and crop provenance;
- `rho_sampled_m.npy`;
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
pipeline. The exact-conjugate final column in a model closure test is not evidence
of experimental correction.

`q20_modal_analysis.py` remains legacy/diagnostic. Its normalized z-order phase
rendering must not be promoted as an SLM correction.

`slm2_complete_mask_preview.py` is also a legacy/nominal visualizer from the old
normalized-coordinate workflow. It is not used by the calibrated controller.

## Tests and CI

`tests/test_miao_full_retrieval.py` covers synthetic `k_perp` recovery,
stationary-phase radial reconstruction, q-vortex exclusion, conjugate-branch
logic, wrapped-phase interpolation, native-SLM2 phase preservation, z-dependent
camera-axis/runout calibration, independent ideal-Bessel acceptance metrics,
and rejection of reused camera datasets or mismatched target provenance.

The q20 GitHub Actions workflow compiles the correction modules and runs those
tests together with the earlier phase-physics regression tests before writing a
passing commit marker.

## Reference

B. Miao, L. Feder, J. E. Shrock, and H. M. Milchberg, “Phase front retrieval and
correction of Bessel beams,” *Optics Express* **30**(7), 11360–11371 (2022),
DOI: 10.1364/OE.454796.
