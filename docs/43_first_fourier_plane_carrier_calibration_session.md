# Stage 9A.1 — First Fourier-Plane Carrier Calibration Session

A small, lab-ready pack to measure, in the first physical CSLM → 4F session, how a command-domain
SLM2 carrier ramp moves the diffraction orders at the accessible Fourier plane. This stage creates
files and a bench procedure only — no physical 4F propagation, camera physics, physical-axicon
propagation, inverse/correction/AI, or material model.

Stage 9A.1 is the **direct Fourier-plane access** mode. It requires a temporary diagnostic method
at or conjugate to the Fourier plane, such as a movable camera, beam profiler, IR card, or power
meter. The currently installed downstream final-focus camera is not direct Fourier-plane access.
For the installed setup, use Stage 9A.1B downstream mode: it supports empirical carrier-and-stop
response characterisation only, not physical Fourier-plane order-position calibration.

Boundary: `physical_4f_filter_modelled=False`; `camera_model_enabled=False`;
`material_model_enabled=False`; `diagnostic_only=True`; `final_export_allowed=False`.

## Why command-domain cycles (not physical frequency)

The physical SLM pixel pitch and the physical Fourier-plane mapping are **not yet calibrated**
(docs/41 level C is blocked). A carrier is therefore defined in the **command domain** as signed
cycles across the displayed active SLM width/height:

```
phi_x = 2*pi * N_x * (pixel_x / display_width_pixels)
phi_y = 2*pi * N_y * (pixel_y / display_height_pixels)
phi   = wrap(phi_x + phi_y)
```

`N_x`, `N_y` are signed integer cycles across the displayed command area. These are **not**
`cycles_per_mm` / `cycles_per_m` and are never labelled as physical Fourier-plane spatial
frequencies until SLM geometry and Fourier-plane calibration exist. Metadata carries
`physical_frequency_status = uncalibrated_command_domain`.

## Phase-mask conventions

- SLM1 stays flat. SLM2 carries only a wrapped carrier ramp — no vortex, axicon, correction map,
  piston (beyond optional zero), or hidden aperture crop.
- Phase is wrapped to `[0, 2π)`; quantised with the existing project convention
  (`quantize_phase_rad` / `phase_to_gray`, 8-bit grayscale by default).
- Each mask exports a phase `.npy` (radians), a quantised `.npy`, a display-safe grayscale PNG, and
  a metadata JSON (mask id, SLM id, command display width/height, carrier cycles x/y, pixels/cycle,
  phase-wrap convention, quantisation levels, `phase_response_calibration_status`,
  `physical_frequency_status`, coordinate frame `SLM2_phase_map_frame`, export checksum, timestamp,
  git commit).
- The exported grayscale is **not** asserted to produce a calibrated 0–2π response at 1030 nm;
  `phase_response_calibration_status = unknown_or_unverified`.

### Sampling guard

`minimum_pixels_per_carrier_cycle` (default 8). A requested carrier whose `display/|cycles|` is
below this fails validation with a clear message; the pattern is never silently aliased.

## First-session capture plan (`cslm_fourier_carrier_calibration_minimal_v1.json`)

Default command-domain sweep: `dark_frame_repeats=5`, `flat_reference_repeats=3`,
`carrier_cycles=[-24,-16,-8,0,8,16,24]`, `command_axes=["x","y"]`, optional diagonals
`(8,8),(8,-8),(-8,8),(-8,-8)`, `capture_repeats=1`. Each carrier capture pairs SLM1-flat with one
SLM2 carrier mask; dark and zero-carrier reference captures bracket the sweep.

## Session package (`create_fourier_carrier_calibration_session`)

`outputs/calibration_runs/<run_id>/`: `run_manifest.json`, `acquisition_plan.csv`,
`capture_manifest_template.csv`, hardware/bench/coordinate snapshots, `phase_masks/slm1`,
`phase_masks/slm2`, `figures/command_domain_carrier_mask_atlas.png`, and `experiment_package/`
(`LAB_README_FIRST_FOURIER_SESSION.md`, bench setup sheet md+csv, camera capture checklist,
carrier sweep log, Fourier-plane observation template, operator notes). Raw subdirs under
`data/calibration_runs/<run_id>/{raw,manifests,derived,figures}`.

## Raw-data storage rules

Raw camera files are immutable source evidence; the generator creates empty `raw/` dirs and never
fabricates capture files. Generated run packages are not committed by default
(`outputs/calibration_runs/` and `data/calibration_runs/` `.gitignore`). Derived preprocessing,
if any, is recorded separately and never overwrites raw (Stage 9A policy).

## First lab procedure (LAB_README)

1. Record profile/run ID and physical bench state. 2. Capture 5 dark frames at the intended
exposure/gain. 3. Display SLM1-flat / SLM2-flat and capture the zero-carrier reference (×3).
4. Run the command-x carrier sequence. 5. Run the command-y carrier sequence. 6. Optional diagonals
if time permits. 7. Per capture record exposure, gain, camera location, visible orders, stop state,
clipping/saturation. 8. Do not move camera/SLMs/lenses without logging. 9. Keep raw files unchanged
under the run ID. 10. Complete the capture manifest before leaving. **Look for**: zero-order
position; +1/−1 positions; order movement under sign reversal; order separation vs command cycles;
saturation; clipping; unexpected multiple orders; asymmetry/rotation.

## How direct-mode results feed Stage 9B and unblock physical-4F readiness

The measured (carrier cycles → observed order position) relation, together with recorded SLM
geometry and the Fourier-plane scale, provides the **Fourier-plane physical-position coordinate
convention and carrier sign** — exactly the items blocking docs/41 level C (initial scalar 4F
model). Stage 9B can then convert command-domain cycles to a calibrated Fourier-plane mapping and
revisit C readiness.

With only the installed downstream camera, those direct coordinate conclusions remain blocked. The
downstream Stage 9A.1B session can select a practical operating point and record sensitivity to
carrier/stop settings, but it does not unblock physical 4F coordinate readiness.

## What cannot be concluded from this session alone

It does **not** validate physical 4F propagation, estimate aberrations, build a correction map, or
predict any fused-silica/material outcome. It records command-domain masks and a measured
order-position dataset only when direct Fourier-plane or conjugate-plane access exists. Everything
physical-unit remains blocked until the mapping is calibrated. Direct Fourier-plane mapping requires
a temporary diagnostic method at or conjugate to the Fourier plane. The installed downstream camera
supports empirical carrier-and-stop response characterisation only.
