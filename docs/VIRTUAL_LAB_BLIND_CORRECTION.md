# Virtual Lab / Blind Correction

## Purpose

The unified lab GUI now has one scientific pipeline with three explicit data-source modes:

- **LIVE LAB** — physical Beamage frames and physical SLM operation after explicit connection.
- **RECORDED LAB** — replay/re-analysis of previously measured numerical camera data; SLM commands are read-only because a historical image cannot respond to a new cast.
- **VIRTUAL LAB** — the repository digital twin impersonates the camera and stage; SLM casts are dry-run commands into the model only.

All three routes meet at the existing `CameraFrame` boundary. Display, metrics, formal storage and downstream analysis therefore operate on the same numerical frame object instead of maintaining separate “real” and “simulation” analysis code.

Simulated frames are always labelled with `data_origin=simulated`, `operating_mode=VIRTUAL LAB`, scenario identity/seed/truth hash, model geometry status, z position and cast-phase hashes. Recorded physical data are labelled `recorded_measured`; live Beamage data are labelled `live_measured` by the mode-aware controller. A simulated frame must never be presented as experimental evidence.

## What was reused

Virtual Lab deliberately reuses existing repository physics rather than adding a toy FFT demo:

- `vbb_study.digital_twin.vortex_beam_slm_errors.gaussian_input_field`
- `vbb_study.digital_twin.vortex_system_route.physical_axicon_on_own_plane`
- `vbb_study.digital_twin.vortex_wavefront_errors.unit_rms_zernike`
- `vbb_study.equations.propagation.angular_spectrum_propagate_bl`
- the existing unified `PhaseService`, which remains the authority for complete SLM phase composition
- the existing `MetricEngine`
- the existing `SensorlessOptimiser` trial/recommend/verify/rollback logic

The virtual SLM provider obtains the same complete configured phase as the real GUI. Because the physical pinhole/relay geometry is not yet fully bound, its selected-order surrogate removes only the locked blaze by asking the **same phase engine** to regenerate the phase with blaze disabled. It does not manually rebuild vortex/Zernike/retrieved-correction terms.

## Current geometry evidence and claim boundary

Known/reported bench information is kept separate from modelling assumptions:

| Quantity | Current value | Status |
|---|---:|---|
| Relay lens 1 focal length | 300 mm | reported |
| Relay lens 2 focal length | 300 mm | reported |
| Lens-centre separation | **300 mm** | confirmed by operator |
| Pinhole function | select +1 order | reported |
| Pinhole axial position | unknown | calibration required |
| Physical axicon | “Thorlabs 20° axicon” | reported label; exact part/angle convention unbound |
| Physical camera z=0 | unknown | calibration required |
| SLM1→SLM2 axial separation | 0.040 mm default | **existing repository placeholder, not measured geometry** |
| Axicon model base angle | 2° default | **existing repository placeholder, not asserted to equal the reported 20° optic** |

A pair of 300 mm focal-length lenses with **300 mm lens-to-lens separation is not silently converted into the repository's old ideal symmetric F300 relay with 600 mm lens separation**. Until the missing positions are measured, Virtual Lab uses `ideal_selected_order_surrogate` and labels every frame `PARTIALLY_BOUND_NOT_BENCH_CALIBRATED`.

This keeps the offline correction-control experiment useful without converting unknown geometry into false bench truth. Once the SLM separation, pinhole position, axicon part/angle convention and camera reference are measured, bind those values in `VirtualLabGeometry` and replace the selected-order surrogate with the corresponding fully propagated route.

## Virtual optical path

The present mock path is:

```text
hidden beam error
  -> Gaussian input
  -> hidden low-order wavefront error
  -> virtual cast SLM1 phase
  -> SLM1->SLM2 propagation (currently placeholder distance, visibly labelled)
  -> virtual cast SLM2 phase
  -> selected +1-order surrogate (relay geometry incomplete)
  -> physical refractive-axicon model
  -> free-space propagation to requested virtual z
  -> intensity-only virtual detector
  -> CameraFrame
```

The optimiser receives intensity frames, public z positions and the SLM commands it issued. It is not given hidden Zernike coefficients, beam offset/pointing or axicon decentre. The private truth can be exposed only by an explicit **Reveal truth** action after/around a validation run.

## Virtual scenarios

Seeded scenarios are deterministic and include `NOMINAL`, `LOW_ORDER_WAVEFRONT`, `ALIGNMENT`, `MIXED` and `STRESS_TEST`. They use only currently implemented model families: low-order declared wavefront terms, Gaussian beam decentre/pointing and physical-axicon decentre. Same scenario class + same seed gives the same hidden-truth hash.

Do not interpret successful intensity compensation as unique recovery of the physical source of an error. Different control combinations can generate similar observed intensities. Virtual Lab measures **compensation performance**, not guaranteed parameter identification.

## Mock z acquisition

The Virtual Lab page accepts a comma-separated z plan. The shipped default is `31, 34, 37, 40, 43, 46` mm because it is a useful offline propagation span; it is not declared to be the physical camera coordinate system. Virtual z is measured from the **post-axicon model plane** until the real camera reference is calibrated.

`Capture Z stack` steps the `VirtualStageProvider`, acquires each `VirtualCameraProvider` frame, analyses it with the existing `MetricEngine`, and calculates a multi-plane objective. The camera view updates as the worker runs. Heavy propagation runs off the Qt main thread.

The objective is deliberately multi-plane. For vortex states it combines normalized centroid walk, ring-radius variation, ring eccentricity, azimuthal non-uniformity, dark-core fraction and edge/saturation penalties. For non-vortex states it uses centroid walk, width variation, ellipticity error and edge/saturation penalties. The exact breakdown is shown in the returned result so the controller cannot “win” by making one attractive plane while damaging the rest of the propagation region.

## Blind SLM correction

`Start blind SLM correction` uses the existing `SensorlessOptimiser`. The first implementation is intentionally interpretable sequential modal probing over:

`defocus`, `astig_x`, `astig_xy`, `coma_x`, `coma_y`, `spherical`.

For each mode it measures start control, symmetric candidate perturbations and end control, then performs a fresh verification before accepting a command. Rejected changes are rolled back and the virtual SLM is recast so simulated hardware matches accepted state. Probe amplitude halves between passes.

Choose `SLM1`, `SLM2` or `BOTH` to compare correction space. The optimiser does not call `reveal_truth()`.

During optimisation casts use the same configured/cast hash semantics as the GUI but skip writing hundreds of redundant full-size PNG cast bundles. Normal operator casts remain provenance-persisted exactly as before.

## Alignment Assist

Alignment is a second explicit phase, never mixed into the initial SLM-only optimisation. The first implementation searches virtual physical-axicon x/y decentre commands and scores them with the same multi-z intensity objective. This lets the user compare:

`SLM-only optimum` versus `SLM correction + limited alignment assist`.

It is deliberately small and interpretable. It is not a generic magic alignment parameter and it is not enabled automatically.

## Camera modes and safety

Switching operating mode is a provider transition, not just a visual label.

Entering **VIRTUAL LAB** closes the previous camera/SLM provider, invalidates old cast receipts, installs `VirtualCameraProvider`, `VirtualStageProvider` and `VirtualSlmProvider`, and requires a new explicit virtual cast. The virtual SLM class contains no HEDS backend and therefore cannot send a physical SLM command.

Entering **RECORDED LAB** installs a read-only SLM provider. Attempting cast/blank raises a provider error. Choose replay files through the existing camera chooser.

Entering **LIVE LAB** invalidates virtual cast receipts and installs a disconnected real Beamage provider. It does **not** reconnect or cast HEDS. The user must explicitly connect physical devices. This prevents a simulated correction from being pushed to hardware merely by changing mode.

## Numerical quality

`preview` and `validation` use different transverse grids. Preview is intended for interactive algorithm development; Validation is the denser numerical setting and should be used before interpreting fine structure. The virtual frame metadata warns if the refractive-axicon radial period is sampled too coarsely.

The virtual detector has two options: clean deterministic intensity and an optional deterministic noise mode with background, shot-like noise and read noise. Exposure is a relative virtual scaling only; it is **not** a radiometric calibration of the Gentec camera. Display colormap/gamma remain display-only.

## Launch

From the repository root after installing the lab requirements:

```bash
python -m pip install -r requirements-lab.txt
python lab_gui/run_lab_control.py
```

Then select **VIRTUAL LAB** from `EXPERIMENT SOURCE`, configure/generate a scenario, set the desired SLM state on Home or the detailed SLM tabs, and use the Virtual Lab page to capture a z stack or start blind correction.

## Two-minute mock run

1. Open the unified GUI and choose **VIRTUAL LAB**.
2. Set the beam you want on SLM1/SLM2 (for example a vortex charge) and press the normal **Cast** buttons, or let the Virtual Lab task make its explicit dry-run cast.
3. Open **Virtual Lab**, choose `MIXED`, seed `1048`, and click **Generate hidden scenario**.
4. Leave `31, 34, 37, 40, 43, 46` mm as the first mock scan and click **Capture Z stack**.
5. Inspect the objective breakdown and beam walk/ring metrics.
6. Choose SLM1/SLM2/BOTH, leave a ~0.15-wave initial probe, and click **Start blind SLM correction**.
7. After the SLM-only result, optionally click **Alignment Assist**.
8. Click **Reveal truth** only when you want to compare the hidden simulated error with the compensation that was discovered.

## What remains physically unverified

Virtual Lab is currently an algorithm-development/control-space environment, not a claim of exact present-bench prediction. The most important outstanding measurements are SLM1-to-SLM2 axial separation, the complete lens/SLM/pinhole axial prescription, exact axicon part number and angle convention, camera z reference/pixel calibration and any phase/LUT calibration not already measured. Those are intentionally visible rather than guessed.

No Shack-Hartmann support, neural correction model or nonlinear material-interaction model is included in this change.
