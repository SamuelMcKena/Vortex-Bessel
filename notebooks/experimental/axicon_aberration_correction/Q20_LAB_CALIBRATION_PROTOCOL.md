# q=20 aberration-correction bench calibration protocol

This is the minimum experimental calibration needed before the current retrieval
is allowed to create a low-gain SLM2 correction trial. The code intentionally
blocks hardware output when any required quantity is unknown.

Do **not** fill missing values by copying nominal drawing values or by fitting the
same aberrated q=20 stack that will later be corrected.

## 1. Preserve the existing baseline stack

Keep the original 18-plane × 4-repeat q=20 BMG acquisition unchanged. The
before/after acceptance code fingerprints every raw BMG byte with SHA-256, so the
post-correction dataset must genuinely be a new capture.

Record the stage coordinate corresponding to every `z0 ... z17` filename. The
integer in the filename is an acquisition index, not an absolute optical z
coordinate.

## 2. Absolute camera-z reference

Required calibration field:

```text
z_at_relative_zero_from_axicon_mm
```

Choose the same relative-z convention used by the analysis and measure the
physical axial distance from the defined axicon/input reference plane to the
camera sensor plane assigned `relative z = 0`.

Do not assume that the first saved image called `z0` is physically z=0. The
current repository metadata deliberately leaves the absolute z reference null.

The sign convention must give a positive axicon-to-camera distance for all 18
planes.

## 3. Nominal transverse spatial frequency

Required field:

```text
k_perp_nominal_m_inv
```

This is the intended/reference Bessel transverse wavenumber against which the
retrieved local `k_perp_opt(z)` is compared. It must be defined independently of
the aberrated stack being corrected.

Preferred routes are:

1. derive it from independently verified axicon/cone geometry and refractive
   index at 1030 nm; or
2. measure it from a known-good/aligned reference configuration whose ring scale
   is accepted as the target.

Do not simply use the median `k_perp` fitted from the distorted q=20 dataset as
the nominal value; that would partly define the target from the error itself.

## 4. Camera optical axis versus z-stage position

Preferred field:

```text
camera_optical_axis_yx_px_by_z
```

Acquire an aligned reference beam at the **same 18 camera/stage positions** used
for the Bessel z scan and save the raw sensor position `[y, x]` of the optical
axis at each position. This separates mechanical camera-stage runout from real
Bessel-beam walk/coma.

Use the same camera ROI, orientation and acquisition path. Do not recenter the
q=20 images themselves to manufacture a stationary axis.

A single value

```text
camera_optical_axis_yx_px
```

is accepted only if the optical axis has been measured and shown not to move with
stage position, in which case set

```text
camera_optical_axis_single_value_valid_for_all_z = true
```

Otherwise leave the single-value flag false.

## 5. Resolve the intensity-only conjugate branch

The inverse problem has a direct/conjugate ambiguity. The simplest Miao-style
check compares the retrieved annular input intensity with an independently
measured input-plane intensity and detects a 180-degree rotation.

For a nearly circular Gaussian input this check may be poorly conditioned because
a 180-degree rotation looks almost identical. In that case use a deliberately
asymmetric, known-sign calibration perturbation instead:

1. keep the optical system otherwise unchanged;
2. apply a small known asymmetric phase perturbation with known orientation/sign;
3. capture a second calibration z stack;
4. retrieve the change; and
5. select the direct/conjugate branch whose retrieved change matches the known
   perturbation after the independently calibrated coordinate transform.

Do not use the q=20 target vortex itself as the branch fiducial; intensity does
not reveal vortex sign.

## 6. Establish whether SLM2 is conjugate to the reconstructed input plane

Required field:

```text
slm2_is_conjugate_to_input_plane
```

This is not inferred from the phrase “4F system.” Verify the real bench.

A practical check is to introduce known localized/asymmetric SLM2 phase features
or fiducials and image/measure their mapped coordinates at the axicon/input
reference plane. Repeat at several positions across the illuminated SLM2 region.
A true conjugate mapping should be described by one consistent magnification,
rotation and parity across the field, within experimental uncertainty.

The repository's digital-twin route currently labels the detailed physical 4F
lens/filter coordinate model as unvalidated, so nominal 100-mm distances in code
must not be used as proof of conjugacy.

### If SLM2 is conjugate

Measure and fill:

```text
slm2_shape
input_plane_m_per_slm2_pixel
slm2_center_yx_px
slm2_rotation_deg
slm2_parity_x
slm2_parity_y
```

The end-to-end metres-per-SLM2-pixel value should include the real relay
magnification. Do not substitute bare SLM pixel pitch unless the measured relay
magnification is exactly unity.

### If SLM2 is not conjugate

Set:

```text
slm2_is_conjugate_to_input_plane = false
```

and stop before hardware application. A simple resized/rotated phase image is
not a valid correction. The next software step is a measured complex-field
propagator through the actual relay from the reconstructed plane back to SLM2.
The current code deliberately blocks instead of inventing that relay.

## 7. SLM2 phase response at 1030 nm

Required field:

```text
slm2_phase_lut_1030nm_calibrated = true
```

Only set this after the phase-versus-drive response used by the actual lab GUI
has been measured/validated at 1030 nm over the required phase stroke.

The correction controller outputs a **phase layer in radians in native SLM2
pixels**. It does not export a linearly scaled greyscale bitmap. The GUI/driver
must add the correction to the existing programmed SLM2 phase, wrap the combined
phase once, and then use the measured LUT.

## 8. Run the baseline full retrieval

Copy:

```text
q20_hardware_calibration_template.json
```

to:

```text
q20_hardware_calibration.json
```

and fill only values established above.

Then run:

```powershell
python run_q20_miao_retrieval.py
```

Inspect `outputs/miao_full_q20/correction_manifest.json`. A low-gain trial must
not be proposed unless `application_ready_for_low_gain_trial` is true.

Also generate baseline experimental metrics:

```powershell
python q20_experimental_acceptance_metrics.py
```

Keep the resulting baseline metrics CSV unchanged.

## 9. First correction trial

Use the v2 controller to create a conservative candidate. The normal starting
point is approximately 5% of the recovered residual correction, not a full 100%
application.

The candidate `.npy` file is an additive signed phase layer in radians and native
SLM2 coordinates. Apply it through the calibrated phase path in the GUI.

## 10. Capture a genuinely new post-mask stack

Without changing the camera calibration, target q, nominal `k_perp`, pixel pitch,
ROI or z positions, capture another complete 18×4 BMG stack.

Run the independent experimental metric generator on that new folder. The CSV
contains a SHA-256 fingerprint of the raw BMG dataset. The controller rejects
before/after files with the same fingerprint or mismatched target/calibration
provenance.

## 11. Experimental acceptance

The candidate is accepted only from the new camera data. Current gates require:

- median measured-vs-ideal correlation improvement >= 0.01;
- median measured-vs-ideal RMSE reduction > 0;
- median **azimuthal** principal-ring CV reduction >= 5%;
- maximum dark-core ratio worsening <= 0.01.

If any gate fails, the iteration is rejected. Do not accumulate a failed
correction into the accepted phase.

## Reference

B. Miao, L. Feder, J. E. Shrock, and H. M. Milchberg, “Phase front retrieval and
correction of Bessel beams,” *Optics Express* **30**(7), 11360–11371 (2022),
DOI: 10.1364/OE.454796.
