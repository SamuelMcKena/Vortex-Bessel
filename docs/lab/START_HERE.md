# Tomorrow's beam correction session

Use this branch of **Vortex-Bessel**. It includes your September 14 v0.5 SLM GUI,
now extended to v0.6 with a **Lab measurements** panel. Your simulation and q20
inverse-retrieval code remain available in their established locations.

## Launch on the lab PC

Use the Python environment that already runs your working HEDS GUI:

```powershell
cd C:\PhD\Code\Vortex-Bessel
git fetch origin
git switch lab/measurement-correction-workflow
python -m pip install -r requirements-lab.txt
python run_lab_gui.py
```

If cloning into a new folder instead:

```powershell
git clone --branch lab/measurement-correction-workflow https://github.com/SamuelMcKena/Vortex-Bessel.git
cd Vortex-Bessel
python -m pip install -r requirements-lab.txt
python run_lab_gui.py
```

If the new folder cannot find your existing HEDS wrapper, pass its **parent**
directory (replace the example with the real location):

```powershell
python run_lab_gui.py --sdk-path "C:\path\containing\HEDS"
```

The vendor SDK and `hedslib` still need to be installed in that Python environment.
The uploaded HEDS ZIP contains Python wrappers, not the whole installed SDK.
Vendor code/binaries and your 0.85 GB extracted measurement set are not bundled in
this public code branch. No laser, camera or stage is automatically actuated.

The lab commands run directly from the checkout on Python 3.12/3.13 without
installing the full simulation package. The simulation's pinned environment and
Python 3.13 requirement have not been changed. Prefer your already-working lab
environment; avoid reinstalling it solely to run the numerical report suite.

## Practical sequence in the GUI

1. **Load your working preset.** Keep the existing SLM1/SLM2 beam arrangement.
   Start with the actual uncorrected baseline you intend to compare against, or
   an already experimentally accepted correction. Do not enable one of the old
   `UNCALIBRATED` retrieved maps merely because it was bundled with v0.5.
2. Select **direct_phase_array**, connect HEDS, and cast the baseline to both
   SLMs. This requests 1030 nm phase mode and sends radians through `showPhaseData`.
   Moving from the legacy PNG route may change the beam; establish a fresh
   baseline in this mode. Do not compare the new mode against yesterday's
   differently encoded command as if only a Zernike coefficient changed.
3. Click **Lab measurements → New session**. Choose a new folder, for example
   `lab_sessions\2026-09-15-q20`. The current GUI phase and both panel states are
   saved. The profile editor opens; fill the measured values below.
4. With the same exposure/gain/background settings, record **at least 3 dark BMGs**
   with the beam blocked. Click **Start with dark frames** and select them. Keep
   any Beamage background correction fixed; this workflow subtracts the mean of
   those exported darks from the exported measurements. The GUI verifies that
   the displayed baseline still matches the phase snapshot.
5. Plan **astig_x** first. The table lists a starting control, four shuffled
   signed trials, then an ending control. Each trial is the complete frozen
   baseline plus its explicitly recorded correction. Start with the small
   ±0.15/±0.30 rad RMS trials; the original GUI's waves fields are a different
   convention and should not be edited to these numerical values.
6. Work down the table in order. Select a trial → **Cast selected to SLM2** →
   **Open capture folder**. Wait for the displayed beam to settle, then manually
   capture each prescribed z position and repeat in Beamage. Save the BMGs there
   using `CAPTURE_NAMES.md`, then click **Import selected capture**. The session
   panel leaves SLM1's phase unchanged and checks its last logged command.
7. **Evaluate sweep → Open report.** If a candidate passes the shape, radial,
   core, power and drift checks, click **Plan fresh verification**. Capture the
   three new groups in order: current control → candidate → current control,
   at the same z positions. These must be new captures, not copies of the sweep.
8. Click **Accept if verified**. On success, click **Restore accepted / round
   control** to put the accepted correction back on SLM2; the last verification
   capture had intentionally restored the control. Then plan the next mode.
   Coma Y and coma X are useful next diagnostics from the earlier coarse scan,
   but the current workflow measures both signs rather than assuming an optimum.
   Leave defocus/spherical until the basic asymmetric modes have been assessed.
9. If a trial worsens the beam, use **Restore accepted / round control**; to undo
   the entire session, use **Restore original baseline**. A failed verification
   never changes the accepted phase. Keep all raw files and the whole session
   directory together.

Do not close the lab dialog to edit/cast unrelated GUI controls midway through a
sweep. If you do change the setup, restore the baseline or start a new session.
Acceptance records do not automatically rewrite your ordinary GUI preset: use
**Restore accepted** when resuming the session. The accepted trial's
`total_phase_rad.npy` is the complete command; `correction_phase_rad.npy` is only
the additive correction relative to the original baseline. Never add the total
command to the usual phase terms a second time.

## Values to enter once

| Profile field | What to enter |
|---|---|
| `slm.center_yx_px` | Measured illuminated footprint centre, **[row y, column x]**, in native SLM2 pixels. The GUI centre is copied as a starting value. |
| `slm.illuminated_radius_px` | Measured correction-disk radius on SLM2. An approximately 2 mm radius would be 250 pixels at 8 µm/pixel, **only if that is the actual measured footprint**. The disk must fit inside the panel. |
| `camera.exposure_us`, `gain` | The real fixed acquisition settings. Gain zero is valid. |
| `camera.settings_id` | A short label covering exposure, gain, attenuation and background/export options. These settings are operator-attested; the code does not control Beamage. |
| `camera.linear_raw_export_verified` | Set true after confirming numeric BMG/raw linear export, rather than auto-scaled screenshots. BMG can contain camera background processing. |
| `planes[].z_mm` | Actual recorded stage positions relative to your stated reference. `z000` is a label, **not zero optical distance**. |
| `planes[].axis_yx_px` | A fixed reference centre in raw camera pixels for each plane. Prefer your aligned reference-axis measurement. Do not recenter each trial on its own brightest spot. |
| `z_reference` | What the stage zero is, and which travel direction is positive. |
| `roi_radius_px` | Fixed disk containing the principal ring and relevant nearby rings, wholly inside the camera image. Inspect the report crop. |
| `repeats` | 3 by default; set 4 to match yesterday's repeats if preferred. |

The copied panel identity/geometry is SLM2 **6010-2381**, 1920×1080, 8 µm pitch;
SLM1 is **6010-2382**. Both keep the original locked 20-pixel, negative-y carrier.
The camera template is Beamage-4M, 2048×2048, nominal 12-bit maximum 4095, grounded
in the supplied files. Verify that tomorrow's export settings match.

Use at least **3 distinct z planes** spanning the region you need corrected.
A three-plane, three-repeat sweep takes 54 frames; fresh verification adds 27.
This validates only those planes, not an unmeasured continuous propagation range.
You can configure one plane for rapid diagnostic screening, but it cannot be
accepted as a multi-plane correction. Capture a denser final z-stack after the
first accepted improvements for the existing propagation/retrieval analysis.

`phase_owner`, `phase_path_evidence` and the HEDS readback are populated at Start
by the connected GUI. `phase_response_optically_verified` remains **false**:
reading back 1030 nm establishes the requested SDK mode, not an interferometric
measurement of the panel LUT. Sensorless trials use measured response to nominal
commands. The older inverse solver's optical-calibration requirements remain.

## Exact capture names

Each trial has its own folder; the beam profiler's automatic `_1` suffix no
longer needs to encode the trial coefficient:

```text
incoming/round01_trial01/
    z000_r01.bmg
    z000_r02.bmg
    z000_r03.bmg
    z001_r01.bmg
    ...
    z002_r03.bmg
```

Rename the resulting files to these exact names if Beamage appends its own
counter. Keep summary TXT logs outside this numeric capture folder. **Your
September 14 `base.txt` and extensionless files are summary tables, not pixel
matrices.** The BMG files carry the actual images. Other supported imports are
numeric NPY/TXT/CSV matrices and unscaled grayscale BMP/PNG/TIFF; colour or
palette screenshots are rejected. Do not save both BMG and TXT for the same
repeat in the import folder.

Imports are copied into immutable trial folders with phase, raw-file and pixel
hashes. Repeated pixels in a differently encoded file are still rejected.
Saturation/shape/size checks happen before the capture is sealed. File timestamps
are ingest times, not camera-trigger telemetry: ingest immediately in capture
order. Never leave a folder full of old captures and recast a new trial over it.

## What the score means

The reference is the **rotational average of the first measured baseline**,
held fixed for the whole session at each measured axis. It is labelled as a
symmetry reference, not an ideal beam. The score is relative L2 error between
unit-power ROI images. Lower is better. Additional gates limit radial-profile
L1 change, central-core filling, power variation and bracketing-control drift.
A two-SEM repeat-noise margin and a 2% relative improvement floor screen noisy
changes; this is an operational heuristic, not a formal confidence interval.

This is appropriate for reducing asymmetric intensity distortion while retaining
your baseline ring family. It will not repair an incorrect cone angle by itself,
prove an optical phase map, certify vortex sign/charge from intensity, or correct
amplitude errors perfectly with a phase-only SLM. Small discrete coordinate
searches can have local optima; no unmeasured parabola-fit optimum is applied.
High-order correction and the old retrieved maps remain separate experiments.

For the same pupil and axes, the new convention relates to the original GUI by
`GUI waves = trial_rad_RMS × sqrt((n+1) × (1 if m=0 else 2)) / (2π)`.
The native correction map uses its measured SLM footprint, which may differ from
the GUI's 9 mm pupil. **Do not use the numerical conversion across different
pupil sizes/centres/rotations.** The GUI trial panel handles the phase map directly.

## Offline practice and command line

No HEDS connection is needed for a fully labelled synthetic practice run:

```powershell
python run_lab_session.py demo lab_sessions\practice
```

Open `lab_sessions\practice\report\index.html` to inspect the generated report.
This demonstration checks software behaviour, not propagation physics or lab
correction performance. A demo never reports experimental acceptance.

The CLI exposes the same session engine:

```powershell
python run_lab_session.py --help
python run_lab_session.py status lab_sessions\2026-09-15-q20
python run_lab_session.py report lab_sessions\2026-09-15-q20
```

For a CLI-only session: `init`, edit `profile.json`, `start --base <radians.npy>
--dark <three files>`, `plan --mode astig_x`, `capture <trial> --settings-id <id>`,
`evaluate`, `verify-plan`, capture its three groups, then `accept`. Unlike the
integrated GUI, CLI capture has only the operator's attestation of which phase
was displayed. No CLI command sends a phase to the SLM.

Inventory yesterday's archive reproducibly:

```powershell
python run_lab_session.py audit-calibration "D:\measurements\Calibration" --output lab_sessions\historical_inventory
```

## Validation and limits

Focused checks:

```powershell
$env:PYTHONPATH = "lab_gui;."
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest tests/test_lab_workflow.py lab_gui/tests -q
Remove-Item Env:QT_QPA_PLATFORM
```

The code and GUI were exercised locally with simulated/mocked hardware and the
52 real BMG files were parsed. A live Windows HEDS/camera session has **not** been
run here. Test the first baseline cast and restore at the bench before committing
to a whole sweep. The HEDS SDK still owns its wavelength-specific conversion;
this GUI path adds no external LUT.
