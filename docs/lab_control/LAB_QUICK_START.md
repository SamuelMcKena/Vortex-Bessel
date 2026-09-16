# Lab-control quick start

## Install

From the repository root in the Python environment that can import the HEDS SDK:

```powershell
python -m pip install -r requirements-lab.txt
```

## Compact dual-SLM GUI

```powershell
python lab_gui\run_slm_gui.py
```

Use this for straightforward phase configuration, presets, cast and blank. The
reference pupil diameter does not crop or blank the panel.

## Unified lab GUI

```powershell
python lab_gui\run_lab_control.py
```

It opens on **Home**: SLM1 and SLM2 quick cards are stacked on the left and the
camera is the dominant view on the right. Each card exposes connection, complete
phase preview, vortex charge, retrieved correction, active layers, cast-state
warning, cast/blank, presets and a Details route. The SLM1 and SLM2 pages contain
the complete phase editors; they edit the same state and use the same hardware
controller as Home and the compact GUI.

Start in dummy mode to inspect the live quantitative path. Use Measure for
camera controls and formal capture, Optimise / recover for recipes, Sessions for
raw-frame comparison, Calibration for dependency readiness, and System for
hardware routes and physical context.

Numeric fields ignore the mouse wheel so scrolling cannot silently change a lab
parameter. Arrow buttons, keyboard stepping and typed entry continue to work.
Unrelated camera/state refreshes do not replace text while it is being edited.

## Headless status and demo

```powershell
python lab_gui\run_labcontrol_cli.py status
python lab_gui\run_labcontrol_cli.py run recover-q20 --camera replay --output outputs\q20_recovery_demo
```

The built-in recovery command generates labelled synthetic matrices and replays
them through the real metric/recipe/rollback paths. It is a software exercise,
not optical validation.

## Tests

```powershell
set PYTHONPATH=lab_gui;.
set QT_QPA_PLATFORM=offscreen
python -m pytest tests/test_lab_workflow.py lab_gui/tests -q
python -m compileall -q vbb_study/lab lab_gui tools/analyze_beam_walk.py
```

On PowerShell use `$env:PYTHONPATH='lab_gui;.'` and
`$env:QT_QPA_PLATFORM='offscreen'`.

## First real-lab validation

1. Begin with no sample processing and keep an immediately available optical
   stop/laser-safe procedure.
2. Load the known-good preset in the compact GUI. Confirm locked panel serials,
   1030 nm and carrier summary.
3. Connect HEDS and select `direct_phase_array`. Confirm both wavelength readbacks
   and phase-mode indicators.
4. Generate/cast one known mask per panel. Compare complete phase hashes and the
   displayed physical result with the known-good v0.6 behavior.
5. Change only the reference pupil diameter. Verify the carrier and full-panel
   content stay visible; local Zernike support alone should change when enabled.
6. Test small dedicated ±x/±y steering commands and record the measured sign/scale.
7. Run the advanced GUI in replay mode with archived BMGs. Check orientation,
   dimensions, centre/ring overlays and stored hashes.
8. Start PC-Beamage with Pipeline enabled and select `beamage`. The official
   pipe route supplies vendor measurements and a BMP live preview. Validate that
   route against a simultaneous numeric export before allowing formal use; the
   GUI deliberately blocks the BMP from formal capture until then.
9. Capture a fixed-camera-settings q=0 three-plane stack, then q20 at identical
   z positions. Treat slopes as camera-relative until rail calibration is valid.
10. Run one deliberately conservative signed correction sweep. Confirm candidate
    ordering, start/end controls, fresh verification, acceptance and rollback.
11. Attach session/report evidence to calibrations and evaluate readiness. Do not
    promote demo or replay readiness to current-bench readiness.

Stop and return to the known-good v0.6 branch if any HEDS wavelength, phase unit,
orientation, carrier, or cast-hash check differs unexpectedly.
