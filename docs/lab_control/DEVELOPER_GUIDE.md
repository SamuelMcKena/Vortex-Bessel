# Developer guide

## Environment

Runtime dependencies remain intentionally small: PySide6, NumPy and Pillow.
Set `PYTHONPATH=lab_gui:.` when running modules from the repository checkout.

## Tests and smoke checks

```bash
PYTHONPATH=lab_gui:. QT_QPA_PLATFORM=offscreen \
  python -m pytest tests/test_lab_workflow.py lab_gui/tests -q

python -m compileall -q vbb_study/lab lab_gui tools/analyze_beam_walk.py tools/test_beamage_pipe.py

PYTHONPATH=lab_gui:. python lab_gui/run_labcontrol_cli.py \
  run recover-q20 --camera replay --output /tmp/labcontrol-demo
```

CI runs Python 3.12 and 3.13, headless Qt, the whole lab test set, compile checks
and the replay/demo smoke path. Hardware is never required in CI.

## Adding a phase term

1. Add typed config fields and a disabled-by-default switch.
2. Implement the radians contribution in `slm_lab_control.phase`.
3. Add it once to the unwrapped component sum and preserve one final wrap.
4. Expose it in the compact editor; do not introduce a second composer in
   `labcontrol` or the advanced GUI.
5. Add exact equivalence/additivity tests and preset migration if necessary.

## Adding a camera

Implement `CameraProvider`. Return only a finite real 2-D numeric matrix with
honest `data_kind`, settings and source metadata. Do not return a rendered screen
capture. Make start/stop/disconnect recoverable and test the provider through
`LabController` and `FormalCaptureService`.

## Validating Beamage

`BeamagePipeClient` is the Windows-only adapter built from the supplied official
Unicode C++ example. Keep vendor command names isolated in that module. Run
`tools/test_beamage_pipe.py` with PC-Beamage Pipeline enabled, then validate the
BMP against a numeric export before changing its live-preview-only status.

## Adding a stage

Implement `StageProvider.request_move` and `confirm_position`. Recipes should
consume the motion result, not inspect a specific motor SDK. Record actual z and
reference in `ExperimentState.camera` before capture.

## Adding a metric

Metrics accept `CameraFrame` plus `ExperimentState`, never a pixmap. State-aware
selection should be based on physical configuration, not a GUI tab. Return values
with explicit units/coordinates and keep acceptance thresholds in recipes.

## Adding a recipe

Create a serialisable `RecipeDefinition`, declare calibration prerequisites and
use existing step kinds. Execution handlers must call controller/capture/metric/
sensorless services. Save a checkpoint after every capture or decision.

## State/event discipline

- Read with `store.snapshot()`.
- Write with `store.update()` and meaningful source/reason.
- Never retain and mutate a snapshot as shared state.
- Never have one widget reach into an unrelated widget.
- Avoid adding mutable globals.

## Hardware implementation labels

Use only:

- `IMPLEMENTED`: code path exists;
- `SOFTWARE-TESTED`: automated behavior tests pass;
- `REPLAY-VALIDATED`: archived/numeric sources execute the intended path;
- `HARDWARE-UNVERIFIED`: adapter exists but has not passed the bench procedure;
- `PHYSICALLY VALIDATED`: evidence from the actual hardware/configuration exists.

One component may carry more than one carefully scoped label. Never infer
physical validation from a mock SDK, synthetic image or successful import.

## Pull-request review checklist

- one state store and one phase composer;
- no change to locked carrier/serial/wavelength without explicit hardware work;
- exact phase and cast hashes retained;
- display transforms separated from metrics;
- formal capture refuses stale/uncast phases;
- provider errors fail closed;
- calibration invalidation is dependency-specific;
- sensorless decisions require fresh verification and retain rollback;
- synthetic/replay/experimental labels remain visible;
- GUI worker shuts down cleanly;
- docs and tests describe what is actually implemented.
