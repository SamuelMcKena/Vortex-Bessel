# Migration from v0.6

## What remains familiar

Launch the compact front end for the existing dual-SLM workflow. It retains:

- separate SLM1/SLM2 editors and previews;
- vortex, digital axicon, focus, wavefront and custom phase;
- low-order Zernikes and retrieved correction;
- locked geometry, serial, wavelength and carrier profiles;
- old JSON presets;
- dummy/HEDS backend and all transfer modes;
- cast, blank and the existing manual correction workbench.

The old `run_lab_gui.py` entry point remains present. The clearer new entry is:

```bash
python lab_gui/run_slm_gui.py
```

## State ownership change

The compact GUI now writes `ExperimentStore` through `LabController`. It still
presents the familiar controls, but its widgets are no longer the authoritative
state. Opening it from the advanced GUI shares the same controller, HEDS handles,
phase results and state events.

## Circular pupil correction

The legacy `circular_pupil` preset key is accepted only for compatibility and is
forced false. `pupil_diameter_mm` is now clearly a **reference diameter for local
terms**, never an aperture or whole-panel gate. Changing it cannot delete the
carrier, wavefront, vortex, axicon, focus, retrieved correction or custom phase
outside the circle.

Old preset files are not modified in place. Saving after loading writes the safe
inert value.

## New steering term

`steering_x_mrad` and `steering_y_mrad` are additive first-order phase gradients.
They are disabled by default in existing presets. The fixed carrier remains
locked and unchanged.

## Output concepts

Normal compact-GUI cast bundles remain available. Advanced formal sessions add
immutable state snapshots, phase hashes, raw frames and metrics. A preset or
preview does not become a measurement simply because it was displayed.
