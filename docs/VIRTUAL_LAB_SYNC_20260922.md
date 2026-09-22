# Personal-device Virtual Lab sync — 22 September 2026

Target GitHub branch: `feature/virtual-lab-blind-correction`.

## Provenance and scope

The starting Virtual Lab commit is `e997d7adee6abe436b5b475a56409a0b00239c9f`.
The current lab-workflow commit `4478665e39f9211ce54809245ea87732ed199b07`
(including its sequential-SLM vector Bessel model) was merged before importing
the personal-device `VirtualLab_20260916/FULL_SYSTEM` source.

The imported source includes subsequent local work by both assistants, not just
the original ZIP: scrollable GUI layout, camera lifecycle/backpressure and
thread dispatch, mask editing, fault/capture/correction workflows, axicon
multirate propagation, fixed relative camera sensitivity, fixed input power,
4F aperture controls, the small-beam designer and sample-plane controls.
Existing repository-only files were retained. The running personal installation
was not modified by the sync.

Use `python lab_gui/run_lab_control.py` from the repository root after installing
`requirements-lab.txt`. Real devices still require their vendor software;
Virtual Lab mode does not require or command them.

## Figures and local-only material

- `outputs/validation/axicon_radius_response_20260917/`: original raw-field radius
  diagnostic, including CSV/JSON. This predates fixed-power camera changes.
- `outputs/validation/fixed_camera_20260917/`: subsequent fixed-sensitivity
  comparison, JSON and ideal-reference figure. The old offscreen GUI screenshot
  contains font-rendering artifacts; it is historical evidence, not a current
  usability screenshot.
- `compare_real_beamage_vs_model_q20.png`: pre-existing model/measurement comparison.
- `figures/virtual_lab/historical_cast_previews/`: all 36 local PNG cast previews
  present at sync time. These are archived phase pictures, not approved masks to
  apply to hardware; their raw arrays and device session state are not bundled.

Excluded: raw `.npy` camera/cast captures, active sessions, personal presets,
crash logs, caches, temporary test directories and installed driver binaries.
No source, numerical analysis or older repository figure was deleted.

## Scientific boundaries

Fixed virtual sensitivity is relative, not calibrated radiometry. Old captures
with per-frame peak normalisation must not be numerically compared to new ones.
Native model previews are not additional camera resolution. The objective/sample
mapping remains an idealised scalar model; high-NA/vector effects and measured
bench calibration are not established by this sync or by GUI tests.

## Validation

Local validation on Windows / Python 3.13: **143 tests passed in 139.74 s**
(22 September 2026), covering the complete `lab_gui/tests` suite plus the
newly merged sequential-SLM vector-model tests. `git diff --check` also passed.
This is software regression evidence, not physical bench validation.

Run the imported GUI suite together with the newly merged vector-model tests:

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pip install pytest
python -m pytest lab_gui/tests tests/test_fu_oscillating_vector_bessel.py -q
```

The GitHub Virtual Lab workflow runs the same suite and publishes a
`virtual-lab-full-system` source artifact only after the tests pass.
