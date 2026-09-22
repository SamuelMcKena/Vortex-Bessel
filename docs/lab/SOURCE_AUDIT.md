# September 15 lab workflow provenance

## Sources inspected

- Current day-to-day repository: `SamuelMcKena/Vortex-Bessel`, main commit
  `5196e33322514c333e6bae5984db7618c9e52826`.
- `Publication_Study (2).zip`: older publication snapshot. The current repository
  README explicitly identifies Vortex-Bessel as its consolidated successor.
- September 14 `slm_lab_gui (2) 1.zip`: the user-uploaded GUI source.
- September 14 `slm_lab_gui_v0_5_heds_phase_safe.zip`: the subsequent phase-path
  update, used as the imported GUI baseline under `lab_gui/`. Only the GUI source,
  requirements and tests were migrated; old retrieved correction arrays, caches,
  output bundles and legacy duplicates were excluded.
- `HEDS(1).zip`: inspected Python wrapper, not redistributed. The relevant
  implementations are `SLM.setWavelength`, `getWavelength`, and `showPhaseData`
  in `holoeye_slmdisplaysdk_slm.py`. Phase data is explicitly distinct from the
  image-data API; phase_unit defaults to 2π. The compiled SDK calibration is
  outside this source, so the software audit cannot certify optical phase stroke.
- `Calibration.zip`: 52 BMG frames in 13 groups of four and 16 summary logs.
  All 52 were parsed using the constrained v1/12-bit/2048×2048 header and
  481-byte-prefix signed-int32 layout; the resulting inventory with hashes is
  `calibration_20260914_inventory.csv`.
- `current_calibration_inference.csv`: an earlier generated interpretation was
  inspected, but its suggested gain/coefficients were not silently installed as
  measured calibration. In particular, the two truncated correction prefixes
  are retained as unresolved gains without additional association evidence.

## Changes to the GUI baseline

The v0.6 integration adds a session dialog and strict complete-phase SLM2 casting,
with HEDS mode/serial checks, explicit display receipts and baseline restore.
The usual GUI controls, per-panel IDs, geometry and carrier remain inherited
from v0.5. The normal GUI still defaults to its legacy PNG transfer setting;
the workbench requires a successful **direct_phase_array** baseline first.
Failed regeneration now prevents a stale phase from being cast. Cast directory
names include microseconds to avoid same-second overwrite.

A session's native Zernike map is independent of the GUI's old unnormalised-wave
controls. The full base and correction are composed before display. The GUI's
retrieved-correction loader is not used to load a total session command, so it
cannot resize the command or add a second carrier. Baseline and cumulative trial
radians are logged separately. Raw camera data remain local to the session.

## Method and scope

This is a discrete, intensity-metric sensorless search of one commanded mode at
a time. Such searches use images acquired under known command perturbations;
see, for example, the primary research paper
[Teich et al., wavefront-sensorless AO with Zernike adjustment](https://arxiv.org/abs/1710.03565).
The metric here is designed for this ring-beam use case, not copied from that
paper's PIV sharpness metric. The fixed rotational-baseline target, power,
radial-profile and core-filling checks must be assessed on the actual beam.

The software tests validate arithmetic, data provenance and operational gates.
The synthetic camera harness is an explicit analytic test response, not a
propagated physical beam and not experimental evidence. No optical correction
performance is claimed before the user obtains a fresh validated measurement.
The legacy q20 inverse retrieval has not been altered or released from its
separate hardware-calibration gates.

## Archive SHA-256

| Archive | SHA-256 |
|---|---|
| Calibration.zip | `66c0dc21b6a0e8ebb7014ee1162083aeda76eed2037b936c514269c265ff89a7` |
| slm_lab_gui (2) 1.zip | `527c8bc22cb03ca4388b378901a08983f33dbc4b8ad8f1c056013b705baa4c99` |
| slm_lab_gui_v0_5_heds_phase_safe.zip | `8f5a451b25d1a78871caca1fb9f460d8f0dd6499d111b7150d1afe51221bc23f` |
| HEDS(1).zip | `032dacbd59b856e8140e8683fa0acddcee7d4775e66201e2e16d3288707e9f71` |

## Local validation record

- 30 focused tests passed: new lab workflow, imported GUI tests, and new trial
  casting/rollback tests.
- Python 3.12.14, NumPy 2.3.5, Pillow 12.3.0, PySide6 6.11.2, pytest 9.1.1.
- GUI instantiated and rendered with Qt's offscreen platform; session list and
  controls populated from a synthetic session.
- All 52 supplied BMG files passed the reader's format/length checks.
- Compile checks and whitespace checks passed.
- CI added for Python 3.12 and 3.13. Live HEDS/Windows/optical response validation
  remains a bench task; these local results do not substitute for it.
