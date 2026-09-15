# Repository and v0.6 audit

## Baseline selection

The relevant branches were inspected before implementation:

| Branch | Relevant head | Finding |
|---|---|---|
| `lab/measurement-correction-workflow` | `fda3ef9` | Phase-safe v0.6 GUI, HEDS workbench, strict BMG session workflow and headless CI |
| `lab/commissioning-and-rapid-recovery` | `4376e93` | Newest superset; adds the three-plane recovery protocol, reusable beam-walk tool and packaged v0.6 GUI |
| q20/retrieval branches | inspected through repository branch/file history | Kept separate where calibration gates and retrieval ownership differ |
| `main` | `5196e33` at audit time | Consolidated research code, but older than the focused lab-control branch for these files |

The new work therefore branches from `lab/commissioning-and-rapid-recovery` and
targets it for review. The original v0.6 implementation remains in history and
its phase/backend modules remain the numerical authority.

The attached `Publication_Study (2).zip` is a publication-study snapshot, not a
second v0.6 GUI distribution. It was inventoried but not substituted for the
newer repository lab GUI. The existing `docs/lab/SOURCE_AUDIT.md` records the
earlier v0.5/v0.6 GUI, HEDS wrapper and 52-file BMG provenance audit.

## Reuse decisions

- Reused `slm_lab_control.phase` as the only complete-phase engine.
- Reused `slm_lab_control.hardware.backends` as the HEDS implementation.
- Reused locked `hardware_profiles.py` and existing preset migration.
- Reused `vbb_study.lab.beamage` and `core.load_array` for quantitative ingestion.
- Promoted gradient-symmetry centring and beam-walk fitting from
  `tools/analyze_beam_walk.py` into reusable core modules; the tool now calls them.
- Preserved the strict existing manual measurement workbench for backwards
  compatibility while adding a general formal-capture service.
- Generalised signed candidates, control bracketing, verification and rollback
  instead of embedding the prior single workflow in Qt.

## Problems corrected

- The short-lived circular-pupil switch replaced the entire phase outside a
  circle. It is now an inert legacy preset key; pupil diameter only normalises
  and bounds pupil-local additive terms.
- First-order steering now has its own physical mrad term. Carrier and coma are
  not repurposed as steering.
- GUI-owned state was replaced by typed snapshots and structured events.
- Live preview and formal evidence now have separate lifecycles.

## Regression constraints retained

- 1030 nm HEDS wavelength set and readback;
- radians and explicit 2π direct-phase semantics;
- exactly one final wrap;
- central hardware geometry and serial identities;
- locked order-selection carrier;
- no second carrier in a retrieved correction;
- no second addition of a complete session phase.
