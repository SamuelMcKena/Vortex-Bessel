# Refactor Plan: Publication_Study → Structured-Beam Simulation Atlas

**Status:** Phase 1 and Phase 2 in progress (tree inventory and safe moves).
**Written:** 2026-06-03
**Author:** Refactor assistant pass — review and amend as work proceeds.

---

## 1. Why This Refactor

The project started as a paper-output folder (`Publication_Study`) and grew into a
broad structured-beam simulation atlas covering scalar, vortex, vector, polygonal,
discrete, and material-proxy beam families.  The name `publication_*` is now
misleading: the study is not only a publication pipeline — it is also a reference
atlas, a hardware-route comparison, and a planning tool for future experiments.

Specific problems found during Phase 1 inventory (2026-06-03):

| Problem | Location | Severity |
|---------|----------|----------|
| All notebooks flat in `Publication_Study/` root | root `*.ipynb` | High — hard to navigate |
| `publication_*` prefix on notebooks blurs study vs. paper-export purpose | all notebooks | High |
| No `notebooks/` subdirectory tree | — | High |
| `NB_*` supplementary notebooks mixed with canonical sequence | root `*.ipynb` | Medium |
| `run_hex_bessel_like_checkpoint - Copy.py` duplicate in root | root | Low (gitignored) |
| `07_publication_materials_application.py` mirrors notebook name (confusing) | root | Low |
| `backups/` (21 timestamped dirs) co-located with active source | root `backups/` | Low (gitignored) |
| No `archive/` directory; stale files have nowhere to go | — | Medium |
| No `tools/` inside `Publication_Study/` (smoke test lives in repo root tools/) | — | Low |
| Docs numbered 00–05 with imprecise names; no project overview or limitations doc | `docs/` | Medium |
| Runner prints `[publication-study]` branding in all messages | runner | Low |
| `setup_study.py` REQUIRED_NOTEBOOKS hardcodes old root-level paths | `setup_study.py` | High (will break after Phase 2) |

What does NOT need changing right now:

- The physics in `bessel_twin_core.py` — do not touch in Phase 1/2.
- The `vbb_study/` package structure (already clean).
- The `equations/` subpackage (already exists; Phase 3 will audit it).
- Root-level compatibility shims (`bessel_twin_core.py`, `vbb_study/`) — keep as-is.
- `reference_kernels/` — provenance-only; stays where it is.
- `tests/` and `tools/` at repo root — not part of this phase.

---

## 2. Current State Map

```
c:\PhD\Code\                        ← repo root (not git-tracked)
  bessel_twin_core.py               ← compatibility shim (686 B)
  vbb_study/                        ← compatibility shim package (8 shim files)
  tests/                            ← 14 test files (stay at repo root for now)
  tools/smoke_test_publication_study.py  ← smoke test (stays at repo root)
  docs/                             ← OLD root docs (not used, kept for reference)
  outputs/                          ← OLD root outputs (not used)

Publication_Study/
  *.ipynb (19 notebooks, flat)      ← MOVING to notebooks/ subdirs
  bessel_twin_core.py (130 KB)      ← stays here (physics engine)
  publication_diagnostics.py (67 KB) ← stays (study-level helper)
  interface_correction_diagnosis.py ← stays
  run_publication_study.py (13 KB)  ← becomes compatibility wrapper
  run_*_checkpoint.py (6 files)     ← stay; are standalone analysis runners
  run_hex_bessel_like_checkpoint - Copy.py  ← MOVE to archive (duplicate)
  07_publication_materials_application.py  ← MOVE to archive (mirrors notebook name)
  finalize_publication_outputs.py   ← stays; new finalize_outputs.py wraps it
  __init__.py                       ← stays
  README.md                         ← UPDATE
  backups/ (21 dirs, gitignored)    ← MOVE to archive/backups/

  vbb_study/                        ← stays, active package
    __init__.py
    setup_study.py                  ← UPDATE (REQUIRED_NOTEBOOKS paths)
    study_taxonomy.py
    equations/ (5 submodules)       ← stays; Phase 3 will audit
    vbb_*.py (22 submodules)        ← stays; Phase 4 will reorganise internals

  docs/                             ← stays; new docs added alongside old
    00_theory.md                    ← rename to 01_theory.md in Phase 7
    01_conventions.md               ← rename to 02_metric_conventions.md in Phase 7
    02_validation.md                ← keep numbering
    03_materials_application.md     ← rename to 07_material_proxy... in Phase 7
    04_actual_lab_vector_case1.md   ← absorb into 06_hardware_routes.md in Phase 7
    05_study_taxonomy.md            ← keep numbering
    HEX_BESSEL_WRITING.md           ← move to archive/docs/ in Phase 7
    HEX_OUTLINE.md                  ← move to archive/docs/ in Phase 7
    HEXAGON_AMPLIFY.md              ← move to archive/docs/ in Phase 7
    PHYSICS_AUDIT.md                ← move to archive/docs/ in Phase 7
    [NEW] 00_project_overview.md    ← created Phase 2
    [NEW] 04_model_limitations.md   ← created Phase 2
    [NEW] 06_hardware_routes.md     ← created Phase 2
    [NEW] 08_refactor_plan.md       ← this file
    [NEW] 09_running_the_study.md   ← created Phase 2

  reference_kernels/                ← stays; provenance only
  outputs/                          ← stays; generated artifacts only
```

---

## 3. Target State (after all phases complete)

```
Publication_Study/
  README.md                         ← updated
  run_study.py                      ← NEW canonical runner
  run_publication_study.py          ← thin compatibility wrapper → run_study.py
  finalize_outputs.py               ← NEW thin wrapper → finalize_publication_outputs.py
  finalize_publication_outputs.py   ← kept for compatibility
  bessel_twin_core.py               ← physics engine (unchanged)
  publication_diagnostics.py        ← study helpers (refactored in Phase 4)
  interface_correction_diagnosis.py ← diagnostic helper

  notebooks/
    00_study_overview_and_conventions.ipynb
    scalar/
      01_scalar_bessel_baseline.ipynb          (stub → create in Phase 4)
      02_scalar_ideal_vs_lab_diagnostics.ipynb (was 01_publication_scalar_...)
      03_scalar_robustness_and_self_healing.ipynb (was 02_publication_robustness_...)
      04_scalar_parameter_sweeps.ipynb         (was 03_publication_parameter_sweep_atlas)
      05_scalar_validation_suite.ipynb         (was 04_publication_validation_benchmarks)
    lab_realism/
      01_holographic_axicon_route.ipynb        (was NB_holographic_axicon)
      02_physical_axicon_route.ipynb           (was NB_physical_axicon)
      03_holographic_vs_physical_axicon.ipynb  (was NB_axicon_method_comparison)
      04_objective_pupil_and_first_order_filtering.ipynb  (stub → Phase 8)
      05_through_sample_interface.ipynb        (was NB_through_sample)
      06_full_source_to_sample_journey.ipynb   (was NB_full_journey)
    vector/
      01_vector_beam_theory_atlas.ipynb        (was 05_publication_vector_parameter_atlas)
      02_vector_ideal_vs_lab_case1.ipynb       (was 06_publication_lab_vs_ideal_vector)
      03_vector_hardware_routes.ipynb          (stub → Phase 9)
    materials/
      01_material_proxy_fluence_and_thresholds.ipynb  (was 07_publication_materials_application)
      02_material_calibration_template.ipynb   (was NB_materials)
      03_application_design_tables.ipynb       (stub → Phase 10)
    advanced/
      01_capsule_weld_feature_design.ipynb     (was 09_publication_capsule_weld_feature_design)
      02_hexagonal_polygonal_beams.ipynb       (was NB_hexagon_study)
      03_discrete_nfold_beams.ipynb            (was 10_publication_discrete_nfold_beams)
    publication_exports/
      01_paper_figures.ipynb                   (stub → Phase 7)
      02_paper_tables.ipynb                    (stub → Phase 7)
      03_report_export.ipynb                   (was 08_publication_calibration_report_export)

  docs/
    00_project_overview.md           (NEW)
    01_theory.md                     (renamed from 00_theory.md — Phase 7)
    02_metric_conventions.md         (renamed from 01_conventions.md — Phase 7)
    03_validation.md                 (was 02_validation.md)
    04_model_limitations.md          (NEW)
    05_study_taxonomy.md             (unchanged)
    06_hardware_routes.md            (NEW; absorbs 04_actual_lab_vector_case1.md)
    07_material_proxy_and_calibration.md  (renamed from 03_materials_application.md — Phase 7)
    08_refactor_plan.md              (this file)
    09_running_the_study.md          (NEW)

  vbb_study/                         (all content as-is; reorganised internally in Phase 4)
    __init__.py
    setup_study.py                   (REQUIRED_NOTEBOOKS updated to new paths)
    study_taxonomy.py
    equations/                       (audited in Phase 3)
    core/                            (new subpackage in Phase 4)
    studies/                         (new subpackage in Phase 4)
    publication/                     (new subpackage in Phase 4)
    legacy/                          (old code archive in Phase 4)

  tools/
    smoke_test_study.py              (NEW; updated smoke test for new paths)
    inventory_repo.py                (NEW; list files by category)
    clean_outputs.py                 (NEW; wrapper around --clean-output)
    migrate_notebooks.py             (NEW; documents the Phase 2 moves)

  archive/
    old_notebooks/
      NB_validation.ipynb            (moved from root; replaced by 05_scalar_validation_suite)
    old_source_copies/
      run_hex_bessel_like_checkpoint - Copy.py
      07_publication_materials_application.py
    old_outputs/                     (empty; for future use)
    backups/                         (moved from Publication_Study/backups/)

  reference_kernels/                 (unchanged)
  outputs/                           (unchanged; generated only)
```

---

## 4. Migration Phases

### Phase 1 — Inventory and Plan (THIS PHASE — 2026-06-03)
- [x] Inspect full repo tree
- [x] Identify problems and risks
- [x] Write this plan (`docs/08_refactor_plan.md`)

### Phase 2 — Clean Tree and Naming (2026-06-03)
- [ ] Create `notebooks/` subdirectory structure
- [ ] Move and rename all notebooks
- [ ] Move stale/duplicate files to `archive/`
- [ ] Move backups to `archive/backups/`
- [ ] Create `run_study.py` (new runner; `--stage` support)
- [ ] Make `run_publication_study.py` a compatibility wrapper
- [ ] Create `finalize_outputs.py` wrapper
- [ ] Update `setup_study.py` REQUIRED_NOTEBOOKS
- [ ] Add new docs: `00_project_overview.md`, `04_model_limitations.md`,
  `06_hardware_routes.md`, `09_running_the_study.md`
- [ ] Update README
- [ ] Update `.gitignore` for `archive/backups/`

### Phase 3 — Equations Out of Notebooks
- [ ] Audit each notebook for reusable equations not yet in `vbb_study/equations/`
- [ ] Move equations to appropriate submodule in `equations/`
- [ ] Add `objective_pupil.py`, `interface.py`, `polygonal.py` to `equations/`
- [ ] Add docstrings and citations
- [ ] Update `equations/__init__.py`
- [ ] Add `tests/test_equations.py`

### Phase 4 — Clean Scalar Core
- [ ] Create `vbb_study/core/` subpackage
- [ ] Split `bessel_twin_core.py` internals into:
  - `core/config.py`, `core/grids.py`, `core/design.py`
  - `core/propagation_engine.py`, `core/hologram_engine.py`
  - `core/objective_engine.py`, `core/metrics.py`
  - `core/energy.py`, `core/validation.py`
- [ ] `bessel_twin_core.py` becomes a re-exporting facade
- [ ] Preserve all public imports

### Phase 5 — Runner and Setup (mostly done in Phase 2)
- [x] `run_study.py` with `--stage` support
- [ ] Add `project_schema_version` field to all manifests
- [ ] Manifest field: `git_commit` (already exists in current runner)
- [ ] `tools/clean_outputs.py` as standalone helper

### Phase 6 — Metric Schema and Outputs
- [ ] Define canonical scalar output schema in `vbb_study/studies/scalar_cases.py`
- [ ] All scalar CSVs must include the full field set (see Phase 6 requirements)
- [ ] Add power QA thresholds: pass ≤ 5 %, marginal ≤ 20 %, fail > 20 %
- [ ] Add test: canonical zone ≠ strict region confusion
- [ ] Regenerate scalar outputs

### Phase 7 — Documentation
- [ ] Rename existing docs to new numbering (check notebook references first)
- [ ] Merge `04_actual_lab_vector_case1.md` into `06_hardware_routes.md`
- [ ] Move `HEX_*.md`, `PHYSICS_AUDIT.md` to `archive/docs/`
- [ ] Fill in `01_theory.md`, `02_metric_conventions.md`, etc. with full content

### Phase 8 — Lab-Realism / Through-Sample
- [ ] Add `notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb`
- [ ] Audit and improve axicon notebooks
- [ ] Explicitly label planes: SLM → filter → objective pupil → focal/surface → sample

### Phase 9 — Vector Study
- [ ] Add `notebooks/vector/03_vector_hardware_routes.ipynb`
- [ ] Ensure every vector output carries the full vector metadata fields

### Phase 10 — Materials / Application
- [ ] Add `notebooks/materials/03_application_design_tables.ipynb`
- [ ] Ensure all material outputs carry `material_model_status`

### Phase 11 — Advanced Beams
- [ ] Audit capsule, hexagon, polygonal, discrete notebooks
- [ ] Add acceptance metrics for hexagon/discrete stability

### Phase 12 — Comments / Docstrings
- [ ] Module-level docstrings in all `vbb_study/*.py`
- [ ] Function docstrings for all public functions
- [ ] Inline comments for non-obvious physics

### Phase 13 — Tests
- [ ] `python -m compileall .`
- [ ] `python run_study.py --list`
- [ ] `python run_study.py --dry-run`
- [ ] `python tools/smoke_test_study.py`
- [ ] Import smoke test for all beam families

---

## 5. Compatibility Strategy

| Old reference | New location | Compatibility mechanism |
|---|---|---|
| `run_publication_study.py` | `run_study.py` | wrapper re-imports and re-exports |
| `finalize_publication_outputs.py` | `finalize_outputs.py` | wrapper imports and delegates |
| `00_publication_theory_conventions.ipynb` | `notebooks/00_study_overview_and_conventions.ipynb` | moved; archived old name |
| `01_publication_scalar_case_diagnostics.ipynb` | `notebooks/scalar/02_scalar_ideal_vs_lab_diagnostics.ipynb` | moved |
| `02_publication_robustness_metrics_visualisations.ipynb` | `notebooks/scalar/03_scalar_robustness_and_self_healing.ipynb` | moved |
| `03_publication_parameter_sweep_atlas.ipynb` | `notebooks/scalar/04_scalar_parameter_sweeps.ipynb` | moved |
| `04_publication_validation_benchmarks.ipynb` | `notebooks/scalar/05_scalar_validation_suite.ipynb` | moved |
| `05_publication_vector_parameter_atlas.ipynb` | `notebooks/vector/01_vector_beam_theory_atlas.ipynb` | moved |
| `06_publication_lab_vs_ideal_vector.ipynb` | `notebooks/vector/02_vector_ideal_vs_lab_case1.ipynb` | moved |
| `07_publication_materials_application.ipynb` | `notebooks/materials/01_material_proxy_fluence_and_thresholds.ipynb` | moved |
| `08_publication_calibration_report_export.ipynb` | `notebooks/publication_exports/03_report_export.ipynb` | moved |
| `09_publication_capsule_weld_feature_design.ipynb` | `notebooks/advanced/01_capsule_weld_feature_design.ipynb` | moved |
| `10_publication_discrete_nfold_beams.ipynb` | `notebooks/advanced/03_discrete_nfold_beams.ipynb` | moved |
| `NB_holographic_axicon.ipynb` | `notebooks/lab_realism/01_holographic_axicon_route.ipynb` | moved |
| `NB_physical_axicon.ipynb` | `notebooks/lab_realism/02_physical_axicon_route.ipynb` | moved |
| `NB_axicon_method_comparison.ipynb` | `notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb` | moved |
| `NB_through_sample.ipynb` | `notebooks/lab_realism/05_through_sample_interface.ipynb` | moved |
| `NB_full_journey.ipynb` | `notebooks/lab_realism/06_full_source_to_sample_journey.ipynb` | moved |
| `NB_hexagon_study.ipynb` | `notebooks/advanced/02_hexagonal_polygonal_beams.ipynb` | moved |
| `NB_materials.ipynb` | `notebooks/materials/02_material_calibration_template.ipynb` | moved |
| `NB_validation.ipynb` | `archive/old_notebooks/NB_validation.ipynb` | archived (content in 05_scalar_validation_suite) |
| `import bessel_twin_core` | unchanged | root shim still works |
| `import vbb_study` | unchanged | root shim still works |

Root-level compatibility shims (`c:\PhD\Code\bessel_twin_core.py`,
`c:\PhD\Code\vbb_study\`) are **not changed** in any phase.

---

## 6. Unresolved Risks

1. **Notebook bootstrap path resolution**: `setup_study.bootstrap()` uses
   `find_repo_root()` which traverses up from CWD.  Notebooks in deeper
   subdirectories (e.g. `notebooks/scalar/`) will work correctly because the
   traversal reaches `Publication_Study/` and finds `bessel_twin_core.py`.
   **Risk**: if a notebook explicitly constructs a path relative to its own
   `__file__` location (not CWD), the path would be wrong.  Each moved notebook
   should be run once with `--only` to confirm before marking Phase 2 complete.

2. **Large notebook outputs**: Notebooks 05 (6.1 MB) and 06 (8.2 MB) have
   substantial embedded outputs.  Moving them is a safe filesystem rename but
   if outputs reference embedded resource paths those would be stale.  Check
   visually after the first post-move run.

3. **`docs/00_theory.md` naming collision**: The new `00_project_overview.md`
   shares the `00_` prefix with the old `00_theory.md`. They coexist in Phase 2
   but Phase 7 must rename `00_theory.md` → `01_theory.md` before they conflict
   semantically.  Any notebook cell that does `open(paths["docs"] / "00_theory.md")`
   must be updated at the same time.

4. **`publication_diagnostics.py` size (67 KB)**: This file has a `TODO: split`
   comment in its header.  Phase 4 is the right time.  Do not touch it in Phase 2.

5. **Checkpoint runners**: `run_hex_*`, `run_zernike_*`, `run_hollow_*`,
   `run_polarized_*`, `run_polygonal_*` are standalone analysis runners that
   predate the study runner.  They are not registered in ORDERED_NOTEBOOKS and
   are not broken by this refactor.  Phase 4/5 should decide whether to fold
   them into the main runner or keep them as convenience scripts.

6. **Root-level `docs/` and `outputs/`**: `c:\PhD\Code\docs\` and
   `c:\PhD\Code\outputs\` exist from an earlier project layout.  They are not
   used by the current `Publication_Study` code.  Do not delete until confirmed
   they contain no unreferenced assets.

7. **REQUIRED_DOCS renaming**: Changing the numbering of existing docs (e.g.
   `00_theory.md` → `01_theory.md`) requires updating any notebook cell that
   references the old filename.  This is a Phase 7 task; do not rename during
   Phase 2.

---

## 7. Definition of Done for Phase 2

- [ ] All 19 notebooks moved to correct subdirectory with new names.
- [ ] `run_study.py` exists and `--list` shows all notebooks in stage order.
- [ ] `run_study.py --dry-run` runs without errors.
- [ ] `run_publication_study.py --list` still works (via wrapper).
- [ ] `setup_study.validate_workspace()` passes with updated REQUIRED_NOTEBOOKS.
- [ ] `python tools/smoke_test_study.py` (or repo root version) passes.
- [ ] Archive contains: duplicate .py file, materials .py runner, old NB_validation.
- [ ] `archive/backups/` contains all 21 old timestamped backup dirs.
- [ ] `.gitignore` updated for `Publication_Study/archive/backups/`.
- [ ] New docs created: `00_project_overview.md`, `04_model_limitations.md`,
  `06_hardware_routes.md`, `09_running_the_study.md`.
- [ ] No physics code was changed.
