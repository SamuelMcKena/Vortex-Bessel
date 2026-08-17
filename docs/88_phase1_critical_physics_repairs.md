# Phase 1 Critical Physics Repairs

**Verdict:** `PHASE1-B` - all four critical contracts are implemented, but inherited artifacts from
the physical-axicon route, old Fourier geometry, and propagation cases above the 5% drift limit need
regeneration or re-export before quantitative reuse.

No accepted output was silently regenerated. No commit was made.

## Findings Addressed

### 1. Physical-axicon vortex preservation

The physical route now defaults to `preserve_vortex`. For nonzero charge, explicit `full`
conjugation raises unless `allow_vortex_removal=True`. The acknowledged route remains available as a
zero-winding diagnostic, and `ell=0` remains valid with `full`.

```text
unsafe full correction: exp[-i(ell phi + phi_prop)]
safe correction:       remove phi_prop while retaining exp(i ell phi)
```

The end-to-end test measures winding on the propagated complex surface field; it does not merely test
a configuration string.

### 2. Fourier-plane physical geometry

```text
old ring radius:      f k_r / (2 pi)
new ring radius:      lambda f k_r / (2 pi) = f k_r / k0
old carrier distance: f nu
new carrier distance: lambda f nu
```

Both helpers require wavelength explicitly. See
`docs/88_phase1_fourier_geometry_impact_audit.md` for the direct artifact audit.

### 3. Propagation power-drift validity

`propagate_volume` owns the authoritative drift:

```text
drift = (max(P_z) - min(P_z)) / mean(P_z)
quantitative validity requires drift <= 0.05
```

Every returned volume and scalar metric row now carries `quantitative_metrics_valid`, an invalid
reason, the evaluated flag, the drift and its label. Missing drift is invalid/not-evaluated and is
never replaced with zero. The shared validity policy supports `flag`, `warn`, and `raise`.
Intentional first-order filtering remains separate from numerical z-stack drift.

The inherited-output scan found 61 CSVs containing the drift field. In 57 files, 415 rows exceed
5%; those rows are blocked from quantitative reuse under the repaired contract. Their stored drift
numbers are retained as diagnostics.

### 4. Explicit optical mapping mode

The two authoritative values are:

- `target_matched_inverse_design`: requested target waist/length may determine required demagnification;
  claim scope is inverse-design feasibility.
- `fixed_physical_optics`: relay/objective mapping is fixed; target changes cannot retune the map and
  target mismatch is reported.

Major scalar result rows expose `mapping_mode`, `objective_map_source`, `objective_map_demag`,
predicted length, claim scope and fixed-bench target status. The historical default remains
target-matched inverse design, preserving its numerical results while removing the semantic ambiguity.
The old `magnification_to_sample` column is retained as a compatibility field for the configured
objective/relay value; `objective_map_demag` and `objective_map_source` are the authoritative fields
under the new contract. Matching `legacy_objective_map_*` fields make this distinction explicit.

## Backwards Compatibility

- Historical scalar default: retained exactly as `target_matched_inverse_design`.
- Characterisation locks that intentionally measure the old `full` physical diagnostic now set
  `allow_vortex_removal=True`; their numerical baselines are intentionally unchanged.
- `ell=0 + full`: unchanged and allowed.
- Fixed relay/objective mapping: never falls back to a target-derived waist map.
- Nathan source-scale/V0 route: not rebound to `ObjectiveMap` and not modified.
- Nathan 4F displacement law: already correct and unchanged.

## Xfails

Pre-edit targeted baseline: `45 passed, 17 xfailed`. Four physical winding xfails, six excessive
power-drift xfails and three physical ring-radius xfails are converted to passing regressions because
the vortex-preserving route now carries its intended topology. Four fast-grid ring-radius/zone
findings remain xfailed; tolerances were not weakened.

## Verification

- Focused Phase 1/vortex suite: `25 passed`.
- Objective/pupil/carrier/F300 suite: `80 passed, 1 deselected`; the deselected test is the governance
  guard that deliberately rejects any dirty change to `bessel_twin_core.py`.
- Core physics/characterisation suite: `61 passed, 4 xfailed` in `981.59 s`.
- Broader Nathan MODE 2 regression: `195 passed` in `143.06 s`.
- Active `tests/` collection: `1033 collected`, no collection errors. Repository-root collection also
  entered archived duplicate test trees and inaccessible historical temp folders: `1083 collected`
  before `24` pre-existing collection errors.
- Python compilation passed. `git diff --check` passed with line-ending warnings only.

## Files Changed By Phase 1

Core and contracts:

- `PHYSICS_VALIDATION_FINDINGS.md` (historical findings retained with a Phase 1 supersession notice)
- `bessel_twin_core.py`
- `vbb_study/config.py`
- `vbb_study/design.py`
- `vbb_study/vbb_planes.py`
- `vbb_study/equations/objective_pupil.py`
- `vbb_study/vbb_regime.py`
- `vbb_study/vbb_axicon.py`
- `vbb_study/vbb_train_viz.py`

Config-aware study call sites and result provenance:

- `vbb_study/vbb_capsule.py`
- `vbb_study/vbb_materials_study.py`
- `vbb_study/vbb_polygonal.py`
- `vbb_study/vbb_sample_study.py`
- `vbb_study/vbb_studies.py`
- `vbb_study/vbb_validation.py`
- `vbb_study/vector_axicon.py`
- `vbb_study/publication/quicklook.py`

Notebook/tool call site:

- `tools/update_lab_realism_notebooks.py`
- `tools/capture_prod_baselines.py` (legacy full-conjugation diagnostic explicitly acknowledged)
- `notebooks/lab_realism/02_physical_axicon_route.ipynb`
- `notebooks/lab_realism/03_holographic_vs_physical_axicon.ipynb`
- `notebooks/lab_realism/05_through_sample_interface.ipynb`
- `notebooks/lab_realism/06_full_source_to_sample_journey.ipynb`
- `notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb` (code cell only; old
  saved outputs intentionally retained and marked stale)

Tests:

- `tests/test_phase1_critical_physics_repairs.py`
- `tests/test_physics_validation.py`
- `tests/test_slm2_preserve_vortex_end_to_end.py`
- `tests/test_characterisation_lock.py`
- `tests/test_characterisation_lock_prod.py`

Validation and audit artifacts are the two `docs/88_*` files and the four files under
`outputs/validation/phase1_critical_repairs/`.

## Regeneration And Claim Status

Exact paths/scopes are listed in `phase1_affected_outputs.csv`; claim-level decisions are in
`phase1_claim_status.csv`.

- Physical-route Stage C, through-sample Stage D and full-journey Stage E fields, comparisons and
  summaries generated with silent full conjugation: regenerate with the vortex-preserving default.
- The Stage C objective-pupil geometry CSV and executed notebook: regenerate with explicit wavelength.
- Existing rows with propagation drift above 0.05: no quantitative reuse until rerun/re-export adds
  the invalidity fields; their old metrics may be retained only as invalid diagnostics.
- Historical scalar exports without mapping labels: retain their numbers as inverse-design results,
  but re-export before describing them as fixed-bench predictions.

## Remaining Limitations

- The hard gate prevents interpretation; it does not repair BL-ASM clipping in large-`k_r` cases.
- The 3% fixed-bench target-match tolerance is a project reporting threshold, not a universal law.
- Physical-route and Fourier-facing inherited artifacts remain stale by design in this pass.
- Four fast-grid ring-radius/zone xfails remain open.
