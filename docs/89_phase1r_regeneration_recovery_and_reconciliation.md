# PHASE 1R - Regeneration, Recovery and Repository Reconciliation

**Outcome:** `PHASE1R-B`.

The Phase 1 source repairs remain valid. All 85 repair-affected registry records now have exactly
one reconciliation action, the definitely stale physical/Fourier artifacts have been regenerated,
and inherited high-drift data have explicit row-level quantitative status. The repository does not
qualify for `PHASE1R-A` because 116 of 119 deduplicated high-drift physical configurations remain
unconverged within the bounded campaign.

No new physics was added, no tolerance was weakened, no Nathan MODE 2/V0 artifact was regenerated,
and no commit was created.

## Artifact Disposition

| reconciliation action | Phase 1 registry records | post-Phase 1R interpretation |
|---|---:|---|
| `regenerate_unchanged_sampling` | 21 | current topology/Fourier artifact, with numerical validity retained separately |
| `rerun_with_convergence_repair` | 57 | inherited CSV re-exported with row-level validity; converged replacements live in the Phase 1R manifest |
| `reexport_metadata_only` | 3 | three directory scopes, expanding to 34 mapping-bearing CSV files |
| `retain_blocked_historical_diagnostic` | 4 | legacy duplicate notebooks/summaries remain historical only |
| **total** | **85** | no unclassified record |

The exact record-level table is
`outputs/validation/phase1_reconciliation/phase1r_artifact_disposition.csv`.

## Physical Vortex Regeneration

The affected Stage C physical route, holographic/physical comparison, Stage D through-sample route,
and Stage E full journey were regenerated using `slm2_conjugate_mode=preserve_vortex`.

Every regenerated physical row measures winding from the actual complex surface field on a closed
contour. The requested charge is 3; all measured values are 3.0 to floating-point precision and
pass the predeclared `abs(measured_winding - requested_ell) < 0.1` gate.

| contract | inherited normal route | regenerated normal route |
|---|---:|---:|
| requested charge | 3 | 3 |
| measured winding | 0 | 3.0 |
| vortex-removal acknowledgement | absent | false |
| SLM2 mode | silent/full | `preserve_vortex` |

The two root-level legacy notebooks and their `NB_*` summaries were not refreshed. They remain
explicit historical diagnostics and are superseded by the authoritative lab-realism notebooks.

## Fourier Geometry Regeneration

`notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb`, its Stage C CSV, and its
controlled executed copy were regenerated using the wavelength-correct formulas.

| regime | inherited radius (mm) | regenerated radius (mm) |
|---|---:|---:|
| general | 8238.245119 | 0.008477154227779409 |
| limits | 37072.103037 | 0.038147194025007346 |

The independent Nathan 1029 nm, 300 mm, 6.25 lp/mm carrier control remains `1.929375 mm`.

## High-Drift Deduplication

The 415 inherited rows in 57 CSV files reduce to **119 unique physical configurations** using a
declared fingerprint of optical path, generation/vector route, charge, target scale/length,
objective/device parameters, propagation method, regime and correction state. Z-profile and report
duplicates retain every source-row reference in `phase1r_convergence_manifest.json`.

Crop-only loss is excluded as the drift cause because the authoritative drift integrates each full
propagated grid before display cropping. The dominant diagnosed mechanisms are z-dependent BL-ASM
bandlimit clipping, insufficient physical window at high transverse wavevector or long distance,
and the fixed focal-window limitation of downsampled realistic SLM simulations.

## Convergence Campaign

The tolerances were fixed before interpretation:

- propagation power drift: at most 5%;
- ring/feature radius: at most 3% between the final adequate resolutions;
- canonical and strict regions: at most 5%;
- sampled peak intensity and side-lobe ratio: at most 5%.

Ten bounded representative configurations were tested. Three recovered quantitatively:

| case | final adequate pair | final drifts | maximum metric change |
|---|---|---|---:|
| `general_holographic_ideal` | N=768, 1024 | 0.004934, 0.001888 | 0.094% |
| `general_physical_ideal` | N=768, 1024 | 0.003762, 0.001196 | 0.052% |
| `near_threshold_D8_L150_ideal` | N=768, 1024 | 0.013149, 0.006932 | 0.060% |

The other seven selected configurations remain `invalid_unconverged`. In particular:

- limits ideal/physical cases still have drift about 0.745 at N=1536;
- the extreme D1/L800 and D3/L800 controls still have drift 5.019 and 1.444 at N=1536;
- realistic device cases recover power conservation after `device_downsample: 4 -> 2` doubles the
  focal window, but their highest-resolution metrics still move beyond tolerance at N=1536/2048;
- SAS was not forced outside validity: the canonical comparison was rejected at 600 um because its
  N=768 reference z-limit is 430 um.

Thus **3 of 119** unique cases have direct recovered replacements and **116 remain blocked**. The
415 inherited low-resolution rows themselves remain diagnostic even where a new high-resolution
replacement exists; their old numerical values were not relabelled as valid.

## Mapping Provenance

Thirty-four mapping-bearing CSV files were metadata-only re-exported. Existing numerical strings
were preserved, and the following fields were added or repaired:

```text
mapping_mode = target_matched_inverse_design
objective_map_source = compute_design_from_targets:w0_sample/beam_radius_on_slm
objective_map_demag = historical magnification_to_sample
mapping_claim_scope = inverse_design_feasibility
hardware_target_achieved = false
```

These rows are not fixed-bench predictions. The Phase 1 fixed-optics regression still verifies that
changing the requested target cannot retune physical demagnification.

## Claims and Unaffected Outputs

The final registry contains 14 claims: seven `validated`, three `validated_with_scope`, two
`superseded`, one `historical_only`, and one explicit `blocked_unconverged` convergence claim. No
ambiguous status strings remain.

Confirmed unaffected:

- Nathan V0/source-scale controls and Nathan MODE 2 results;
- Nathan F300 carrier displacement and operating point;
- acknowledged full-conjugation characterisation-lock diagnostic;
- historical target-matched scalar numerical values;
- the frequency-domain Stage C carrier/cone diagram.

Still blocked:

- quantitative use of the 415 inherited high-drift row values;
- 116 deduplicated configurations without converged replacements;
- realistic-route metrics that conserve power but fail metric stability;
- any fixed-bench wording attached to historical inverse-design exports.

## Outputs

```text
outputs/validation/phase1_reconciliation/
    phase1r_artifact_disposition.csv
    phase1r_claim_disposition.csv
    phase1r_regeneration_manifest.json
    phase1r_convergence_manifest.json
    phase1r_final_claim_registry.csv
    phase1r_convergence_runs.csv
    phase1r_selected_convergence_results.json
```

## Execution and Regression

- controlled notebook execution: five affected notebooks passed with zero saved error outputs;
- bounded convergence ladder: 271.4 s;
- realistic focal-window ladders: 261.2 s total;
- focused Phase 1R artifact tests: 10 passed in 6.99 s;
- focused Phase 1 plus Phase 1R plus vortex end-to-end: 32 passed in 22.58 s;
- core physics, characterisation locks, vortex and Phase 1/1R: 89 passed, 4 xfailed in 1011.17 s;
- objective/Fourier/carrier-stop/F300: 96 passed, one intentional dirty-file guard deselected, in
  138.22 s;
- broader Nathan MODE 2 regression: 195 passed in 140.09 s;
- authoritative `tests/` collection: 1043 tests, zero collection errors, in 5.90 s.

The four xfails are the unchanged fast limits-grid ring/zone cases. Sandboxed objective and Nathan
runs encountered Windows `tmp_path` ACL errors before affected tests executed; the exact selections
passed in approved unsandboxed reruns. This did not alter test selection or tolerance.

`git diff --check` is required before closure. No commit is authorised.
