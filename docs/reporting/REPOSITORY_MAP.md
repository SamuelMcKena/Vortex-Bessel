# Repository Map

**Freeze scope:** authoritative evidence preparation for the vortex Bessel report. The hexagonal
vector Bessel and experimental validation reports remain separate future products.

## Authoritative Areas

| path | role | report use |
|---|---|---|
| `vbb_study/` | importable model, solver, digital-twin and calibration source | authoritative source code |
| `tests/` | contract, regression and evidence-integrity tests | software verification |
| `tools/` | explicit pipelines and metadata builders | reproducibility tooling |
| `outputs/validation/` | governed CSV/JSON numerical records | quantitative evidence |
| `outputs/figures/phase2b_visual_diagnostics/` | accepted scalar propagation views | vortex report figures |
| `outputs/figures/phase2c/` | accepted vector objective/interface comparisons | vortex report figures |
| `calibration/templates/` | blank measured-data contracts | future experimental bridge |
| `docs/reporting/` | scope, claims, provenance and figure selection | report governance |
| `report/` | existing hexagonal-vector manuscript material | second-report scope |

The exact vortex subset and SHA-256 hashes are frozen in
`outputs/validation/report_freeze/vortex_report_manifest.json`. Quantitative report values must be
read from governed CSV/JSON or native arrays, never inferred from rendered pixels.

## Freeze Audit Summary

The completed on-disk inventory contains 9,872 files: 1,003 tracked, 75 untracked and 8,794
ignored. The large ignored count is dominated by temporary/cache trees and generated output trees;
it is not a recommendation to commit them wholesale. The selected vortex freeze contains 150 files
(29,797,122 bytes) and no selected file exceeds 90 MB.

The secret-signature scan found no credential pattern in tracked or untracked text. The reference
audit found no broken path in `README.md`, `AGENTS.md`, or `docs/reporting/`. It retained 68 broken
references in historical/non-report-ready documents. Four of those are genuinely absent Nathan
reference-image artifacts named by docs/52 and docs/53:

- `missing: outputs/reference/nathan_marco_report_figure4_page7_crop.png`
- `missing: outputs/reference/nathan_marco_report_figure4_page7_crop.provenance.json`
- `missing: outputs/reference/nathan_marco_report_page7_render.png`

These absences do not remove evidence required by the vortex report. Calibration paths containing
`<run_id>` are recorded as templates rather than incorrectly reported as missing files.

## Evidence Maturity

- Phase 1 and Phase 1R establish the repaired scalar physics and reconcile accepted outputs.
- Phase 2A defines canonical G0, B0, V1 and V3 cases and ideal-to-realistic route contracts.
- Phase 2B supplies governed two-dimensional, longitudinal and transverse 3D visual diagnostics.
- Phase 2C supplies vector Debye and Fresnel-interface quantitative references.
- Phase 2D supplies calibration schemas and dependency status; it does not supply measured data.
- Nathan/H1 materials are retained for the later hexagonal-vector report and are not evidence for a
  vortex claim unless a claim row explicitly names them.

## Repository-Wide Classification

`outputs/validation/report_freeze/repository_inventory.csv` classifies every file found on disk as
source code, tests, tools, governed numerical outputs, figures, reports/documentation, calibration
templates, report evidence, historical/superseded diagnostics, temporary/cache, duplicate legacy
notebook, patch, large binary, or unknown. It also records tracked, untracked, ignored, and export
disposition states. This generated inventory is the detailed audit; the table here is the human map.

## Excluded From A Standalone Git Export

- `.tmp_*`, `.pytest*`, `__pycache__`, `*.pyc`, and `vbb_study.egg-info` are disposable execution
  products.
- Root-level legacy notebooks and `vbb_study - Copy/` are duplicate working material.
- Root/archive ZIP bundles are transport snapshots, not authoritative evidence. In particular,
  `outputs.zip` and `archive/backups/pre_refactor_baseline_20260531_172040.zip` exceed 90 MB.
- Historical and superseded diagnostics remain on disk for provenance but are not automatically
  selected as report evidence.
- Ignored Phase 2B/2C figures listed in `vortex_report_files.csv` are intentional evidence and need
  an explicit force-add when preparing the private standalone repository.

No historical file is deleted by the freeze process.

## Audit Records

- `repository_inventory.csv`: complete on-disk file classification and Git disposition.
- `path_reference_audit.csv`: Markdown/LaTeX and explicit repository-path checks.
- `vortex_report_files.csv`: selected evidence, sizes, Git-ignore state and hashes.
- `vortex_report_claims.csv`: frozen copy of the report claim-to-evidence matrix.
- `vortex_report_manifest.json`: environment, branch, commit, solver provenance and blockers.

The freeze builder is metadata-only:

```powershell
C:\PhD\.venv2\Scripts\python.exe tools\build_vortex_report_freeze.py
```

It does not execute propagation, regenerate accepted numerical results, or modify optical physics.
