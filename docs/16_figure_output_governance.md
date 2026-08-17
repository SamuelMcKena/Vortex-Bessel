# Figure And Output Governance

Stage 8.6 introduced the registry gate for figures, tables, JSON metadata, and
legacy output families. Final export code must classify an artifact through
`vbb_study.publication.figure_registry` before consuming it.

## Status Meanings

| status | meaning |
| --- | --- |
| `final_export_allowed` | Allowed for later final report or paper assembly with required caveats and QA labels. |
| `diagnostic_allowed` | Useful for review/debug/research, blocked from final export by default. |
| `exploratory_only` | Development-only, blocked from final export. |
| `legacy_quarantined` | Old or compatibility artifact retained for traceability, blocked from final export. |
| `rejected` | Misplaced, stale, unknown, or unsafe artifact, blocked. |

## Stage 8.7 Quick-Look Outputs

Stage 8.7 adds the quick-look simulator at
`notebooks/quicklook/00_quick_beam_to_sample_simulator.ipynb`.

Its outputs are intentionally diagnostic:

- `outputs/csv/quicklook/quicklook_metric_summary.csv`
- `outputs/csv/quicklook/quicklook_four_condition_phase_comparison_summary.csv`
- `outputs/csv/quicklook/four_condition_phase_comparison_metrics.csv`
- `outputs/json/quicklook/quicklook_config.json`
- `outputs/figures/quicklook/*.png*`

The notebook is live-first: `save_outputs=False` is the default in-notebook
workflow, and CSV/JSON/PNG writing is opt-in. The registry marks these families
as `diagnostic_allowed`, with `final_export_allowed=False`. They are not
publication-export sources. Use them to inspect parameters, phase-mask
conventions, quick lab/ideal differences, and the four-condition phase-state
comparison before rerunning the governed stage at balanced/publication sampling.

Stage 8.7D adds visual-physics guardrails to the quicklook path. The notebook
now separates the true scalar Bessel-Gauss target, the conical axicon
propagated field, the holographic lab-realistic route, the through-sample route,
and the material proxy. Lab-realistic fields are usable only when the raw-array
visual sanity checks and first-order/filter sanity checks pass. If those checks
fail, figures and tables must say `visual sanity failed — do not use this as a
beam prediction.`

Quick-look figures may use labelled display interpolation, but interpolation is
visual only; metric calculations use the raw sampled arrays. Fail and marginal
propagation-power labels must remain visible and must not be relaxed. The
material proxy is blocked when upstream optical/sample sanity fails.

## Export Rule

Use:

```python
from vbb_study.publication.figure_registry import assert_export_allowed

item = assert_export_allowed("Publication_Study/outputs/figures/publication_study/example.png")
```

If the call raises `PublicationExportGateError`, the artifact must not be used
in final report or paper export unless a deliberate future registry change
allow-lists it.
