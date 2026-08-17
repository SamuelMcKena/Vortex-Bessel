# Structured-Beam PhD Study — clean working tree

This branch is a clean, current working tree for the structured-beam PhD codebase. It is built from the Phase 2K mathematical/physics audit and then integrates the current experimental axicon-aberration correction study.

## Scientific status

The repository deliberately separates four levels of evidence:

1. **Analytic / numerical reference** — equations and independent numerical checks used to validate solvers.
2. **Nominal optical-system prediction** — simulations of the dual-SLM, 4F and physical-axicon route before complete bench calibration.
3. **Measured experimental evidence** — camera/BeamGage data and quantities derived directly from those measurements.
4. **Hardware-validated correction** — requires calibrated camera-to-SLM mapping, SLM LUT/phase stroke, beam footprint/parity and a new post-correction z-scan that passes the experimental acceptance gates.

The current aberration-correction phase maps are **model predictions**, not post-correction camera measurements.

## Canonical source layout

- `vbb_study/` — authoritative beam models, propagation, optical-route and digital-twin source.
- `notebooks/` — curated analysis notebooks by topic; quicklook/export-only notebooks are omitted.
- `notebooks/experimental/axicon_aberration_correction/` — current measured q=20 z-scan reconstruction and correction work.
- `reference_kernels/` — independent/reference numerical kernels.
- `tests/` — numerical, physics-contract and regression tests.
- `tools/` — reproducibility, validation and audit utilities.
- `calibration/` — calibration contracts/templates and bench-calibration support.
- `docs/` — theory, conventions, limitations, calibration and reporting documentation.
- `references/` — literature/reference material used by the codebase.
- `outputs/validation/` — governed validation records only; bulk historical generated outputs are excluded from this clean tree.

## Figure policy

The clean tree does not keep every historical render. For the current q=20 aberration-correction study, figures are curated under:

`notebooks/experimental/axicon_aberration_correction/figures/current_q20/`

That directory favours the newest realigned, comprehensive-validation, single-mask, phase-error-recreation and closed-loop outputs. Pre-realignment duplicates and the earliest root-level correction plots are intentionally omitted.

Older non-aberration figure families are not promoted here merely because they exist. Phase 2K requires their generating paths to satisfy the relevant analytic/reference, independent numerical, convergence and hardware-provenance gates before they are treated as current scientific evidence.

## Reproducibility

Install the report environment with `requirements-report.txt`; the experimental q=20 correction package also has its own local `requirements.txt`.

Run core checks with pytest and compileall, e.g.:

```powershell
python -m pytest tests -q
python -m compileall -q vbb_study tools tests
```

The experimental axicon-correction package contains additional controller tests and explicit hardware-readiness blockers.

## Provenance

See `CLEANROOM_PROVENANCE.md` for the source branches, integration strategy and claim boundaries used to build this clean working tree.
