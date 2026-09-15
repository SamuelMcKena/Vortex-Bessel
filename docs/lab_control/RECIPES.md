# Experiment recipes

`RecipeEngine` is Qt-independent. A `RecipeDefinition` declares prerequisites
and ordered steps; a `RecipeRun` records the starting state revision, parameters,
completed results, current step, status and errors. Runs serialise to JSON and
resume without rebuilding logic from GUI state.

## Implemented definitions

| Recipe | Purpose |
|---|---|
| `measure_beam_walk` | Guided repeated z captures and beam-camera relative fit |
| `z_stack` | General quantitative z-stack acquisition |
| `vortex_charge_scan` | Arbitrary integer charge scan with state-aware metrics |
| `correction_gain_scan` | Retrieved-map gain scan with fresh verification |
| `optimise_zernike` | Signed, drift-bracketed low-order command optimisation |
| `gaussian_characterisation` | q=0 centre, width, symmetry and propagation metrics |
| `commission_q20` | Dense q20 commissioning and golden-reference creation |
| `recover_q20` | Quick three-plane q=0/q20 diagnosis, correction and verification |

q20 is one recipe in the catalogue, not an application-wide mode.

## Step kinds

Recipes use reusable `CONFIGURE`, `GUIDED_MOVE`, `CAPTURE`, `ANALYSE`,
`OPTIMISE`, `VERIFY`, `ACCEPT_OR_ROLLBACK` and `REPORT` steps. Current manual z
motion yields a prompt. A future stage provider can satisfy the same step without
altering analysis.

## Prerequisites and resumption

Starting a recipe checks the calibration registry. Missing or stale requirements
raise `RecipePrerequisiteError` with exact keys. Completing a step appends an
immutable result record and publishes the next step to `ExperimentState.session`.

Save after every consequential step:

```python
RecipeEngine.save(run, session_root / "recipe_run.json")
run = RecipeEngine.load(session_root / "recipe_run.json")
```

The engine sequences work; physical handlers execute steps. Automated handlers
must still use `FormalCaptureService` and `SensorlessOptimiser` so recipe code
cannot bypass cast verification, provenance or rollback.
