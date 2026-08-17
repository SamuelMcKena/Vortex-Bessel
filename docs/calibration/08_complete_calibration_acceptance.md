# 08 Complete Calibration Acceptance

## Purpose

Assemble reviewed measurements into one versioned JSON bundle, validate dependencies and preserve the
distinction between calibrated prediction and measured experimental validation.

## Required Equipment

Reviewed outputs from procedures 01-07, Stage 9A checksums, calibration certificates, the Phase 2D
templates/validator, version control status, and authorised laboratory governance.

## Beam-Safe Setup

This is a data-review procedure and should require no live beam. Any missing acquisition must return
to the relevant approved procedure under the laboratory's formal safety controls.

## Measurement Sequence

Review identities, dates, units, uncertainty definitions, repeats, acceptance decisions and hashes.
Populate a new JSON bundle without replacing nulls by assumptions. Run the validator, resolve errors,
review warnings, then run the canonical Phase 2D wrapper. Preserve both the bundle and validator log.

## Equation And Units

All bundle values use the canonical units encoded in schema version `1.0`: metres, joules, radians,
degrees where named, metres per pixel, and dimensionless fractions. Derived quantities must cite the
input rows and equation used.

## Repeats And Uncertainty

Use the repeat requirements from procedures 01-07. The final bundle carries standard uncertainties
from those procedures; Phase 2D must not invent replacements. Use a fixed Monte Carlo seed for an
auditable software run and record sample count and failed samples.

## Acceptance Criteria

`valid_schema=True`; no impossible values or unit conflicts; every intended calibrated claim has no
missing/non-calibrated dependency; exactly one selected first-order factor; upstream hashes unchanged;
all required regressions pass. Synthetic data cannot pass laboratory acceptance.

## Output Format

Store the reviewed bundle outside `calibration/templates/` with immutable Stage 9A evidence links.
Archive validator output and the generated Phase 2D manifest. Do not overwrite accepted Phase 2A,
2B or 2C artifacts.

## Code Mapping And Claims

Run:

```powershell
python tools/validate_calibration_bundle.py calibration/my_lab_calibration.json
python tools/run_phase2d_calibration_bridge.py
```

A complete real bundle can mature eligible claims to calibrated optical or calibrated fluence. The
final experimentally validated level additionally requires an accepted measured-output comparison,
an evidence path and passed comparison criteria.
