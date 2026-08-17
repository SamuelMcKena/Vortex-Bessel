# Phase 2D Canonical Vector Route and Experimental Calibration Bridge

**Outcome: PHASE2D-B.** Solver governance is complete, but important predictions remain blocked by
missing laboratory calibration values. This phase integrates accepted Phase 2A and Phase 2C evidence;
it does not introduce another optical solver or overwrite an accepted upstream artifact.

## Governed Route

The public entry point is `run_canonical_optical_prediction`. Its allowed fidelity modes are
`fast_scalar_screening`, `quantitative_vector_reference`, and `automatic_by_claim`. In automatic
mode G0/B0 global morphology remains scalar; longitudinal and component claims use vector Debye.
V1/V3 peak location uses Debye, while a ring-radius result explicitly names its scalar or vector
reference. H1 uses scalar FFT only for fast broad-shape screening and Debye for feature radius,
edge sharpness, ridge/transition width, components, longitudinal power, and dimensional/fluence
claims. Component-resolved interface claims use spectral vector Fresnel; bounded morphology and
interface-power screening may use the accepted scalar normal-incidence approximation.

The wrapper returns governed references to the accepted Phase 2A route, power ledger, and Phase 2C
benchmark records. It does not persist a second hardware manifest, regenerate beam families, or
reapply the Phase 2A sample/interface factor. Every realistic ledger contains exactly one
`simulated_selected_first_order` factor.

## Calibration Contract

The versioned JSON schema is represented by
`calibration/templates/canonical_lab_calibration_template.json`. Nulls are intentional. The loader
does not replace them with assumed values. The command-line validator is:

```powershell
python tools/validate_calibration_bundle.py calibration/my_lab_calibration.json
```

It reports schema errors, missing values, unit conflicts, impossible values, operator warnings,
and enabled/blocked claims. Raw measurement acquisition remains governed by the existing Stage 9A
acquisition and checksum package; Phase 2D consumes reviewed calibration results and does not replace
that raw-data provenance layer.

## Missing Evidence

Current calibrated blockers include pulse energy; beam radius; SLM LUT, stroke, orientation, and
date; 4F focal/iris/order geometry; effective objective focal length, pupil and fill; relay and camera
scale; axicon angle/index/aperture; per-stage transmissions; and material index, coating state and
surface orientation. Manufacturer values can satisfy an identified manufacturer dependency, but
assumed and placeholder values do not mature a calibrated claim.

## Uncertainty

The uncertainty engine supplies deterministic nominal evaluation and fixed-seed normal Monte Carlo
propagation. It samples only uncertainty values present in the bundle. A missing dependency returns
`unavailable_missing_calibration`. A supplied zero uncertainty produces zero propagated uncertainty.
Energy propagation uses pulse energy, measured stage transmissions, and the accepted selected-order
efficiency exactly once. H1 edge-sharpness uncertainty remains
`unavailable_numerical_stability_not_demonstrated`; Phase 2C did not establish a dedicated stability
bound for that derived metric.

The partial and full populated bundles under `outputs/validation/phase2d/synthetic_bundles/` are
software tests labelled `synthetic_not_experimental`. The full bundle exercises dimensions, energy,
fluence, and uncertainty, but cannot unlock a real absolute claim.

## Maturity

Claim maturity is ordered as: relative optical prediction, fixed-bench nominal prediction,
calibration-ready prediction, calibrated optical prediction, calibrated fluence prediction, and
experimentally validated prediction. Synthetic data is capped at calibration-ready. Experimental
validation additionally requires an accepted measured-output comparison with an evidence path.

At closure, absolute dimensions are not unlocked, absolute fluence is not unlocked, and no result is
experimentally validated. H1 automatic routing selects `vector_debye`, and the accepted vector
strict-hexagon result remains true.

## Laboratory Procedures

The eight procedures in `docs/calibration/` define the controlled measurement bridge. All work must
follow the laboratory's formal laser safety procedure, engineering controls, approved risk
assessment, and authorised-person requirements. These documents do not authorise open-beam alignment.

## Authoritative Outputs

Machine-readable policy, dependency, readiness, demonstrations, uncertainty, maturity, claims,
outcome, and hashes are under `outputs/validation/phase2d/`. The 2D readiness, policy, uncertainty,
and H1 state figures are under `outputs/figures/phase2d/`. No decorative or propagation-axis 3D plot
is used in this calibration dashboard.
