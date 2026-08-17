# Report Scope and Claim Maturity

## Report Order

1. Vortex-Bessel numerical and lab-realistic study: G0, B0, V1, V3.
2. Hexagonal-vector Bessel study: H1 and Nathan MODE 2 evidence.
3. Experimental calibration and validation study: measured hardware and measured-output comparison.

The first report may cite H1 only as an out-of-scope branch. It must not borrow H1 success to imply a
vortex result, or vice versa.

## Maturity Vocabulary

| Maturity | Meaning |
|---|---|
| `relative_optical_prediction` | Numerical field/metric under explicit model assumptions |
| `fixed_bench_nominal_prediction` | Canonical hardware binding with assumed or manufacturer values |
| `calibration_ready_prediction` | Software dependencies are explicit, but real measurements are incomplete |
| `calibrated_optical_prediction` | Real optical geometry/scale calibration is supplied |
| `calibrated_fluence_prediction` | Real energy, throughput and physical-area calibration are supplied |
| `experimentally_validated_prediction` | A real measured-output comparison passes accepted criteria |

Levels are not interchangeable. Synthetic calibration data is capped at calibration-ready.

## Supported for the Vortex Report

- Analytic G0/B0/V1/V3 definitions and controlled route taxonomy.
- Repaired vortex preservation and topological winding in simulation.
- Native numerical ring, dark-core, side-lobe, axial and convergence metrics with named planes.
- Fixed-bench nominal ideal/realistic/mild/degraded comparisons.
- Relative energy-ledger closure and single-count selected-order efficiency.
- Phase 2C scalar-versus-Debye and scalar-versus-vector-Fresnel bounded comparisons.
- Phase 2B 2D and transverse intensity-surface presentation with native-metric provenance.

## Narrowed Claims

- Scalar vortex morphology does not authorise quantitative peak location; use Debye.
- Ring-radius numbers must identify plane and scalar/vector reference.
- Old target-matched or unreconciled Stage 6 exports are diagnostic, not canonical fixed-bench data.
- Mild/degraded routes are controlled diagnostics, not measured bench states.
- The scalar interface result is bounded to the tested nominal interface and claim class.

## Calibration Blockers

Real camera/objective scale, pulse energy, beam radius, SLM LUT/stroke/orientation, 4F geometry,
objective pupil/fill, relay magnification, axicon geometry, per-stage transmission, and material
interface state remain unresolved or not repository-verified. Therefore absolute dimensions,
absolute fluence, calibrated masks, and experimental agreement remain blocked.

## Excluded Physics

No nonlinear response, pulse accumulation, thermal response, damage threshold, ablation, or material
modification is predicted. A linear optical field or fluence proxy cannot be relabelled as material
outcome evidence.
