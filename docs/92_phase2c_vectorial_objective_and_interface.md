# Phase 2C Vectorial Objective and Fresnel Interface Benchmark

**Outcome:** `PHASE2C-B`. reference solvers validate, but one or more scalar objective/interface approximations are materially different.

**Scope:** independent optical-reference benchmark only. The accepted Phase 1/1R/2A/2B arrays were
not overwritten (`upstream hashes unchanged = True`). Debye amplitudes remain relative
and are not eligible for the Phase 2A absolute energy ledger. No nonlinear response, modification,
microfabrication or calibrated sample-plane claim is made.

The canonical mapping contract is `fixed_physical_optics`. No target-matched optics
were derived for these comparisons.

## Solver Validation

The polar Debye controls and the Cartesian sine-condition reference both use the declared
`sqrt(cos(theta))` aplanatic convention and `relative_morphology_reference` field normalisation.
Uniform x/y, radial, azimuthal, four-level low-NA, global-phase and linear-scaling controls pass.

| level | Nr | Nphi | corr to high | corr change | feature change | longitudinal change |
|---|---:|---:|---:|---:|---:|---:|
| `low` | 24 | 72 | 0.999999574 | baseline | baseline | baseline |
| `medium` | 32 | 96 | 0.999999593 | 4.854e-08 | 0.000e+00 | 1.617e-06 |
| `high` | 48 | 144 | 1.000000000 | 4.071e-07 | 0.000e+00 | 8.672e-05 |

All adjacent quadrature changes meet the predeclared `1e-3`, `1%`, and `1e-3 absolute` limits.

The final accepted-model benchmark uses the native Cartesian Debye transform at pupil `N=1024` with
2x zero-padding for G0/B0/V1/V3 and 4x zero-padding for H1, giving H1
`0.1029 um` focal sampling.
The `N=512 -> 1024` reported scalar/vector correlation changes are at most
`1.784e-04` and longitudinal-fraction changes are at most `1.140e-04`; all
predeclared convergence requirements pass.

Normal-incidence s/p coefficients agree at
`0.581395349`. The Brewster p-reflection
residual is `1.200e-16`, spectral plane-wave
`R+T` is `1.000000000000`, and the
transmitted transversality residual is
`6.825e-18`.

## Objective Benchmark

| case | scalar/vector corr | feature radius rel. diff | peak shift (um) | longitudinal fraction | class |
|---|---:|---:|---:|---:|---|
| `G0` | 0.999473 | 0.0000 | 0.000 | 3.792% | `morphology_equivalent` |
| `B0` | 0.993095 | 0.0000 | 0.000 | 10.013% | `morphology_equivalent` |
| `V1` | 0.993143 | 0.0000 | 1.029 | 10.145% | `morphology_shifted` |
| `V3` | 0.994780 | 0.0000 | 2.301 | 10.123% | `morphology_shifted` |
| `H1` | 0.996356 | 0.1333 | 0.000 | 10.150% | `morphology_non_equivalent` |

G0 and B0 meet the project morphology-equivalent gate. V1 and V3 retain high correlations and
unchanged ring radii but are `morphology_shifted` because the brightest point moves by more than one
scalar output pixel around the ring. H1 remains strict-hexagonal with high full-field correlation and
stable ridge width, but is `morphology_non_equivalent` because its dominant radial feature-radius
difference exceeds the predeclared 10% gate. The scalar route has no modelled focal `Ez`; it is never
treated as a numerical zero prediction.

V1 winding is scalar `1.000000` and vector transverse
`1.000000` for requested charge 1. V3 winding is scalar
`3.000000` and vector transverse
`3.000000` for requested charge 3.

## H1 Finding

H1 remains strictly hexagonal under both objective models: scalar
`True`, vector `True`. The Debye
longitudinal fraction is `10.150%`. Global
morphology correlation is `0.996356`; dominant radial
feature radius changes by `13.33%`, ridge width
by `0.00%`, and edge sharpness by
`18.76%`; transition width changes by
`0.00%`. C6 changes from
`0.961055` to `0.963375` and C3 from
`0.946132` to `0.953852`. The source-scale strict-hexagon claim
remains valid, but the scalar focal-detail approximation is materially different under the
predeclared feature-radius gate and must be replaced or explicitly narrowed for quantitative use.

## Interface Benchmark

The scalar field comparator uses the uncoated normal-incidence Fresnel coefficient for `n1=1.0` and
the project Cr:ZnSe placeholder `n2=2.44`. The separate Phase 2A surface-ledger factor `0.96` is
reported but not applied to either benchmark field. Fresnel transmission and the declared material
propagation are evaluated on the full-FOV native Debye plane (`N=1024`
for H1); the finite central comparison crop is never used as the spectral-interface input. Complex
fields are evaluated on the matched local objective coordinates by band-limited Fourier resampling
after the full-plane physics step.

| case | scalar T | vector T | interface corr | material-plane corr | k.E residual |
|---|---:|---:|---:|---:|---:|
| `G0` | 0.824770 | 0.824596 | 0.999548 | 0.999980 | 9.736e-17 |
| `B0` | 0.824770 | 0.824596 | 0.994214 | 0.999148 | 7.842e-17 |
| `V1` | 0.824770 | 0.824596 | 0.994206 | 0.999076 | 1.103e-16 |
| `V3` | 0.824770 | 0.824590 | 0.995689 | 0.999283 | 8.983e-17 |
| `H1` | 0.824770 | 0.824590 | 0.997463 | 0.999896 | 9.150e-17 |

For H1, the immediate post-interface strict class is
`visual_hexagonal_field` and the local transverse-polarisation fidelity is
`0.995839`. The material-plane
comparison is made after the same declared `10.0 um`
vector-ASM propagation, never against an unmatched plane.

## Claim Governance

The scalar objective remains an acceptable global morphology approximation for G0/B0. V1/V3
peak-location claims are narrowed to the vector reference. H1 remains a valid strict-hexagon result,
but scalar focal-detail claims are `approximation_materially_different` and require the vector
reference. The scalar normal-incidence interface approximation remains acceptable for morphology and
power at this bounded optical benchmark. Absolute objective transmission, the material index/identity,
relay/sample scale, coating state and focal energy remain calibration-required.

## Outputs

The H1 3D surfaces use x8 local complex-field
band-limited Fourier synthesis for render sampling only. This reduces the displayed focal-plane spacing from
`0.1029 um` to
`0.0129 um`.
Every native sample is preserved, no resampled array is used for a metric, and the interactive hover
readout reports linear `I/Imax` even in shape-emphasis mode. Linear parity is the default: it uses the
same Matplotlib `magma` colour definition as the 2D panels with flat ambient lighting. The interactive
file opens as a full-size heatmap of the same high-density array for exact top-down parity; perspective
oblique 3D remains available as the alternate view. Neither view contributes to benchmark metrics.

- `outputs/validation/phase2c/`
- `outputs/figures/phase2c/`
- `outputs/figures/phase2c/h1_3d_intensity_surfaces/h1_vector_debye_interactive.html`
- `docs/92_phase2c_vectorial_objective_and_interface.md`
