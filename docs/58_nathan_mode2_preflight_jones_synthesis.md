# Nathan MODE 2P - Jones-Chain Synthesis Preflight

**Status:** MODE 2-preflight only. This is a component-level Jones-chain equivalence proof at
P2. No downstream propagation, no carrier/iris/panel realism, and no MODE 2A/2B physical-route
approval are authorised here.

## Question

Can the ideal upstream component equations synthesize Nathan's validated six-sector P2 input field?

```text
E_target(x,y) = A(r) [cos(alpha(x,y)), sin(alpha(x,y))]^T
alpha(theta) = theta + phi0(theta)
phi0 = 0 in radial sectors, pi/2 in azimuthal sectors
```

The sector convention is kept aligned with V0: each 120 degree cell starts with an azimuthal sector
and ends with a 60 degree radial sector.

## Routes Tested

| route | selected convention | overlap | phase-aligned RMS | Stokes RMS | result |
|---|---|---:|---:|---:|---|
| ideal patterned HWP | `beta = alpha/2`, horizontal input | 1.000000 | 0.000e+00 | 0.000e+00 | pass |
| circular dual-SLM identity | `E_R ~ exp(+i alpha)`, `E_L ~ exp(-i alpha)` | 1.000000 | 1.766e-16 | 5.072e-16 | pass |
| ideal dual-SLM linear + QWP | `H: +alpha`, `V: -alpha + pi/2`, `QWP: -pi/4` | 1.000000 | 2.127e-16 | 5.701e-16 | pass |

The circular-channel normalization is `A/sqrt(2)` per circular component, so the target power is
preserved rather than recovered by amplitude scaling.

## Centre Policy

The default P2 handoff grid straddles the optical axis, so the centre diagnostic reports four
nearest-centre pixels rather than a single exact-axis sample. No route-specific centre editing is
applied. Including and excluding those pixels both keep overlap at 1.000000 for the accepted routes.
The axis-sampled source-grid test also passes, so the centre convention does not create the pass.

## Outputs

Generated in `outputs/figures/digital_twin/nathan_mode2_preflight_jones/`:

- `target_alpha_and_sector_map.png`
- `route_patterned_hwp_ideal_vs_target.png`
- `route_dual_slm_qwp_ideal_vs_target.png`
- `jones_synthesis_summary.json`
- `jones_synthesis_summary.csv`
- `simulation_scope_manifest.json`

## Outcome

**M2P-A.** Both ideal patterned-HWP and ideal dual-SLM/QWP routes reproduce Nathan's P2 target field
with complex vector overlap >= 0.999.

This validates the upstream synthesis mathematics only. It does **not** unblock full physical
realisation: MODE 1C remains `M1C-C`, so the current downstream ring-count/NA/aperture constraints
still block MODE 2A/2B.

## Next Action

If MODE 2 is revisited, the next authorised work should remain explicitly scoped: either component
realism at P2, or an optical redesign study that first resolves the MODE 1C ring-count/NA/aperture
block. Do not treat this preflight proof as a sample-plane hexagon claim.
