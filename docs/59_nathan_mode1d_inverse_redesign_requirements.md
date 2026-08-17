# Nathan MODE 1D - Inverse Redesign Requirements

**Status:** MODE 1D inverse redesign audit only. This does not simulate HWP/QWP/SLM panel realism,
does not approve MODE 2A/2B physical realisation, and does not replace the need for a redesigned
MODE 1 downstream simulation.

## Question

What optical numbers would be required for the downstream system to produce a micro-scale version of
Nathan's validated hexagonal Bessel target?

MODE 1D uses the MODE 1C mapping:

```text
N_rings = R_P2 k_r,pre / (2 pi)
k_r,surface = M_k k_r,pre
NA_required = k_r,surface / k0
```

with `R_P2 = 1995.6 um`, SLM-safe radius `3888.0 um`, mapping factor `M_k = 123.899`,
current objective NA `0.45`, and V0 ring count `31.06`.

## V0 Requirement

| radius case | radius (um) | required NA for V0 ring count | comment |
|---|---:|---:|---|
| current P2 radius | 1995.6 | 1.984 | not realistic for this architecture |
| 0.75 SLM-safe radius | 2916.0 | 1.358 | not realistic for this architecture |
| SLM-safe radius | 3888.0 | 1.018 | immersion-class or exploratory NA |
| full half-short-axis SLM radius | 4320.0 | 0.917 | aggressive high-NA air redesign |

At the current P2 radius, reproducing the V0 ring count would require roughly `k_r,pre =
9.78e4 m^-1`, `k_r,surface = 1.21e7 m^-1`, and NA `1.98`.

## Radius And Mapping Requirements

At current NA `0.45`, the required P2 radius for the V0 ring count is `8798.6 um`, or a diameter of
`17597.2 um`, which does not fit the SLM short axis. At NA `0.9`, the required radius is
`4399.3 um`, just beyond the safe short-axis radius and slightly beyond the half-short-axis radius.

The mapping factor would also need to drop substantially:

- Current P2 radius, NA 0.45: required mapping factor `28.10`, reduction factor `4.41`.
- SLM-safe radius, NA 0.45: required mapping factor `54.75`, reduction factor `2.26`.
- SLM-safe radius, NA 0.9: required mapping factor `109.50`, reduction factor `1.13`.

## Achievable Ring Budget

| case | ring count | fraction of V0 | likely regime |
|---|---:|---:|---|
| current NA/current radius | 7.04 | 0.227 | low_ring_triangular |
| current NA/SLM-safe radius | 13.72 | 0.442 | low_ring_triangular |
| NA 0.7/SLM-safe radius | 21.35 | 0.687 | intermediate_uncertain |
| NA 0.9/SLM-safe radius | 27.45 | 0.884 | V0_like_possible |
| NA 1.0/SLM-safe radius | 30.50 | 0.982 | V0_like_possible |
| NA 1.2/SLM-safe radius | 36.60 | 1.178 | V0_like_possible |

The inherited actual system remains below the lower-ring threshold: actual current P2 ring count is
`4.11`, and the current-radius NA-limited maximum is `7.04`.

## Lower-Ring Source Sweep

The source-style sweep varied the validated Nathan source model to target ring counts:

```text
4, 6, 8, 10, 12, 14, 16, 20, 24, 31
```

Acceptance required a visual hexagonal classification, a dark core, angular similarity to the V0
template, and scaled V0-template `xy` correlation. The first accepted lower-ring target was:

```text
minimum accepted source ring count = 12
```

`N = 8` classified as `visual_hexagonal_field`, but failed the scaled V0-template `xy` correlation,
so it was not accepted as a robust Nathan-style target. `N = 12` passed with `xy` correlation
`0.665`.

## Outputs

Generated in `outputs/figures/digital_twin/nathan_mode1d_inverse_redesign/`:

- `mode1d_required_na_table.csv/json`
- `mode1d_required_radius_table.csv/json`
- `mode1d_required_mapping_table.csv/json`
- `mode1d_achievable_ring_count_table.csv/json`
- `mode1d_source_ring_count_sweep.csv/json`
- `mode1d_source_ring_count_sweep_summary.png`
- `mode1d_redesign_budget_plot.png`
- `mode1d_outcome_report.json`
- `simulation_scope_manifest.json`

## Outcome

**M1D-A.** A realistic redesign appears possible within available SLM aperture if the P2 beam is
expanded toward the SLM-safe radius and/or the downstream design is reworked under a plausible
high-NA budget. The minimum lower-ring source target is `N = 12`, and the SLM-safe/current-NA budget
can reach `13.72`.

This does **not** open MODE 2A/2B. The next required step is a redesigned MODE 1 downstream
simulation using P2 radius/mapping/NA values that can reach the `N >= 12` regime and confirming that
the sample-plane field actually remains Nathan-style hexagonal.
