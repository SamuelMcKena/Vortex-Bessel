# Nathan MODE 1C - k_r / Aperture / NA Feasibility

**Status:** MODE 1C audit only. No patterned HWP, QWP, SLM serial route, 4F carrier/iris, panel
realism, HWP mosaics, or MODE 2A/2B physical realisation is authorised here.

## Question

Given the actual inherited architecture constraints - P2 beam size, SLM aperture, relay/objective
mapping, objective NA, and inherited radial wavevector mapping - can the system reach a ring-count and
symmetry regime capable of reproducing a micro-scale version of the validated V0 hexagonal Bessel
field?

## k_r Mapping

| quantity | value |
|---|---:|
| wavelength | 1.029 um |
| current pre-axicon `k_r` | 12,940.6 m^-1 |
| current surface/sample `k_r` | 1,603,333.3 m^-1 |
| surface/pre mapping factor | 123.90 |
| objective NA | 0.45 |
| objective surface `k_r` limit | 2,747,748.7 m^-1 |
| NA-limited pre-axicon `k_r` | 22,177.3 m^-1 |
| current surface NA fraction | 0.584 |
| current pre-axicon phase period | 485.54 um |
| current surface radial period | 3.92 um |

Important model boundary:

- `physical_axicon.axicon_base_angle_deg` cleanly overrides the pre-axicon phase angle.
- surface `k_r` is target/design-derived through `TwinConfig.target` and ObjectiveMap.
- changing a high pre-axicon proxy is not automatically a valid current-architecture surface-k_r
  candidate.

## Aperture Ring Limit

| budget item | ring count |
|---|---:|
| current P2 radius + current `k_r_pre` | 4.11 |
| current P2 radius + NA-limited `k_r_pre` | 7.04 |
| SLM-safe radius + current `k_r_pre` | 8.00 |
| SLM-safe radius + NA-limited `k_r_pre` | 13.72 |
| V0 source target | 31.06 |

Radii:

- current measured P2 radius: 1995.6 um
- SLM-safe radius: 3888.0 um

The maximum current-architecture budget is about `13.72 / 31.06 = 0.44` of the V0 ring count. Under
the audit rule, this is below half of V0 and is not a V0-like ring-count regime.

## Constrained Search

The constrained search sampled target ring counts:

```text
4, 6, 8, 10, 12, 16, 20, 24, 31
```

at three P2 radii:

- current measured P2 radius;
- 0.75 x SLM-safe radius;
- SLM-safe radius.

Results:

- 27 constrained candidates evaluated.
- 11 were within objective NA and SLM aperture and were run as plane-corrected free-space proxies.
- 16 exceeded constraints and were not propagated.
- 0 proxy candidates passed the V0-template hexagon gate.
- 0 actual inherited downstream candidates passed.

Actual target-k_r override note:

The code can express a target-derived surface `k_r` by changing the legitimate
`TwinConfig.target.target_core_diameter_m` design field, but the current MODE 1 downstream path contains
a locked surface-k_r fingerprint for the inherited design. Therefore MODE 1C records proxy candidates
clearly and does not pretend they are actual inherited downstream confirmations.

## Outcome

**M1C-C.** Current architecture cannot reach the required V0-like ring-count regime because of the
combined objective NA, SLM aperture, and inherited k_r mapping budget.

MODE 2A/2B remains blocked. Optical redesign is required before physical route work can be justified.

## Outputs

Generated in:

```text
outputs/figures/digital_twin/nathan_mode1c_kr_aperture/
```

Files:

- `mode1c_kr_mapping.json`
- `mode1c_aperture_ring_limit.json`
- `mode1c_constrained_search.csv`
- `mode1c_constrained_search.json`
- `mode1c_feasibility_plot.png`
- `mode1c_outcome_report.json`
