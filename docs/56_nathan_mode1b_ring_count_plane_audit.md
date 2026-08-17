# Nathan MODE 1B - Ring-Count Plane Audit

**Status:** MODE 1B audit only. No MODE 2A/2B physical HWP/QWP/SLM work is authorised here.

## Finding

The old inherited MODE 1B ring-count estimate mixed a **sample-plane radius** with the **pre-axicon**
transverse wavevector. That produced the tiny value `0.033`, but that value is not a physically
meaningful axicon/P2 ring count.

The old `M1B-B` conclusion is therefore not valid for the reason previously stated. The corrected
search has been regenerated with plane-labelled P2/axicon radii; it still finds no candidate passing
the full V0-template hexagon gate, but the interpretation is now based on plane-correct diagnostics.

## Plane Table

| plane_id | radius source | k_r source | radius (um) | ring count | valid for hexagon scaling |
|---|---|---|---:|---:|---|
| `v0_source_plane` | `NathanSourceParityConfig.beam_radius_m` | `NathanSourceParityConfig.k_r_m_inv` | 2000.0 | 31.06 | yes |
| `old_mode1b_mixed_sample_radius_pre_axicon_kr` | `design.w0_sample_m` | pre-axicon `k_r_pre_m_inv` | 16.14 | 0.033 | no |
| `p2_handoff_plane` | measured canonical P2 total intensity | pre-axicon `k_r_pre_m_inv` | 1995.6 | 4.11 | yes |
| `axicon_or_pupil_input_plane` | P2 handoff radius | pre-axicon `k_r_pre_m_inv` | 1995.6 | 4.11 | yes |
| `sample_plane_design_radius` | `design.w0_sample_m` | sample `k_r_surface_m_inv` | 16.14 | 4.12 | no for axicon-input scaling |

Old/new ratio:

```text
corrected_p2_ring_count / old_mixed_ring_count = 123.6
```

## Model Separation

MODE 1B now keeps these distinct:

- **V0 source model:** validated source-scale Nathan free-space target.
- **Actual inherited MODE 1:** ideal P2 field through the actual inherited downstream machinery;
  still visually triangular/lobed, so MODE 2A/2B remain blocked.
- **Plane-corrected free-space continuation:** simplified ideal sweep using P2/axicon-scale radii;
  useful for physics insight, but not proof of current-lab feasibility.

The free-space continuation cannot by itself prove current-lab feasibility.

## Corrected Search Interpretation

The corrected search uses P2-scale radii for tier 1 and P2-scale exploratory radii for tier 2. It
regenerated:

- `outputs/figures/digital_twin/nathan_mode1b_geometry_search/mode1b_ring_count_plane_audit.json`
- `outputs/figures/digital_twin/nathan_mode1b_geometry_search/mode1b_ring_count_plane_audit.csv`
- refreshed `mode1b_search_summary.*`, shortlist figures, outcome report, and manifest.

No corrected candidate passed the full hexagon gate. Some P2-scale free-space candidates can reach
`visual_hexagonal_field`, but they fail the V0-template/dark-core gate and/or are redesign-regime
candidates. That does not open MODE 2A/2B.

## RCA Outcome

**RCA-B.** The old MODE 1B ring-count plane usage was wrong. The previous `M1B-B` result is downgraded
as an explanation based on `0.033`. The corrected plane-aware search has been run and still does not
authorise physical realisation. Current-lab MODE 2A/2B remain blocked.

## Next Action

If this path is continued, the next study should compare the actual inherited downstream model and the
plane-corrected free-space continuation under the same P2-scale radius and axicon-angle semantics,
rather than using sample-plane radii as candidate input radii.
