# Phase 1 Fourier-Geometry Impact Audit

**Status:** formula repaired in source; inherited artifacts that contain the old physical distance
remain stale and are not authorised for quantitative reuse.

## Corrected Contract

For radial transverse wavevector `k_r` in rad/m,

```text
old: r_fourier = f k_r / (2 pi)
new: r_fourier = f k_r / k0 = lambda f k_r / (2 pi)
```

For carrier frequency `nu` in cycles/m,

```text
old: x_plus1 = f nu
new: x_plus1 = lambda f nu
```

The public helpers now require `wavelength_m`; no hidden global wavelength is used. A repository-wide
search found no additional live Python/notebook/doc expression equivalent to the two old formulas.
Angular-spectrum k-space calculations were not changed.

## Directly Affected Artifacts

| artifact | audience | impact | regeneration |
|---|---|---|---|
| `notebooks/lab_realism/04_objective_pupil_and_first_order_filtering.ipynb` | diagnostic source notebook | code cell repaired; saved output cells still show old values | required |
| `outputs/csv/stage_c/objective_pupil_geometry_summary.csv` | report-facing Stage C summary | `fourier_ring_radius_mm` is wrong by a factor `1/lambda` | required |
| `outputs/notebook_triage/04_objective_pupil_and_first_order_filtering_executed.ipynb` | diagnostic executed notebook | code and displayed values are stale | required |

At 1029 nm the two stale Stage C values change as follows:

| regime | old radius (mm) | corrected radius (mm) |
|---|---:|---:|
| `general` | 8238.245119 | 0.008477154 |
| `limits` | 37072.103037 | 0.038147194 |

The source notebook was not executed in this repair pass because accepted outputs must not be silently
regenerated.

## Confirmed Unaffected

- `outputs/figures/stage_c/nb04_carrier_cone_diagram.png` compares carrier and cone spatial
  frequencies in lp/mm and does not consume the physical-distance helper.
- Propagated scalar fields, beam radii, Bessel-zone metrics and material proxies do not consume this
  reporting helper; their arrays are not numerically changed by the wavelength-factor repair.
- The Nathan 4F branch already uses `x=lambda f nu`. Its canonical 1029 nm, 300 mm, 6.25 lp/mm
  displacement remains `1.929375 mm`.
- No Nathan V0/source-scale field generation calls `ObjectiveMap` or these older Stage C distance
  helpers.

## Conclusion

The Stage C physical Fourier-plane ring-distance claim is blocked until the notebook and CSV are
regenerated. This correction does not overturn a canonical Nathan conclusion and does not alter the
underlying scalar propagation arrays.
