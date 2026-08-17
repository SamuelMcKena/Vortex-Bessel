# Metric Conventions Glossary

Single source of truth for reported scalar-core metrics. Each entry gives the
meaning, units, plane/medium, and implementing function.

## Vortex Ring Radius

| | |
|---|---|
| Symbol | `ring_radius_m`, `vortex_main_ring_radius_m` |
| Formula | `jnp_zeros(abs(ell), 1)[0] / k_r`, the first zero of `J'_ell` |
| Units | metres in code, micrometres in tables |
| Plane / medium | transverse, sample medium |
| Implementing function | `bessel_twin_core.compute_design_from_targets`; measured by `vbb_study.vbb_metrics.peak_plane_radial_metrics` |
| Notes | Defined for `abs(ell) > 0`. The vortex center is dark, so the bright feature is the annular ring, not a central core. |

## J0 First-Zero Radius

| | |
|---|---|
| Symbol | `core_first_zero_radius_m`, `equivalent_l0_first_zero_radius_m` |
| Formula | `j_{0,1} / k_r`, where `j_{0,1}=2.404825557...` is the first zero of `J0` |
| Units | metres in code, micrometres in tables |
| Plane / medium | transverse design scale, sample medium |
| Implementing function | `bessel_twin_core.compute_design_from_targets`; radial metrics also report `core_first_zero_radius_m` |
| Notes | For `ell = 0`, the central maximum is at `r = 0`; this radius is the first dark ring. For `ell > 0`, it is only an equivalent l0 scale and is not the vortex ring radius. |

## HWHM Core Radius

| | |
|---|---|
| Symbol | `core_hwhm_radius_m`, `core_hwhm_diameter_m` |
| Formula | Outer radius where the `ell = 0` azimuthal-average intensity falls to 50% of the central maximum |
| Units | metres in code, micrometres in tables |
| Plane / medium | transverse, sample medium |
| Implementing function | `vbb_study.vbb_metrics.peak_plane_radial_metrics` |
| Notes | This is the measured bright-core size for `ell = 0`. Legacy `core_radius_m` remains for compatibility and is disambiguated by `core_radius_definition`; new code should prefer `core_hwhm_radius_m` or `core_first_zero_radius_m`. |

## Feature Radius And Diameter

| | |
|---|---|
| Symbol | `feature_radius_m`, `feature_diameter_m` |
| Formula | `core_hwhm_radius_m` for `ell = 0`; `ring_radius_m` for `abs(ell) > 0` |
| Units | metres in code, micrometres in tables |
| Plane / medium | transverse, sample medium |
| Implementing function | `vbb_study.vbb_metrics.radial_feature_metrics` |
| Notes | Use this when a plot or CSV needs one shape-size column that works for both central-core and vortex-ring cases. |

## Equivalent l0 Target Diameter

| | |
|---|---|
| Symbol | `target_core_diameter_m`, `target_equivalent_l0_core_diameter_m` |
| Formula | `target_core_diameter_m = 2 * j_{0,1} / k_r` using the exact SciPy Bessel root |
| Units | metres in code, micrometres in tables |
| Plane / medium | inverse-design scale, sample medium |
| Implementing function | `bessel_twin_core.compute_design_from_targets` |
| Notes | Warning: `target_core_diameter_m` currently means `target_scale_definition = "equivalent_l0_first_zero_diameter"`. For vortex beams, the actual bright ring diameter is `vortex_main_ring_diameter_m`, not `target_core_diameter_m`. |

## Ring Width

| | |
|---|---|
| Symbol | `ring_width_m` |
| Formula | Radial half-width at half maximum of the azimuthal-average profile around the vortex ring: `r_half_outer_m - r_half_inner_m` |
| Units | metres in code, micrometres in tables |
| Plane / medium | transverse, sample medium |
| Implementing function | `vbb_study.vbb_metrics.peak_plane_radial_metrics` |

## Canonical Bessel Zone

| | |
|---|---|
| Symbol | `bessel_zone_um`, also exported as `canonical_zone_um` |
| Formula | FWHM of the axial peak-intensity trace: z range where `max_xy I(z) >= 0.5 * max_z(max_xy I)` |
| Units | micrometres |
| Plane / medium | axial, same medium as the volume scan |
| Implementing function | `bessel_twin_core.bessel_zone_metrics(z, peak, level=0.5)` |
| Notes | This is the canonical single-observable zone metric. It is shorter than the geometric `z_max` for finite Gaussian apertures. |

## Strict Bessel Region

| | |
|---|---|
| Symbol | `strict_bessel_region_um`, also exported as `bessel_region_um` |
| Formula | Contiguous intersection of axial-peak threshold, fixed-bucket feature-power threshold, and ring/core radius-stability threshold |
| Units | micrometres |
| Plane / medium | axial, same medium as the volume scan |
| Implementing function | `bessel_twin_core.bessel_region_metrics` |
| Notes | This is the conservative useful region for fabrication-planning reports. It must not be silently replaced by the canonical FWHM zone. |

## Side-To-Core Peak Ratio

| | |
|---|---|
| Symbol | `side_to_core_peak_ratio` |
| Formula | Brightest excluded side-lobe peak divided by the bright feature peak |
| Units | dimensionless |
| Plane / medium | transverse, peak intensity plane |
| Implementing function | `bessel_twin_core.fluence_metrics` |
| Notes | For vortices, the bright feature is the annular HWHM bucket. For `ell = 0`, it is the HWHM core disk. Values above 0.5 indicate significant side-lobe contamination; values above 0.8 are marginal. |

## Fluence

| | |
|---|---|
| Symbol | `F` |
| Formula | `F = E_pulse * I(x, y) / integral(I dA)`, converted from J/m^2 to J/cm^2 |
| Units | J/cm^2 |
| Plane / medium | one transverse XY plane |
| Implementing function | `bessel_twin_core.fluence_from_intensity` or `vbb_study.vbb_materials.fluence_from_intensity` |
| Notes | Energy-conserving on a single plane. Centerline XZ fluence plots are planning proxies, not calibrated material-response predictions. |

## Incubated Threshold

| | |
|---|---|
| Symbol | `F_th,N` |
| Formula | `F_th,N = F_th,1 * N_eff ** (S - 1)` |
| Units | J/cm^2 |
| Plane / medium | material proxy |
| Implementing function | `vbb_study.vbb_materials_study.incubated_threshold`; `vbb_study.vbb_materials.incubated_threshold_J_cm2` |
| Notes | Threshold values are planning proxies until calibrated by experiment. |

## Sampling Validity

| | |
|---|---|
| Symbol | `sampling_valid`, `qa_status`, `phase_sampling_label` |
| Criteria | Feature size, radial period, axial sampling, and SLM/propagation Nyquist checks |
| Units | dimensionless counts or labels |
| Implementing function | `bessel_twin_core.sampling_report`; `vbb_study.vbb_regime.sampling_validity` |
| Notes | Marginal sampling is diagnostic; failed sampling should not be treated as a publication-quality result. |

## First-Order Geometry Validity

| | |
|---|---|
| Symbol | `first_order_geometry_valid` |
| Formula | `cone_lpmm + filter_lpmm < carrier_lpmm` with a small bin margin |
| Units | lp/mm |
| Implementing function | `bessel_twin_core.first_order_filter_geometry` |
| Notes | If false, holographic diffraction orders overlap. The physical axicon route is not constrained by this first-order filtering geometry. |

## Energy Budget Symbols

| symbol | key | units | formula |
|---|---|---|---|
| `E_in` | `pulse_energy_in_J` | uJ | laser output per pulse |
| `E_surface_air` | `pulse_energy_at_surface_air_J` | uJ | `E_in` times pre-surface transmissions |
| `E_sample` | `pulse_energy_at_sample_J` | uJ | `E_surface_air` times surface transmission |
| `T_total` | `total_transmission` | dimensionless | product of modeled transmissions |
