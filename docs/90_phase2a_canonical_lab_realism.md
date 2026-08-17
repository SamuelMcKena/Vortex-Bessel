# PHASE 2A Canonical Lab-Realism and Power-Ledger Unification

**Outcome:** `PHASE2A-B`.

PHASE 2A introduces no new beam family, broadband model, nonlinear material response, or recovery of the 116 blocked historical configurations. It binds the controlled laboratory routes to one hardware/provenance manifest and one energy-accounting contract.

## Canonical Contract

- Hardware ID: `PHAROS_DUAL_PLUTO_FIXED_BENCH_V1`
- Mapping mode: `fixed_physical_optics`
- SLM fill-factor model: `throughput_only`
- Wavelength: 1029 nm
- Beam radius at SLM: 2 mm (1/e field amplitude; calibration required)
- Panels: two HOLOEYE PLUTO-2.1 NIR-149, 1920 x 1080, 8 um pitch, 8-bit, 93% fill factor
- Carrier: 6.25 lp/mm; nominal 4F focal length: 300 mm; derived iris radius: 0.77175 mm
- Objective: NA 0.45, effective focal length 4 mm (inherited assumptions; calibration required)
- Axicon: 2 degree base angle, n=1.458 (source-model assumptions; clear aperture unresolved)

The fixed mapping is stable in software, but relay magnification is not bench-calibrated. Source-scale dimensions therefore remain optical predictions rather than absolute sample-plane claims.

## SLM Models

- `throughput_only`: `Eout = sqrt(FF) exp(i phi) Ein`.
- `resolved_pixel_aperture`: `Eout = M exp(i phi) Ein` on a grid with at least two samples per pixel.
- `coherent_unmodulated_deadspace`: `Eout = [M exp(i phi) + (1-M)] Ein`.

The canonical 10 mm validation grid cannot resolve 8 um pixel borders, so it uses `throughput_only`. The vector route's previous coherent model remains available under its explicit name; no established Nathan output was silently rebound.

## Energy and Fluence

The ledger follows laser, input aperture, SLM1, SLM2, simulated first-order selection, relay, objective pupil, surface, and sample. The calculated first-order selected fraction is the only first-order factor. No configured diffraction efficiency is multiplied again.

`F(x,y) = E_plane I(x,y) / integral(I dx dy)`

All 25 ledgers close to a maximum relative residual of 1.694e-16. Absolute fluence remains calibration-required because pulse energy and component transmissions are not measured in this repository.

## Canonical Family

| case | realistic morphology | first-order efficiency | power drift | vector purity | quantitative |
|---|---|---:|---:|---:|---:|
| `G0` | gaussian_calibration_field | 0.999951 | 5.829e-16 | n/a | true |
| `B0` | bright_core_bessel | 0.999951 | 5.829e-16 | n/a | true |
| `V1` | vortex_bessel_ring | 0.997861 | 7.797e-16 | n/a | true |
| `V3` | vortex_bessel_ring | 0.981670 | 3.973e-16 | n/a | true |
| `H1` | visual_hexagonal_field | 0.949354 | 1.131e-09 | 0.9746 | true |

Each family also includes analytic/target, ideal optical, mild-error, and deliberately degraded controls. Mild/degraded rows are diagnostics even when their numerical power gate passes.

## Claim Boundary

- `P2A-C1` `validated_with_scope`: One canonical 1029 nm dual-PLUTO hardware binding governs PHASE 2A lab runs.
- `P2A-C2` `validated`: SLM fill factor has three explicit, non-interchangeable physical models.
- `P2A-C3` `validated`: Every canonical energy ledger closes by sequential multiplication.
- `P2A-C4` `validated`: Calculated first-order selection feeds the ledger without a second configured efficiency.
- `P2A-C5` `validated`: G0, B0, V1, V3 and H1 all pass the 5% numerical propagation-power gate in every controlled route.
- `P2A-C6` `validated_with_scope`: The realistic H1 route retains the source-scale visual hexagon and local vector purity.
- `P2A-C7` `validated_with_scope`: Mild and deliberately degraded controls are error-response diagnostics.
- `P2A-C8` `validated_with_scope`: The model-plane fluence map integrates to the final ledger energy.
- `P2A-C9` `calibration_required`: Absolute sample-plane fluence is bench-calibrated.
- `P2A-C10` `calibration_required`: Reported source-scale beam dimensions are absolute sample-plane dimensions.
- `P2A-C11` `calibration_required`: The linear 8-bit phase model is the calibrated NIR-149 LUT/stroke response.
- `P2A-C12` `validated`: Every audited perturbation declares its physical or diagnostic injection plane.

## Regression

- `phase2a_slm_energy_fluence`: 86 passed in 8.22s
- `phase1_phase1r_vortex`: 35 passed in 45.11s
- `nathan_mode2_16_modules`: 195 passed in 367.86s
- `active_test_collection`: 1053 tests collected in 9.65s

## Conclusion

Core fixed-bench, SLM, error-plane, energy and fluence contracts are unified and numerically validated. Absolute energy, sample-plane scale and SLM LUT/stroke claims remain calibration-limited.

Machine-readable outputs are under `outputs/validation/phase2a/`.
