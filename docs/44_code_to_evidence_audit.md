# Stage 9A.3 Verified Methods Evidence and Open Validation Layers

Stage 9A.3 integrates the supplied verified seed bibliography into the Stage
9A.2 claim registry. It is documentation and evidence work only: no physical 4F
propagation, camera model, inverse correction, neural network, material-response
physics, or active CSLM/axicon physics has been added.

## Boundary

```text
fourier_filter_physics_available = False
camera_model_enabled = False
material_model_enabled = False
diagnostic_only = True
final_export_allowed = False
```

Literature support for a principle is not the same thing as validation of this
numerical implementation, calibration of this bench, or demonstration of a
fused-silica process outcome.

## Verified Seed References Linked To Claims

- `angular_spectrum_or_bl_asm_propagation` -> matsushima2009bandlimited
- `finite_sampling_and_aliasing_control` -> matsushima2009bandlimited
- `phase_only_slm_mask_generation` -> engstrom2013slmcalibration
- `phase_quantisation_and_grayscale_export` -> engstrom2013slmcalibration
- `command_domain_carrier_grating` -> zhang2009zeroorder
- `pixelated_slm_zero_order_and_unwanted_orders` -> zhang2009zeroorder
- `multi_plane_phase_retrieval_future` -> miao2022besselretrieval
- `effective_aberration_inference_future` -> neil2000closedloop
- `zernike_or_phase_conjugate_correction_future` -> neil2000closedloop, lopezquesada2009slmcorrection
- `fused_silica_bessel_channel_or_tgv_future` -> bhuyan2010microchannels
- `fused_silica_welding_future` -> zhang2018besselwelding

The canonical bibliography is `references/structured_beam_methods.bib`. It was
copied from `references/incoming/structured_beam_methods_verified_seed.bib`
without supplementing it from memory.

## Multi-Layer Claim Register

Full structured records are stored in
`configs/evidence/project_claim_registry.json`.

| claim ID | status | verified literature | manufacturer status | bench status | currently claimable | not claimable |
|---|---|---|---|---|---|---|
| `angular_spectrum_or_bl_asm_propagation` | implemented_active | matsushima2009bandlimited | not_required | not_bench_validated | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `finite_sampling_and_aliasing_control` | implemented_active | matsushima2009bandlimited | needs_manufacturer_data | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `phase_only_slm_mask_generation` | implemented_command_phase_model | engstrom2013slmcalibration | needs_manufacturer_data | needs_bench_measurement | generate mathematically wrapped phase command maps for SLM1/SLM2 with principle-level SLM calibration literature attached | claim calibrated physical phase response at 1030 nm, order efficiency, or local SLM non-uniformity correction |
| `phase_quantisation_and_grayscale_export` | implemented_command_export_model | engstrom2013slmcalibration | needs_manufacturer_data | needs_bench_measurement | export wrapped/quantised command maps while declaring physical phase response unresolved | claim grayscale commands produce calibrated physical phase without manufacturer and bench evidence |
| `command_domain_carrier_grating` | implemented_command_carrier_model | zhang2009zeroorder | needs_manufacturer_data | needs_bench_measurement | generate command-domain carrier sweeps and cite the diffraction-order principle | claim local carrier-to-Fourier-plane scaling, stop placement, selected-order purity, or direct Fourier-plane coordinates from downstream images |
| `pixelated_slm_zero_order_and_unwanted_orders` | planned_future | zhang2009zeroorder | needs_manufacturer_data | needs_bench_measurement | cite zero-order/unwanted-order behaviour as a future 4F/order-selection concern | claim an active validated order-efficiency or physical 4F model |
| `physical_fourier_filtering_future_route` | not_implemented | none | needs_manufacturer_data | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `ideal_selected_order_handoff` | implemented_diagnostic_only | none | not_required | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `slm1_to_slm2_propagation` | implemented_active | none | not_required | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `slm_registration_and_coordinate_frames` | implemented_diagnostic_only | none | not_required | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `physical_axicon_bessel_conversion` | implemented_benchmark_only | none | needs_manufacturer_data | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `physical_axicon_aperture_and_decentre` | implemented_benchmark_only | none | needs_manufacturer_data | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `vortex_phase_generation` | implemented_active | none | not_required | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `energy_and_fluence_accounting` | implemented_active | none | not_required | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `ring_centre_radius_dark_core_and_uniformity_metrics` | implemented_active | none | not_required | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `camera_z_stack_acquisition` | implemented_diagnostic_only | none | not_required | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `camera_coordinate_calibration` | implemented_diagnostic_only | none | needs_manufacturer_data | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `multi_plane_phase_retrieval_future` | planned_future | miao2022besselretrieval | not_required | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `effective_aberration_inference_future` | planned_future | neil2000closedloop | not_required | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `zernike_or_phase_conjugate_correction_future` | planned_future | neil2000closedloop, lopezquesada2009slmcorrection | needs_manufacturer_data | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `neural_fast_estimator_future` | not_implemented | none | not_required | needs_bench_measurement | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `fused_silica_application_boundary` | superseded_by_specific_application_claims | none | not_required | not_bench_validated | state only that the broad fused-silica boundary has been superseded by specific future application claims | claim any fused-silica TGV/channel, waveguide, welding, or modification outcome |
| `legacy_crznse_material_proxy_branch` | implemented_legacy | none | not_required | not_bench_validated | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `vector_beam_branch` | implemented_legacy | none | not_required | not_bench_validated | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `hexagonal_polygonal_discrete_beam_branch` | implemented_legacy | none | not_required | not_bench_validated | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `capsule_or_weld_feature_geometry_branch` | implemented_legacy | none | not_required | not_bench_validated | only the implemented diagnostic or method-boundary statement described by current_status and evidence layers | local bench validation, manufacturer calibration, or material/process outcome unless separately evidenced |
| `fused_silica_bessel_channel_or_tgv_future` | planned_future | bhuyan2010microchannels | not_required | needs_bench_measurement | cite Bessel microchannel fabrication as a principle-level future application direction | claim TGV/channel formation in this apparatus or any fused-silica process window |
| `fused_silica_waveguide_future` | planned_future | none | not_required | needs_bench_measurement | state only that a targeted fused-silica waveguide source search is still required | claim waveguide writing, index change, loss, or mode quality |
| `fused_silica_welding_future` | planned_future | zhang2018besselwelding | not_required | needs_bench_measurement | cite Bessel welding as a principle-level future application direction | claim weld strength, interface quality, symmetry, or local process validity |

## Fused-Silica Application Split

The broad `fused_silica_application_boundary` record is retained only as a
backwards-compatible superseded boundary. Specific future branches now carry
their own evidence needs.

This branch contains CrZnSe-specific proxy assumptions and is not validated for
fused-silica TGV, waveguide, welding, or modification predictions. The legacy
Cr:ZnSe branch remains quarantined from fused-silica decisions.

| claim ID | literature status | readiness | boundary |
|---|---|---|---|
| `fused_silica_bessel_channel_or_tgv_future` | verified_principle_only | not_claimable | Bhuyan 2010 supports Bessel microchannel principle only; it does not validate this bench or process. |
| `fused_silica_waveguide_future` | targeted_search_required | not_claimable | No verified direct waveguide source is integrated; this branch remains unresolved. |
| `fused_silica_welding_future` | verified_principle_only | not_claimable | Zhang 2018 supports Bessel welding principle only; it does not validate this apparatus or interface. |

## Deliberately Unresolved Claims

These remain unresolved because the seed bibliography does not validate them and
they require targeted literature, manufacturer specifications, bench data, or a
legacy/quarantine label.

| claim ID | literature | manufacturer | bench | next action |
|---|---|---|---|---|
| `physical_fourier_filtering_future_route` | targeted_search_required | needs_manufacturer_data | needs_bench_measurement | Attach manufacturer specification and then run the required bench measurement. |
| `physical_axicon_bessel_conversion` | targeted_search_required | needs_manufacturer_data | needs_bench_measurement | Attach manufacturer specification and then run the required bench measurement. |
| `physical_axicon_aperture_and_decentre` | targeted_search_required | needs_manufacturer_data | needs_bench_measurement | Attach manufacturer specification and then run the required bench measurement. |
| `camera_z_stack_acquisition` | not_applicable | not_required | needs_bench_measurement | Acquire the required bench measurement before narrowing the claim. |
| `camera_coordinate_calibration` | not_applicable | needs_manufacturer_data | needs_bench_measurement | Attach manufacturer specification and then run the required bench measurement. |
| `neural_fast_estimator_future` | targeted_search_required | not_required | needs_bench_measurement | Acquire the required bench measurement before narrowing the claim. |
| `legacy_crznse_material_proxy_branch` | targeted_search_required | not_required | not_bench_validated | Complete targeted source search without filling gaps from generic papers. |
| `fused_silica_waveguide_future` | targeted_search_required | not_required | needs_bench_measurement | Acquire the required bench measurement before narrowing the claim. |

## Evidence Registers

- Literature search plan: `configs/evidence/literature_search_plan.json`
  (19 entries)
- Manufacturer evidence register:
  `configs/evidence/manufacturer_evidence_register.json`
- Bench evidence register:
  `configs/evidence/bench_evidence_register.json`

## Immediate Lab Action

Run the downstream carrier-stop characterisation session from Stage 9A.1B:
record actual SLM/camera/lens/stop/axicon identifiers, capture dark and flat
references, then measure downstream response versus SLM2 command-domain carrier
cycles and Fourier-stop settings without moving the fixed downstream route.
