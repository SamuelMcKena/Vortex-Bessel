"""Stage 9A.2 code-to-evidence audit data and artifact writer.

This module contains no optical propagation, fitting, camera model, material
model, correction engine, or AI.  It is a static evidence map for the current
repository state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


STAGE = "9A.2"
FINAL_EXPORT_ALLOWED = False

ACTIVE_ROUTE_COMPONENTS = (
    "source_field",
    "input_conditioning_boundary",
    "SLM1_phase_plane",
    "SLM1_to_SLM2_segment",
    "SLM2_phase_plane",
    "SLM2_to_pre_4F_diagnostic_segment",
    "post_SLM2_pre_4F_diagnostic_plane",
    "ideal_selected_order_handoff_plane",
    "physical_axicon_benchmark_plane",
    "post_axicon_benchmark_segment",
    "ideal_axicon_benchmark_reference_plane",
)


def _claim(
    claim_id: str,
    title: str,
    component: str,
    paths: list[str],
    route_relevance: str,
    claim_type: str,
    current_status: str,
    physics_status: str,
    evidence_category: str,
    evidence_status: str,
    *,
    refs: list[str] | None = None,
    manufacturer: list[str] | None = None,
    bench: list[str] | None = None,
    assumption: str = "",
    limitations: list[str] | None = None,
    not_prove: list[str] | None = None,
    validation: list[str] | None = None,
    readiness: str = "not_claimable",
    active_components: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "title": title,
        "project_component": component,
        "implementation_paths": paths,
        "route_relevance": route_relevance,
        "claim_type": claim_type,
        "current_status": current_status,
        "physics_status": physics_status,
        "evidence_category": evidence_category,
        "evidence_status": evidence_status,
        "literature_reference_keys": refs or [],
        "manufacturer_evidence_required": manufacturer or [],
        "bench_evidence_required": bench or [],
        "assumption_or_placeholder": assumption,
        "known_limitations": limitations or [],
        "what_this_does_not_prove": not_prove or [],
        "validation_required": validation or [],
        "publication_claim_readiness": readiness,
        "implemented_route_components": active_components or [],
        "final_export_allowed": FINAL_EXPORT_ALLOWED,
    }


CLAIMS: list[dict[str, Any]] = [
    _claim(
        "angular_spectrum_or_bl_asm_propagation",
        "Band-limited angular-spectrum free-space propagation",
        "scalar optical field engine",
        ["vbb_study/equations/propagation.py", "vbb_study/digital_twin/component_plane_pipeline.py"],
        "canonical_active",
        "numerical_method",
        "implemented_active",
        "numerically_modelled",
        "textbook_or_standard_reference",
        "needs_verified_source",
        refs=["search_bl_asm_sampling"],
        assumption="linear scalar free-space propagation, n=1.0 unless explicitly configured",
        limitations=["no nonlinear propagation", "finite grid/FOV validity must be checked per case"],
        not_prove=["bench alignment", "material modification", "camera agreement"],
        validation=["sampling/FOV convergence", "power drift checks", "bench z-stack comparison"],
        readiness="simulation_claim_possible",
        active_components=["SLM1_to_SLM2_segment", "SLM2_to_pre_4F_diagnostic_segment", "post_axicon_benchmark_segment"],
    ),
    _claim(
        "finite_sampling_and_aliasing_control",
        "Finite sampling, Nyquist, and aliasing guardrails",
        "sampling utilities and carrier-mask guard",
        ["vbb_study/equations/propagation.py", "vbb_study/digital_twin/slm_calibration_masks.py"],
        "canonical_active",
        "numerical_method",
        "implemented_active",
        "numerically_modelled",
        "textbook_or_standard_reference",
        "needs_verified_source",
        refs=["search_bl_asm_sampling"],
        manufacturer=["actual SLM display resolution and pixel pitch"],
        bench=["observed order locations versus command carrier"],
        limitations=["guardrails do not validate the real bench by themselves"],
        validation=["carrier sweep", "grid convergence"],
        readiness="simulation_claim_possible",
    ),
    _claim(
        "phase_only_slm_mask_generation",
        "Phase-only SLM1/SLM2 mask generation",
        "holography and CSLM route",
        ["vbb_study/equations/holography.py", "vbb_study/digital_twin/cslm_route.py"],
        "canonical_active",
        "physical_model",
        "implemented_active",
        "physically_modelled",
        "manufacturer_specification",
        "needs_manufacturer_data",
        refs=["search_lcos_slm_phase_calibration"],
        manufacturer=["SLM phase stroke at 1030 nm", "polarisation requirement", "fill factor", "reflectivity"],
        bench=["phase response calibration", "zero/order power split"],
        limitations=["ideal phase-only command until SLM response is measured"],
        validation=["interferometric or diffraction-efficiency phase calibration"],
        readiness="bench_validation_required",
        active_components=["SLM1_phase_plane", "SLM2_phase_plane"],
    ),
    _claim(
        "phase_quantisation_and_grayscale_export",
        "Phase wrapping, quantisation, and grayscale export",
        "SLM command files",
        ["vbb_study/equations/holography.py", "vbb_study/digital_twin/slm_calibration_masks.py"],
        "calibration_infrastructure",
        "measurement_method",
        "implemented_active",
        "numerically_modelled",
        "manufacturer_specification",
        "needs_manufacturer_data",
        refs=["search_lcos_slm_phase_calibration"],
        manufacturer=["gray-voltage LUT", "bit depth", "wavelength-specific phase response"],
        bench=["grayscale-to-phase response at 1030 nm"],
        readiness="bench_validation_required",
    ),
    _claim(
        "command_domain_carrier_grating",
        "Command-domain carrier grating for first Fourier session",
        "Stage 9A.1 carrier masks",
        ["vbb_study/digital_twin/slm_calibration_masks.py", "configs/studies/cslm_fourier_carrier_calibration_minimal_v1.json"],
        "calibration_infrastructure",
        "measurement_method",
        "implemented_active",
        "measurement_only",
        "bench_measurement",
        "needs_bench_measurement",
        manufacturer=["SLM display dimensions", "SLM orientation"],
        bench=[
            "downstream response versus carrier command cycles in the installed setup",
            "direct Fourier-plane order displacement only with temporary Fourier-plane access",
        ],
        assumption="carrier units are command cycles across the displayed area, not physical cycles/mm",
        validation=[
            "Stage 9A.1B downstream carrier-stop response session in the installed setup",
            "direct Fourier-plane carrier session only with temporary Fourier-plane access",
        ],
        readiness="bench_validation_required",
    ),
    _claim(
        "pixelated_slm_zero_order_and_unwanted_orders",
        "Pixelated SLM zero order and unwanted diffraction orders",
        "future 4F/order-selection realism",
        ["vbb_study/digital_twin/cslm_route.py", "vbb_study/digital_twin/slm_calibration_masks.py"],
        "future_planned",
        "physical_model",
        "planned_future",
        "future_not_implemented",
        "manufacturer_specification",
        "needs_bench_measurement",
        refs=["search_pixelated_slm_zero_order"],
        manufacturer=["fill factor", "reflectivity", "phase LUT"],
        bench=["zero/+1/residual order power fractions"],
        limitations=["current code records no order efficiency and no zero-order rejection"],
        readiness="bench_validation_required",
    ),
    _claim(
        "physical_fourier_filtering_future_route",
        "Component-owned physical 4F filtering",
        "CSLM 4F route",
        ["vbb_study/digital_twin/cslm_route.py", "vbb_study/digital_twin/bench_inventory.py"],
        "future_planned",
        "physical_model",
        "not_implemented",
        "future_not_implemented",
        "bench_measurement",
        "needs_bench_measurement",
        refs=["search_finite_aperture_4f_model"],
        manufacturer=["lens focal lengths", "clear apertures", "coatings"],
        bench=["lens positions", "Fourier-stop centre/radius", "carrier-to-order mapping"],
        readiness="bench_validation_required",
    ),
    _claim(
        "ideal_selected_order_handoff",
        "Ideal selected-order handoff before physical 4F validation",
        "CSLM diagnostic route",
        [
            "vbb_study/digital_twin/cslm_route.py",
            "docs/39_cslm_to_physical_axicon_handoff.md",
            "docs/41_measured_bench_inventory_and_4f_readiness.md",
        ],
        "canonical_active",
        "diagnostic_boundary",
        "implemented_diagnostic_only",
        "diagnostic_placeholder",
        "bench_measurement",
        "assumption_declared",
        bench=["measured Fourier-plane order selection and residual-order rejection"],
        assumption="selected-order field handoff is ideal until the physical 4F/filter plane is measured and implemented",
        limitations=["not a physical 4F filter transform", "does not validate order efficiency or finite-aperture clipping"],
        not_prove=["physical Fourier filtering", "bench order purity"],
        validation=["first Fourier-plane carrier calibration session", "measured 4F stop centre/radius"],
        readiness="bench_validation_required",
        active_components=["ideal_selected_order_handoff_plane"],
    ),
    _claim(
        "slm1_to_slm2_propagation",
        "Free-space propagation from SLM1 to SLM2",
        "CSLM route",
        ["vbb_study/digital_twin/cslm_route.py"],
        "canonical_active",
        "physical_model",
        "implemented_active",
        "physically_modelled",
        "bench_measurement",
        "needs_bench_measurement",
        refs=["search_bl_asm_sampling"],
        bench=["actual SLM1-to-SLM2 distance and relay geometry"],
        limitations=["demo geometry until bench measured"],
        readiness="bench_validation_required",
        active_components=["SLM1_to_SLM2_segment"],
    ),
    _claim(
        "slm_registration_and_coordinate_frames",
        "SLM/lab/Fourier/camera coordinate-frame registration",
        "coordinate contract",
        ["vbb_study/digital_twin/coordinate_contract.py", "docs/41_measured_bench_inventory_and_4f_readiness.md"],
        "calibration_infrastructure",
        "measurement_method",
        "implemented_diagnostic_only",
        "record_only",
        "bench_measurement",
        "needs_bench_measurement",
        bench=["SLM2-to-lab transform", "Fourier-plane frame", "camera-frame mapping"],
        readiness="bench_validation_required",
    ),
    _claim(
        "physical_axicon_bessel_conversion",
        "Thin scalar physical axicon Bessel conversion benchmark",
        "physical axicon branch",
        ["vbb_study/digital_twin/route_aware_axicon.py", "vbb_study/digital_twin/cslm_route.py"],
        "canonical_supporting",
        "physical_model",
        "implemented_benchmark_only",
        "physically_modelled",
        "manufacturer_specification",
        "needs_manufacturer_data",
        refs=["search_axicon_bessel_tolerance"],
        manufacturer=["axicon cone angle", "apex quality", "clear aperture", "coating"],
        bench=["axicon position/orientation", "z-stack validation"],
        readiness="bench_validation_required",
        active_components=["physical_axicon_benchmark_plane", "ideal_axicon_benchmark_reference_plane"],
    ),
    _claim(
        "physical_axicon_aperture_and_decentre",
        "Physical axicon finite aperture, decentre, and offset diagnostics",
        "physical axicon route",
        ["vbb_study/digital_twin/route_aware_axicon.py"],
        "canonical_supporting",
        "physical_model",
        "implemented_benchmark_only",
        "physically_modelled",
        "bench_measurement",
        "needs_bench_measurement",
        refs=["search_axicon_bessel_tolerance"],
        manufacturer=["clear aperture", "mechanical mount tolerances"],
        bench=["beam-to-axicon centring", "axicon axial/lateral position"],
        readiness="bench_validation_required",
    ),
    _claim(
        "vortex_phase_generation",
        "Vortex/topological-charge phase generation",
        "SLM1 route",
        ["vbb_study/equations/holography.py", "vbb_study/digital_twin/cslm_route.py"],
        "canonical_active",
        "physical_model",
        "implemented_active",
        "physically_modelled",
        "textbook_or_standard_reference",
        "needs_verified_source",
        refs=["search_vortex_bessel_structured_beams"],
        bench=["measured charge-dependent annular field"],
        readiness="simulation_claim_possible",
        active_components=["SLM1_phase_plane"],
    ),
    _claim(
        "energy_and_fluence_accounting",
        "Energy ledger and optical fluence scaling",
        "digital twin energy/fluence layer",
        ["vbb_study/digital_twin/energy_accounting.py", "vbb_study/digital_twin/field_fluence.py"],
        "canonical_active",
        "derived_metric",
        "implemented_active",
        "numerically_modelled",
        "engineering_estimate",
        "assumption_declared",
        bench=["pulse energy before optics", "transmission through actual chain"],
        limitations=["fluence is not absorbed energy or material dose"],
        readiness="simulation_claim_possible",
        active_components=["source_field", "input_conditioning_boundary", "post_SLM2_pre_4F_diagnostic_plane"],
    ),
    _claim(
        "ring_centre_radius_dark_core_and_uniformity_metrics",
        "Annular ring centre/radius/dark-core/uniformity metrics",
        "axis tracking and measured image metrics",
        ["vbb_study/digital_twin/annular_axis_tracking.py", "vbb_study/digital_twin/measured_image_metrics.py"],
        "canonical_active",
        "derived_metric",
        "implemented_active",
        "numerically_modelled",
        "bench_measurement",
        "needs_bench_measurement",
        refs=["search_camera_annular_beam_profiling"],
        bench=["camera z-stack images and scale/orientation calibration"],
        limitations=["pixel metrics are not physical metrics until calibrated"],
        readiness="bench_validation_required",
    ),
    _claim(
        "camera_z_stack_acquisition",
        "Camera z-stack acquisition records",
        "Stage 9A acquisition plan",
        ["vbb_study/digital_twin/calibration_acquisition.py", "configs/studies/cslm_physical_axicon_calibration_campaign_v1.json"],
        "calibration_infrastructure",
        "measurement_method",
        "implemented_diagnostic_only",
        "measurement_only",
        "bench_measurement",
        "needs_bench_measurement",
        bench=["raw camera files", "z positions", "exposure/gain logs"],
        readiness="bench_validation_required",
    ),
    _claim(
        "camera_coordinate_calibration",
        "Camera coordinate calibration and physical-unit metric gate",
        "measured image metrics",
        ["vbb_study/digital_twin/measured_image_metrics.py", "docs/42_calibration_acquisition_and_measured_data_ingestion.md"],
        "calibration_infrastructure",
        "measurement_method",
        "implemented_diagnostic_only",
        "measurement_only",
        "bench_measurement",
        "needs_bench_measurement",
        manufacturer=["camera pixel pitch", "sensor format"],
        bench=["magnification", "orientation", "reference-plane relation"],
        readiness="bench_validation_required",
    ),
    _claim(
        "multi_plane_phase_retrieval_future",
        "Multi-plane phase retrieval from measured intensity z-stacks",
        "future correction research",
        ["docs/42_calibration_acquisition_and_measured_data_ingestion.md"],
        "future_planned",
        "future_capability",
        "planned_future",
        "future_not_implemented",
        "unsupported_future_work",
        "needs_verified_source",
        refs=["search_multi_plane_phase_retrieval"],
        bench=["calibrated multi-plane z-stack"],
        readiness="not_claimable",
    ),
    _claim(
        "effective_aberration_inference_future",
        "Effective aberration inference from measured/model mismatch",
        "future correction research",
        ["docs/42_calibration_acquisition_and_measured_data_ingestion.md"],
        "future_planned",
        "future_capability",
        "planned_future",
        "future_not_implemented",
        "unsupported_future_work",
        "needs_verified_source",
        refs=["search_phase_diversity_zernike"],
        bench=["repeatable calibrated z-stacks"],
        readiness="not_claimable",
    ),
    _claim(
        "zernike_or_phase_conjugate_correction_future",
        "Zernike or phase-conjugate SLM correction",
        "future SLM correction",
        ["vbb_study/digital_twin/control_contract.py", "docs/40_editable_hardware_geometry_contract.md"],
        "future_planned",
        "future_capability",
        "planned_future",
        "future_not_implemented",
        "unsupported_future_work",
        "needs_verified_source",
        refs=["search_phase_diversity_zernike"],
        manufacturer=["SLM phase response"],
        bench=["identified correction target and validation capture"],
        readiness="not_claimable",
    ),
    _claim(
        "neural_fast_estimator_future",
        "Neural fast estimator for alignment/correction",
        "future optional research",
        ["docs/42_calibration_acquisition_and_measured_data_ingestion.md"],
        "future_planned",
        "future_capability",
        "not_implemented",
        "future_not_implemented",
        "unsupported_future_work",
        "needs_verified_source",
        refs=["search_synthetic_to_real_uncertainty"],
        bench=["large calibrated dataset with uncertainty labels"],
        readiness="not_claimable",
    ),
    _claim(
        "fused_silica_application_boundary",
        "Fused-silica application boundary and evidence template",
        "future material application",
        ["configs/materials/fused_silica_evidence_template.json", "docs/44_code_to_evidence_audit.md"],
        "future_planned",
        "material_application_hypothesis",
        "not_implemented",
        "future_not_implemented",
        "unsupported_future_work",
        "needs_bench_measurement",
        refs=["search_fused_silica_bessel_tgv", "search_fused_silica_waveguide", "search_fused_silica_welding"],
        bench=["fused-silica sample/process outcomes", "microscopy/metrology"],
        readiness="not_claimable",
    ),
    _claim(
        "legacy_crznse_material_proxy_branch",
        "Legacy Cr:ZnSe material proxy branch",
        "legacy materials work",
        ["vbb_study/equations/materials.py", "vbb_study/config.py", "notebooks/materials/01_material_proxy_fluence_and_thresholds.ipynb"],
        "legacy_retained",
        "material_application_hypothesis",
        "implemented_legacy",
        "diagnostic_placeholder",
        "diagnostic_placeholder",
        "assumption_declared",
        refs=["search_crznse_material_proxy_recheck"],
        assumption="Cr:ZnSe-specific proxy assumptions; not valid for fused silica decision-making",
        limitations=["not calibrated", "not transferable to fused silica"],
        readiness="not_claimable",
    ),
    _claim(
        "vector_beam_branch",
        "Vector/Jones beam branch",
        "legacy or optional vector studies",
        ["vbb_study/equations/vector_jones.py", "vbb_study/publication/vector.py", "notebooks/vector/01_vector_beam_theory_atlas.ipynb"],
        "experimental_development",
        "physical_model",
        "implemented_legacy",
        "numerically_modelled",
        "textbook_or_standard_reference",
        "needs_verified_source",
        refs=["search_vector_beam_jones_reference"],
        limitations=["not part of current scalar CSLM -> 4F -> physical-axicon campaign"],
        readiness="method_principle_only",
    ),
    _claim(
        "hexagonal_polygonal_discrete_beam_branch",
        "Hexagonal, polygonal, and discrete N-fold beam studies",
        "advanced exploratory beam shaping",
        ["vbb_study/equations/polygonal.py", "vbb_study/studies/polygonal_cases.py", "notebooks/advanced/03_discrete_nfold_beams.ipynb"],
        "experimental_development",
        "numerical_method",
        "implemented_legacy",
        "numerically_modelled",
        "engineering_estimate",
        "assumption_declared",
        limitations=["not prerequisite for first scalar CSLM/4F calibration"],
        readiness="method_principle_only",
    ),
    _claim(
        "capsule_or_weld_feature_geometry_branch",
        "Capsule/weld feature geometry branch",
        "advanced material-geometry proxy",
        ["vbb_study/equations/capsule_geometry.py", "vbb_study/publication/capsule.py", "notebooks/advanced/01_capsule_weld_feature_design.ipynb"],
        "experimental_development",
        "material_application_hypothesis",
        "implemented_legacy",
        "diagnostic_placeholder",
        "diagnostic_placeholder",
        "assumption_declared",
        limitations=["geometric planning only; no weld/material response model"],
        readiness="not_claimable",
    ),
]


CODE_INVENTORY: list[dict[str, Any]] = [
    {
        "item_id": "dt_cslm_route",
        "path": "vbb_study/digital_twin/cslm_route.py",
        "classification": "canonical_active",
        "main_purpose": "component-owned CSLM diagnostic route and opt-in physical-axicon benchmark",
        "current_route_or_study_relevance": "central scalar CSLM -> future 4F -> physical axicon programme",
        "imports_or_dependencies_checked": ["tests/test_stage8c3r5_cslm_component_route.py", "tests/test_stage8c3r5_1_cslm_axicon_handoff.py", "notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb"],
        "affects_current_canonical_execution": True,
        "affects_current_laboratory_decisions": True,
        "replacement_or_successor": None,
        "why_retained": "canonical route declaration and claim boundary",
        "safe_to_archive_later": False,
    },
    {
        "item_id": "dt_4f_readiness_inventory",
        "path": "vbb_study/digital_twin/bench_inventory.py",
        "classification": "calibration_infrastructure",
        "main_purpose": "records measured/placeholder bench inventory and 4F readiness blockers",
        "current_route_or_study_relevance": "prevents fake physical 4F activation",
        "imports_or_dependencies_checked": ["tests/test_stage8c3r5_3_bench_inventory_and_4f_readiness.py", "configs/hardware/*.json"],
        "affects_current_canonical_execution": False,
        "affects_current_laboratory_decisions": True,
        "replacement_or_successor": "measured bench inventory overlay",
        "why_retained": "level C/D readiness gate",
        "safe_to_archive_later": False,
    },
    {
        "item_id": "dt_stage9a_acquisition",
        "path": "vbb_study/digital_twin/calibration_acquisition.py",
        "classification": "calibration_infrastructure",
        "main_purpose": "calibration campaign packages, immutable raw ingestion, pixel metrics",
        "current_route_or_study_relevance": "first lab-data acquisition path",
        "imports_or_dependencies_checked": ["tests/test_stage9a_calibration_acquisition_and_ingestion.py"],
        "affects_current_canonical_execution": False,
        "affects_current_laboratory_decisions": True,
        "replacement_or_successor": None,
        "why_retained": "creates acquisition evidence without modelling new physics",
        "safe_to_archive_later": False,
    },
    {
        "item_id": "dt_stage9a1_carrier_masks",
        "path": "vbb_study/digital_twin/slm_calibration_masks.py",
        "classification": "calibration_infrastructure",
        "main_purpose": "command-domain SLM2 carrier masks and first Fourier session pack",
        "current_route_or_study_relevance": "P0 carrier-to-Fourier-plane mapping",
        "imports_or_dependencies_checked": ["tests/test_stage9a1_fourier_carrier_session_pack.py"],
        "affects_current_canonical_execution": False,
        "affects_current_laboratory_decisions": True,
        "replacement_or_successor": "measured carrier mapping",
        "why_retained": "first physical 4F calibration evidence generator",
        "safe_to_archive_later": False,
    },
    {
        "item_id": "dt_stage9a1b_downstream_carrier_stop",
        "path": "vbb_study/digital_twin/downstream_carrier_stop.py",
        "classification": "calibration_infrastructure",
        "main_purpose": "downstream-focus empirical carrier/stop characterisation session pack",
        "current_route_or_study_relevance": "installed downstream camera operating-point selection without claiming Fourier-plane coordinates",
        "imports_or_dependencies_checked": ["tests/test_stage9a1b_downstream_carrier_stop_characterisation.py", "configs/studies/cslm_carrier_stop_characterisation_downstream_v1.json"],
        "affects_current_canonical_execution": False,
        "affects_current_laboratory_decisions": True,
        "replacement_or_successor": "direct Fourier-plane mapping when temporary diagnostic access exists",
        "why_retained": "honest installed-camera calibration path before physical 4F activation",
        "safe_to_archive_later": False,
    },
    {
        "item_id": "dt_measured_image_metrics",
        "path": "vbb_study/digital_twin/measured_image_metrics.py",
        "classification": "calibration_infrastructure",
        "main_purpose": "pixel-space measured-image metrics with physical-unit gate",
        "current_route_or_study_relevance": "camera z-stack evidence once acquired",
        "imports_or_dependencies_checked": ["tests/test_stage9a_calibration_acquisition_and_ingestion.py"],
        "affects_current_canonical_execution": False,
        "affects_current_laboratory_decisions": True,
        "replacement_or_successor": "calibrated camera comparison after scale/reference relation",
        "why_retained": "safe pre-calibration image QA",
        "safe_to_archive_later": False,
    },
    {
        "item_id": "equations_propagation_holography",
        "path": "vbb_study/equations/propagation.py; vbb_study/equations/holography.py",
        "classification": "canonical_active",
        "main_purpose": "BL-ASM, phase wrapping, SPP, carrier, quantisation helpers",
        "current_route_or_study_relevance": "core numerical/phase primitives",
        "imports_or_dependencies_checked": ["digital_twin route modules", "physics validation tests"],
        "affects_current_canonical_execution": True,
        "affects_current_laboratory_decisions": True,
        "replacement_or_successor": None,
        "why_retained": "core active numerical methods",
        "safe_to_archive_later": False,
    },
    {
        "item_id": "notebook_full_cockpit",
        "path": "notebooks/digital_twin/00_full_beam_to_write_cockpit_MVP.ipynb",
        "classification": "canonical_supporting",
        "main_purpose": "visible cockpit, route, calibration, and claim-boundary dashboard",
        "current_route_or_study_relevance": "human-facing canonical status view",
        "imports_or_dependencies_checked": ["tests/test_stage8c*_notebook*.py", "tests/test_stage9a2_code_to_evidence_audit.py"],
        "affects_current_canonical_execution": False,
        "affects_current_laboratory_decisions": True,
        "replacement_or_successor": None,
        "why_retained": "notebook visibility",
        "safe_to_archive_later": False,
    },
    {
        "item_id": "configs_hardware_and_studies",
        "path": "configs/hardware/*.json; configs/studies/*.json",
        "classification": "calibration_infrastructure",
        "main_purpose": "demo/measured templates and first-session calibration plans",
        "current_route_or_study_relevance": "evidence capture and readiness gating",
        "imports_or_dependencies_checked": ["bench_inventory.py", "slm_calibration_masks.py", "calibration_acquisition.py"],
        "affects_current_canonical_execution": False,
        "affects_current_laboratory_decisions": True,
        "replacement_or_successor": "filled measured bench profiles",
        "why_retained": "structured source of required inputs",
        "safe_to_archive_later": False,
    },
    {
        "item_id": "materials_proxy_branch",
        "path": "vbb_study/equations/materials.py; vbb_study/config.py; notebooks/materials/*.ipynb; docs/03_materials_application.md",
        "classification": "legacy_retained",
        "main_purpose": "Cr:ZnSe/material planning proxies and threshold-style geometry",
        "current_route_or_study_relevance": "not part of immediate fused-silica CSLM calibration",
        "imports_or_dependencies_checked": ["tests/test_stage8a1_literature_anchors.py", "publication/materials.py"],
        "affects_current_canonical_execution": False,
        "affects_current_laboratory_decisions": False,
        "replacement_or_successor": "future calibrated fused-silica evidence profile",
        "why_retained": "historical exploratory material-planning context",
        "safe_to_archive_later": True,
    },
    {
        "item_id": "vector_branch",
        "path": "vbb_study/equations/vector_jones.py; vbb_study/publication/vector.py; notebooks/vector/*.ipynb",
        "classification": "experimental_development",
        "main_purpose": "optional vector/Jones beam studies",
        "current_route_or_study_relevance": "excluded from scalar first lab path",
        "imports_or_dependencies_checked": ["notebooks/vector/*.ipynb", "publication/vector.py"],
        "affects_current_canonical_execution": False,
        "affects_current_laboratory_decisions": False,
        "replacement_or_successor": "future vector engine if needed",
        "why_retained": "exploratory structured-beam branch",
        "safe_to_archive_later": True,
    },
    {
        "item_id": "polygonal_discrete_branch",
        "path": "vbb_study/equations/polygonal.py; vbb_study/studies/*polygonal*; notebooks/advanced/02_*.ipynb; notebooks/advanced/03_*.ipynb",
        "classification": "experimental_development",
        "main_purpose": "polygonal, hexagonal, and discrete N-fold studies",
        "current_route_or_study_relevance": "not prerequisite for scalar CSLM/4F calibration",
        "imports_or_dependencies_checked": ["advanced notebooks", "publication/advanced.py"],
        "affects_current_canonical_execution": False,
        "affects_current_laboratory_decisions": False,
        "replacement_or_successor": None,
        "why_retained": "optional beam-shaping research",
        "safe_to_archive_later": True,
    },
    {
        "item_id": "capsule_weld_branch",
        "path": "vbb_study/equations/capsule_geometry.py; vbb_study/publication/capsule.py; notebooks/advanced/01_capsule_weld_feature_design.ipynb",
        "classification": "experimental_development",
        "main_purpose": "feature-geometry/capsule/weld planning proxy",
        "current_route_or_study_relevance": "excluded from immediate optical calibration",
        "imports_or_dependencies_checked": ["advanced capsule notebook", "publication/capsule.py"],
        "affects_current_canonical_execution": False,
        "affects_current_laboratory_decisions": False,
        "replacement_or_successor": "future material-calibrated geometry model",
        "why_retained": "advanced geometry exploratory branch",
        "safe_to_archive_later": True,
    },
    {
        "item_id": "generated_outputs",
        "path": "outputs/**",
        "classification": "generated_output",
        "main_purpose": "figures, CSVs, manifests, calibration packages, pytest scratch",
        "current_route_or_study_relevance": "evidence/preview outputs only",
        "imports_or_dependencies_checked": ["figure registry", "tests write under tmp outputs"],
        "affects_current_canonical_execution": False,
        "affects_current_laboratory_decisions": False,
        "replacement_or_successor": "regenerated from source code/configs",
        "why_retained": "diagnostic previews and generated evidence packages",
        "safe_to_archive_later": True,
    },
]


def _backlog(
    research_id: str,
    title: str,
    priority: str,
    horizon: str,
    claim_ids: list[str],
    gap: str,
    terms: list[str],
    deliverable: str,
    unblocks: str,
    *,
    literature: str = "",
    manufacturer: str = "",
    bench: str = "",
    owner: str = "project",
    status: str = "not_started",
    blocker_level: str | None = None,
) -> dict[str, Any]:
    return {
        "research_id": research_id,
        "title": title,
        "priority": priority,
        "time_horizon": horizon,
        "linked_claim_ids": claim_ids,
        "linked_code_paths": [
            p for cid in claim_ids for c in CLAIMS if c["claim_id"] == cid for p in c["implementation_paths"]
        ][:6],
        "why_the_code_needs_this": "Moves the labelled assumption from placeholder/search-needed to measured or source-verified evidence.",
        "current_gap": gap,
        "literature_needed": literature,
        "manufacturer_data_needed": manufacturer,
        "bench_data_needed": bench,
        "recommended_search_terms": terms,
        "suggested_primary_source_types": ["peer-reviewed primary paper", "manufacturer datasheet", "bench measurement log"],
        "expected_deliverable": deliverable,
        "what_code_or_lab_stage_it_unblocks": unblocks,
        "what_claim_it_makes_possible": "A narrower simulation or bench-validation claim with explicit evidence provenance.",
        "owner": owner,
        "status": status,
        "readiness_blocker_level": blocker_level,
    }


RESEARCH_BACKLOG: list[dict[str, Any]] = [
    _backlog("P0_SLM_SPEC", "Actual SLM make/model, pitch, active area, resolution, orientation", "P0", "immediate", ["phase_only_slm_mask_generation", "phase_quantisation_and_grayscale_export"], "SLM geometry and phase response are placeholders.", ["LCOS SLM 1030 nm pixel pitch phase response"], "manufacturer specification pack plus bench orientation note", "Level C physical 4F readiness", manufacturer="SLM datasheets/LUTs", bench="orientation and displayed active area", blocker_level="Level-C"),
    _backlog("P0_SLM_PHASE", "SLM phase response and polarisation requirement at 1030 nm", "P0", "immediate", ["phase_only_slm_mask_generation", "phase_quantisation_and_grayscale_export"], "Grayscale command is not yet calibrated to phase.", ["LCOS SLM phase calibration 1030 nm diffraction efficiency"], "phase-response calibration plan and first measurement", "SLM command trust boundary", manufacturer="phase stroke, polarisation", bench="grayscale-phase or diffraction curve", blocker_level="Level-C"),
    _backlog("P0_LENS_GEOMETRY", "Lens focal lengths, clear apertures, coatings, and real positions", "P0", "immediate", ["physical_fourier_filtering_future_route"], "4F lens geometry is declaration only.", ["finite aperture 4F optical system scalar diffraction"], "measured lens table", "initial scalar 4F model", manufacturer="lens specs/coatings", bench="lens positions", blocker_level="Level-C"),
    _backlog("P0_CARRIER_MAPPING", "Carrier command to Fourier-plane order-position mapping", "P0", "immediate", ["command_domain_carrier_grating", "physical_fourier_filtering_future_route"], "Command cycles are not yet physical Fourier-plane coordinates.", ["pixelated spatial light modulator zero order Fourier filtering"], "downstream carrier-stop response now; direct Fourier-plane mapping only with temporary diagnostic access", "Fourier-plane mapping", bench="downstream response versus carrier command cycles now; direct order positions versus carrier cycles only with temporary Fourier-plane access", blocker_level="Level-C"),
    _backlog("P0_FOURIER_STOP", "Fourier-stop geometry, centre, radius, and adjustment convention", "P0", "immediate", ["physical_fourier_filtering_future_route"], "Stop model cannot be placed without measured geometry.", ["Fourier plane spatial filter SLM first order"], "stop-centre/radius measurement log", "selected-order 4F model", manufacturer="stop/mount specs", bench="centre/radius scans", blocker_level="Level-C"),
    _backlog("P0_CAMERA", "Camera pixel pitch, magnification, orientation, linearity, saturation", "P0", "immediate", ["camera_coordinate_calibration", "camera_z_stack_acquisition"], "Pixel metrics cannot be physical metrics yet.", ["camera annular beam profiling z stack calibration"], "camera calibration sheet", "measured-to-model comparison", manufacturer="sensor pixel pitch/bit depth", bench="magnification/orientation/linearity", blocker_level="Level-D"),
    _backlog("P0_AXICON", "Physical axicon specification, aperture, orientation, and bench position", "P0", "immediate", ["physical_axicon_bessel_conversion", "physical_axicon_aperture_and_decentre"], "Benchmark uses demo axicon parameters until measured.", ["physical axicon decentration tilt Bessel beam tolerance"], "axicon spec and alignment log", "post-axicon z-stack validation", manufacturer="axicon angle/aperture/apex", bench="position/orientation", blocker_level="Level-D"),
    _backlog("P0_INPUT_BEAM", "Input beam size, ellipticity, centring, and polarisation at relevant planes", "P0", "immediate", ["slm1_to_slm2_propagation", "energy_and_fluence_accounting"], "Source field is demo geometry.", ["Gaussian beam measurement SLM plane polarisation"], "beam profiling sheet", "route-normalised simulation comparison", bench="beam size/centering/polarisation", blocker_level="Level-D"),
    _backlog("P1_BLASM", "BL-ASM/ASM sampling validity for carrier, aperture, and distances", "P1", "pre_4f_model", ["angular_spectrum_or_bl_asm_propagation", "finite_sampling_and_aliasing_control"], "Need source-backed validity criteria.", ["band limited angular spectrum propagation aliasing finite aperture"], "validated sampling memo", "publication simulation claim", literature="primary BL-ASM/sampling references"),
    _backlog("P1_4F_MODEL", "Finite-aperture thin-lens / 4F modelling", "P1", "pre_4f_model", ["physical_fourier_filtering_future_route"], "No component-owned lens model exists.", ["finite aperture thin lens Fourier optics 4F scalar propagation"], "4F model design note", "Stage 9B physical 4F implementation", literature="Fourier optics primary/textbook sources"),
    _backlog("P1_SLM_DIFFRACTION", "Pixelated SLM diffraction and zero-order behaviour", "P1", "pre_4f_model", ["pixelated_slm_zero_order_and_unwanted_orders"], "Order power split unknown.", ["pixelated spatial light modulator zero order Fourier filtering"], "order-efficiency evidence plan", "4F throughput model", literature="SLM diffraction papers", manufacturer="fill factor/reflectivity"),
    _backlog("P1_PHASE_QUANT", "Phase quantisation and calibration effects", "P1", "pre_4f_model", ["phase_quantisation_and_grayscale_export"], "Quantisation exists but physical response unverified.", ["LCOS SLM phase calibration 1030 nm diffraction efficiency"], "phase quantisation sensitivity note", "SLM mask publication method", literature="SLM calibration primary sources"),
    _backlog("P1_AXICON_TOL", "Axicon tolerance, decentre, tilt, apex quality, and aperture behaviour", "P1", "pre_4f_model", ["physical_axicon_aperture_and_decentre"], "Tolerance model needs literature/manufacturer context.", ["physical axicon decentration tilt Bessel beam tolerance"], "axicon tolerance register", "alignment tolerance claim", literature="axicon tolerance papers", manufacturer="apex/angle tolerances"),
    _backlog("P1_CAMERA_PROFILING", "Camera-based annular beam profiling and z-stack metrology", "P1", "pre_4f_model", ["ring_centre_radius_dark_core_and_uniformity_metrics", "camera_z_stack_acquisition"], "Metric reliability needs metrology backing.", ["camera based annular beam profiling z-stack metrology"], "metric validity memo", "measured validation figures", literature="beam profiling/metrology sources"),
    _backlog("P2_PHASE_RETRIEVAL", "Multi-plane phase retrieval", "P2", "pre_inverse_correction", ["multi_plane_phase_retrieval_future"], "No phase retrieval implemented.", ["multi plane phase retrieval vortex Bessel beam intensity z stack"], "algorithm literature review", "future phase reconstruction", literature="phase retrieval primary papers"),
    _backlog("P2_PHASE_DIVERSITY", "Phase-diversity wavefront sensing", "P2", "pre_inverse_correction", ["effective_aberration_inference_future"], "No phase-diversity inference implemented.", ["phase diversity Zernike aberration estimation spatial light modulator correction"], "phase-diversity feasibility note", "future effective aberration inference"),
    _backlog("P2_IDENTIFIABILITY", "Effective aberration versus component-root-cause identifiability", "P2", "pre_inverse_correction", ["effective_aberration_inference_future"], "Need to know what can be inferred from intensity-only data.", ["identifiability aberration inference intensity z stack"], "identifiability risk register", "safe correction claims"),
    _backlog("P2_ZERNIKE_BASIS", "Zernike basis and normalisation", "P2", "pre_inverse_correction", ["zernike_or_phase_conjugate_correction_future"], "Basis conventions must be locked before correction.", ["Zernike polynomial normalisation optical aberration correction"], "basis convention doc", "phase-correction route"),
    _backlog("P2_SLM_CONJ", "SLM conjugate-phase correction", "P2", "pre_inverse_correction", ["zernike_or_phase_conjugate_correction_future"], "No correction map is validated.", ["SLM conjugate phase correction aberration compensation"], "correction validation plan", "future SLM correction stage"),
    _backlog("P2_SENSORLESS", "Sensorless optimisation", "P2", "pre_inverse_correction", ["zernike_or_phase_conjugate_correction_future"], "No optimisation loop exists.", ["sensorless adaptive optics spatial light modulator optimisation"], "sensorless optimisation literature note", "future adaptive optimisation"),
    _backlog("P2_NEURAL", "Synthetic-to-real training and uncertainty for neural estimators", "P2", "later_research", ["neural_fast_estimator_future"], "No dataset or uncertainty model exists.", ["synthetic to real optical alignment neural estimator uncertainty"], "AI feasibility note", "optional future estimator"),
    _backlog("P3_FS_REGIMES", "Fused-silica internal modification regimes", "P3", "pre_fused_silica_pilot", ["fused_silica_application_boundary"], "No fused-silica process evidence in repo.", ["femtosecond laser fused silica Bessel beam channel drilling"], "fused-silica literature table", "pilot parameter window", literature="primary fused-silica fs processing papers"),
    _backlog("P3_FS_INTERFACE", "Bessel/vortex-Bessel propagation at the sample interface", "P3", "pre_fused_silica_pilot", ["fused_silica_application_boundary"], "No sample-interface model is active.", ["fused silica Bessel beam interface aberration femtosecond"], "interface research note", "sample-entry claim boundary"),
    _backlog("P3_TGV", "TGV/channel formation and etching", "P3", "pre_fused_silica_pilot", ["fused_silica_application_boundary"], "TGV etch response is not modelled.", ["femtosecond laser fused silica Bessel beam channel drilling etching"], "TGV evidence plan", "TGV pilot design"),
    _backlog("P3_WAVEGUIDE", "Waveguide-writing regimes and characterisation", "P3", "pre_fused_silica_pilot", ["fused_silica_application_boundary"], "Waveguide outcomes are not predicted.", ["ultrafast laser fused silica waveguide Bessel beam writing"], "waveguide evidence plan", "waveguide pilot criteria"),
    _backlog("P3_WELDING", "Ultrafast glass welding / symmetric weld-feature conditions", "P3", "pre_fused_silica_pilot", ["fused_silica_application_boundary", "capsule_or_weld_feature_geometry_branch"], "Weld-feature branch is geometry only.", ["femtosecond laser fused silica glass welding Bessel beam"], "welding evidence plan", "future weld-feature claim"),
    _backlog("LEGACY_CRZNSE", "CrZnSe legacy materials proxy work", "P3", "later_research", ["legacy_crznse_material_proxy_branch"], "Retained but excluded from fused-silica decisions.", ["CrZnSe femtosecond laser waveguide writing threshold"], "legacy separation note", "no immediate lab path dependency"),
    _backlog("LEGACY_VECTOR", "Vector beams and Jones modelling", "P3", "later_research", ["vector_beam_branch"], "Optional vector branch not needed for scalar campaign.", ["Jones vector beam spatial light modulator"], "optional vector review", "future vector extension"),
    _backlog("LEGACY_POLYGONAL", "Hexagonal / polygonal / discrete beam studies", "P3", "later_research", ["hexagonal_polygonal_discrete_beam_branch"], "Exploratory branch not needed for first lab calibration.", ["polygonal Bessel beams discrete n-fold beams"], "optional beam-shaping review", "advanced studies only"),
    _backlog("LEGACY_CAPSULE", "Capsule and weld-feature geometry studies", "P3", "later_research", ["capsule_or_weld_feature_geometry_branch"], "Geometry proxy not material physics.", ["ultrafast laser glass welding feature geometry"], "optional geometry review", "future material-calibrated design"),
]


def _search(claim_id: str, question: str, queries: list[str], source_types: list[str]) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "research_question": question,
        "exact_search_queries": queries,
        "target_journals_or_source_types": source_types,
        "required_evidence_standard": "verified primary source or authoritative textbook/standard; DOI/publisher metadata must be checked before BibTeX entry is added",
        "candidate_source_status": "search_needed_no_verified_citation",
        "why_this_reference_is_needed": "Prevents principle-level literature support from being confused with validation of this repository's numerical representation or bench.",
    }


LITERATURE_SEARCH_PLAN = [
    _search("angular_spectrum_or_bl_asm_propagation", "What sampling/FOV rules are accepted for BL-ASM/ASM finite-aperture propagation?", ["band limited angular spectrum propagation aliasing finite aperture"], ["peer-reviewed optics methods paper", "Fourier optics textbook"]),
    _search("phase_only_slm_mask_generation", "What source supports LCOS SLM phase calibration and diffraction efficiency at near-1030 nm?", ["LCOS SLM phase calibration 1030 nm diffraction efficiency"], ["peer-reviewed SLM calibration paper", "manufacturer application note"]),
    _search("pixelated_slm_zero_order_and_unwanted_orders", "How do pixelated SLMs distribute zero, first, and unwanted orders?", ["pixelated spatial light modulator zero order Fourier filtering"], ["peer-reviewed SLM diffraction paper"]),
    _search("physical_axicon_bessel_conversion", "How do physical axicon decentre, tilt, apex and finite aperture affect Bessel beams?", ["physical axicon decentration tilt Bessel beam tolerance"], ["peer-reviewed axicon tolerance paper", "manufacturer axicon datasheet"]),
    _search("multi_plane_phase_retrieval_future", "Which multi-plane phase retrieval methods are suitable for vortex-Bessel intensity z-stacks?", ["multi plane phase retrieval vortex Bessel beam intensity z stack"], ["peer-reviewed phase retrieval paper"]),
    _search("effective_aberration_inference_future", "What phase-diversity/Zernike methods can estimate aberrations from intensity data?", ["phase diversity Zernike aberration estimation spatial light modulator correction"], ["peer-reviewed adaptive optics paper"]),
    _search("neural_fast_estimator_future", "What evidence standards are required for synthetic-to-real neural optical estimators?", ["synthetic to real optical alignment neural estimator uncertainty"], ["peer-reviewed ML uncertainty paper"]),
    _search("fused_silica_application_boundary", "What regimes exist for fs-laser fused-silica Bessel channel/TGV processing?", ["femtosecond laser fused silica Bessel beam channel drilling"], ["peer-reviewed ultrafast processing paper"]),
    _search("fused_silica_application_boundary", "What regimes exist for fs-laser fused-silica waveguide writing with Bessel beams?", ["ultrafast laser fused silica waveguide Bessel beam writing"], ["peer-reviewed waveguide-writing paper"]),
    _search("fused_silica_application_boundary", "What evidence exists for fs-laser fused-silica glass welding with Bessel-like beams?", ["femtosecond laser fused silica glass welding Bessel beam"], ["peer-reviewed glass welding paper"]),
    _search("legacy_crznse_material_proxy_branch", "Which Cr:ZnSe/ZnSe-family material proxy assumptions are material-specific and non-transferable?", ["CrZnSe femtosecond laser waveguide writing threshold", "ZnSe femtosecond laser modification threshold"], ["peer-reviewed material processing paper"]),
]


MANUFACTURER_EVIDENCE_REGISTER = {
    "register_type": "manufacturer_evidence",
    "value_state_vocabulary": ["unknown", "placeholder", "estimated", "manufacturer_specified", "measured"],
    "entries": [
        {"evidence_id": "M_SLM1_SPEC", "component": "SLM1", "required_for_claim_ids": ["phase_only_slm_mask_generation", "phase_quantisation_and_grayscale_export"], "needed_fields": ["make", "model", "pixel_pitch_um", "resolution", "phase_stroke_at_1030_nm", "polarisation"], "current_value_state": "unknown", "source_document": None, "verified": False},
        {"evidence_id": "M_SLM2_SPEC", "component": "SLM2", "required_for_claim_ids": ["phase_only_slm_mask_generation", "command_domain_carrier_grating"], "needed_fields": ["make", "model", "pixel_pitch_um", "active_area", "fill_factor", "reflectivity"], "current_value_state": "unknown", "source_document": None, "verified": False},
        {"evidence_id": "M_4F_LENSES", "component": "4F lenses", "required_for_claim_ids": ["physical_fourier_filtering_future_route"], "needed_fields": ["focal_length_mm", "clear_aperture_mm", "coating", "part_number"], "current_value_state": "unknown", "source_document": None, "verified": False},
        {"evidence_id": "M_CAMERA", "component": "camera", "required_for_claim_ids": ["camera_coordinate_calibration"], "needed_fields": ["pixel_pitch_um", "bit_depth", "linearity", "sensor_size"], "current_value_state": "unknown", "source_document": None, "verified": False},
        {"evidence_id": "M_AXICON", "component": "physical axicon", "required_for_claim_ids": ["physical_axicon_bessel_conversion"], "needed_fields": ["cone_angle", "clear_aperture", "apex_quality", "coating"], "current_value_state": "unknown", "source_document": None, "verified": False},
    ],
}


BENCH_EVIDENCE_REGISTER = {
    "register_type": "bench_evidence",
    "value_state_vocabulary": ["unknown", "placeholder", "estimated", "manufacturer_specified", "measured"],
    "entries": [
        {"evidence_id": "B_CARRIER_MAPPING", "measurement": "downstream carrier-stop response now; direct Fourier-plane order position only with temporary Fourier-plane access", "required_for_claim_ids": ["command_domain_carrier_grating", "physical_fourier_filtering_future_route"], "current_value_state": "unknown", "raw_data_path": None, "derived_data_path": None, "ready": False},
        {"evidence_id": "B_STOP_GEOMETRY", "measurement": "Fourier-stop centre/radius/adjustment convention", "required_for_claim_ids": ["physical_fourier_filtering_future_route"], "current_value_state": "unknown", "raw_data_path": None, "derived_data_path": None, "ready": False},
        {"evidence_id": "B_CAMERA_SCALE", "measurement": "camera magnification/orientation/reference-plane relation", "required_for_claim_ids": ["camera_coordinate_calibration"], "current_value_state": "unknown", "raw_data_path": None, "derived_data_path": None, "ready": False},
        {"evidence_id": "B_AXICON_ZSTACK", "measurement": "Gaussian/vortex post-axicon z-stack", "required_for_claim_ids": ["physical_axicon_bessel_conversion", "ring_centre_radius_dark_core_and_uniformity_metrics"], "current_value_state": "unknown", "raw_data_path": None, "derived_data_path": None, "ready": False},
        {"evidence_id": "B_INPUT_BEAM", "measurement": "beam size/ellipticity/centring/polarisation", "required_for_claim_ids": ["slm1_to_slm2_propagation"], "current_value_state": "unknown", "raw_data_path": None, "derived_data_path": None, "ready": False},
    ],
}


FUSED_SILICA_TEMPLATE = {
    "material_profile_id": "fused_silica_evidence_template",
    "status": "template_no_numeric_claims",
    "material": "fused silica",
    "all_values_unknown_until_evidence_attached": True,
    "properties": {
        "supplier": {"value": None, "unit": None, "evidence_status": "unknown"},
        "grade": {"value": None, "unit": None, "evidence_status": "unknown"},
        "thickness_mm": {"value": None, "unit": "mm", "evidence_status": "unknown"},
        "surface_polish": {"value": None, "unit": None, "evidence_status": "unknown"},
        "refractive_index_at_1030_nm": {"value": None, "unit": None, "evidence_status": "needs_verified_source"},
        "modification_threshold": {"value": None, "unit": "J/cm^2", "evidence_status": "needs_bench_measurement"},
        "etch_selectivity": {"value": None, "unit": None, "evidence_status": "needs_bench_measurement"},
        "waveguide_loss": {"value": None, "unit": "dB/cm", "evidence_status": "needs_bench_measurement"},
    },
    "forbidden_transfer": "Do not copy CrZnSe proxy values into this profile.",
}


def build_project_claim_registry() -> dict[str, Any]:
    return {
        "stage": STAGE,
        "purpose": "code-to-evidence audit and claim-boundary register",
        "claim_boundary": "audit only; no new optical propagation, no physical 4F, no camera model, no inverse correction, no AI, no material response",
        "active_route_components": list(ACTIVE_ROUTE_COMPONENTS),
        "code_inventory": CODE_INVENTORY,
        "claims": CLAIMS,
    }


def build_research_backlog() -> dict[str, Any]:
    return {
        "stage": STAGE,
        "status": "audit_backlog_not_started",
        "priority_vocabulary": ["P0", "P1", "P2", "P3"],
        "time_horizon_vocabulary": ["immediate", "pre_4f_model", "pre_inverse_correction", "pre_fused_silica_pilot", "later_research"],
        "items": RESEARCH_BACKLOG,
    }


def build_literature_search_plan() -> dict[str, Any]:
    return {
        "stage": STAGE,
        "status": "search_plan_no_fabricated_citations",
        "entries": LITERATURE_SEARCH_PLAN,
    }


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _counts() -> dict[str, int]:
    return {
        "canonical_active_claims": sum(1 for c in CLAIMS if c["route_relevance"] == "canonical_active"),
        "placeholders": sum(1 for c in CLAIMS if c["evidence_status"] == "assumption_declared" or c["physics_status"] == "diagnostic_placeholder"),
        "manufacturer_data_blockers": sum(1 for c in CLAIMS if c["manufacturer_evidence_required"]),
        "bench_data_blockers": sum(1 for c in CLAIMS if c["bench_evidence_required"]),
        "literature_blockers": sum(1 for c in CLAIMS if c["evidence_status"] == "needs_verified_source"),
    }


def make_markdown_doc() -> str:
    counts = _counts()
    inv_rows = "\n".join(
        f"| `{i['item_id']}` | `{i['path']}` | {i['classification']} | {i['current_route_or_study_relevance']} | {i['affects_current_canonical_execution']} | {i['affects_current_laboratory_decisions']} |"
        for i in CODE_INVENTORY
    )
    claim_rows = "\n".join(
        f"| `{c['claim_id']}` | {c['current_status']} | {c['physics_status']} | {c['evidence_status']} | {', '.join(c['bench_evidence_required'][:2]) or ', '.join(c['manufacturer_evidence_required'][:2]) or 'literature/source verification'} |"
        for c in CLAIMS
    )
    backlog_rows = "\n".join(
        f"| `{b['research_id']}` | {b['priority']} | {b['time_horizon']} | {b['title']} | {b['current_gap']} | {b['expected_deliverable']} |"
        for b in RESEARCH_BACKLOG
    )
    legacy = "\n".join(
        f"- `{i['item_id']}` ({i['classification']}): {i['path']} -- {i['current_route_or_study_relevance']}"
        for i in CODE_INVENTORY
        if i["classification"] in {"legacy_retained", "experimental_development", "deprecated_do_not_extend"}
    )
    return f"""# Stage 9A.2 Code-to-Evidence Audit

This audit maps the current repository from implemented code to claim boundary,
evidence need, and next research or measurement task.  It does not add optical
propagation, 4F modelling, camera modelling, inverse correction, neural
networks, or material-response physics.

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

## Counts

- Canonical active claims: {counts['canonical_active_claims']}
- Placeholder/assumption claims: {counts['placeholders']}
- Manufacturer-data blockers: {counts['manufacturer_data_blockers']}
- Bench-data blockers: {counts['bench_data_blockers']}
- Literature/source blockers: {counts['literature_blockers']}

## Code And Claim Inventory

| item | path | classification | relevance | affects execution | affects lab decisions |
|---|---|---:|---|---:|---:|
{inv_rows}

## Physical And Numerical Claim Register

Full structured records are stored in
`configs/evidence/project_claim_registry.json`.

| claim ID | current status | physics status | evidence status | next action |
|---|---|---|---|---|
{claim_rows}

## Research Backlog

Full structured records are stored in `configs/evidence/research_backlog.json`.

| research ID | priority | horizon | title | current gap | deliverable |
|---|---|---|---|---|---|
{backlog_rows}

## Fused-Silica / Cr:ZnSe Separation

The immediate scalar CSLM -> future 4F -> physical-axicon calibration campaign
is material-neutral until the fused-silica application profile is populated from
verified literature and bench evidence.  The legacy Cr:ZnSe branch is retained
for historical planning context only.

This branch contains CrZnSe-specific proxy assumptions and is not validated for
fused-silica TGV, waveguide, welding, or modification predictions.

- Material-neutral current path: `vbb_study/digital_twin/cslm_route.py`,
  `bench_inventory.py`, `calibration_acquisition.py`,
  `slm_calibration_masks.py`, and `measured_image_metrics.py`.
- Fused-silica template: `configs/materials/fused_silica_evidence_template.json`
  with null/unknown values only.
- Cr:ZnSe-specific or material-proxy paths: `vbb_study/config.py`,
  `vbb_study/equations/materials.py`, `vbb_study/publication/materials.py`,
  `notebooks/materials/*.ipynb`, `docs/03_materials_application.md`.
- These must not be used for fused-silica decision-making until replaced by
  fused-silica evidence and bench validation.

## Legacy Branches Retained But Excluded From Immediate Lab Path

{legacy}

## Evidence Registers

- Literature search plan:
  `configs/evidence/literature_search_plan.json`
- Manufacturer evidence register:
  `configs/evidence/manufacturer_evidence_register.json`
- Bench evidence register:
  `configs/evidence/bench_evidence_register.json`
- Bibliography placeholder:
  `references/structured_beam_methods.bib`

No BibTeX entry is added until DOI/publisher metadata is verified from the
actual source.
"""


def make_summary() -> str:
    counts = _counts()
    priorities = {p: sum(1 for b in RESEARCH_BACKLOG if b["priority"] == p) for p in ("P0", "P1", "P2", "P3")}
    return f"""# Stage 9A.2 Code-to-Evidence Audit Summary

Starting checkpoint: Stage 9A.1 first Fourier carrier calibration session pack (`51071bc`).

Stage 9A.2 creates an evidence-aware project map and research backlog.  No new
optical propagation, calibration fitting, correction, AI, camera model, or
material-response physics is implemented.

## Created

- `docs/44_code_to_evidence_audit.md`
- `configs/evidence/project_claim_registry.json`
- `configs/evidence/research_backlog.json`
- `configs/evidence/literature_search_plan.json`
- `configs/evidence/manufacturer_evidence_register.json`
- `configs/evidence/bench_evidence_register.json`
- `configs/materials/fused_silica_evidence_template.json`
- `references/README.md`
- `references/structured_beam_methods.bib`
- `outputs/figures/digital_twin/stage9a2_code_to_evidence_roadmap.png`
- `tests/test_stage9a2_code_to_evidence_audit.py`

## Claim Counts

- Canonical active claims: {counts['canonical_active_claims']}
- Placeholder/assumption claims: {counts['placeholders']}
- Manufacturer-data blockers: {counts['manufacturer_data_blockers']}
- Bench-data blockers: {counts['bench_data_blockers']}
- Literature/source blockers: {counts['literature_blockers']}

## Backlog Counts

- P0: {priorities['P0']}
- P1: {priorities['P1']}
- P2: {priorities['P2']}
- P3 including legacy optional branches: {priorities['P3']}

## Immediate Lab Action

Run the first Fourier-plane carrier calibration session from Stage 9A.1:
record actual SLM/camera/lens/stop/axicon identifiers, capture dark and flat
references, then measure SLM2 command-domain carrier cycles versus observed
Fourier-plane order position without changing the bench mid-run.
"""


def make_references_readme() -> str:
    return """# References For Structured-Beam Methods

This directory stores verified bibliography records only.

Acceptable references:

- peer-reviewed primary research papers for methods, propagation models,
  phase retrieval, SLM diffraction, axicon tolerance, and material processing;
- authoritative textbooks or standards for Fourier optics and sampling
  conventions;
- manufacturer datasheets or application notes for component properties.

Do not add a BibTeX entry until DOI, title, authors, journal/publisher, year,
and source URL or PDF metadata have been checked against the actual source.
Review articles may guide searches, but claim support should point to primary
literature whenever possible.  Manufacturer specifications and bench evidence
are stored separately in `configs/evidence/manufacturer_evidence_register.json`
and `configs/evidence/bench_evidence_register.json`.

Each accepted reference must list the claim IDs it supports.  A reference can
support a physical principle without validating this repository's numerical
implementation, the real bench alignment, or fused-silica process outcome.
"""


def make_empty_bib() -> str:
    return "% Stage 9A.2 placeholder: no verified bibliography records yet.\n"


def plot_roadmap(path: str | Path = "outputs/figures/digital_twin/stage9a2_code_to_evidence_roadmap.png") -> Path:
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.axis("off")
    fig.suptitle("Stage 9A.2 Code-to-Evidence Roadmap\nAudit only: no physical 4F, camera model, correction engine, AI, or material model active", fontsize=14)

    boxes = [
        (0.04, 0.68, 0.18, 0.16, "ACTIVE\nCSLM diagnostic\nSLM1 vortex\nSLM2 correction/carrier\nfree-space n=1.0", "#d7f0ff"),
        (0.29, 0.68, 0.18, 0.16, "FUTURE\nphysical 4F filter\nneeds lens/stop/order\nbench evidence", "#fff1c9"),
        (0.54, 0.68, 0.18, 0.16, "BENCHMARK\nphysical axicon\nideal selected-order only\nnot experimental prediction", "#dff3df"),
        (0.79, 0.68, 0.17, 0.16, "MEASUREMENT\ncamera z-stack\npixel metrics now\nphysical metrics blocked", "#f1e4ff"),
        (0.04, 0.34, 0.27, 0.18, "Evidence lanes\nLiterature: principle/source support\nManufacturer: SLM/lens/camera/axicon specs\nBench: carrier mapping, stop, z-stacks", "#f7f7f7"),
        (0.37, 0.34, 0.27, 0.18, "Backlog\nP0 immediate bench blockers\nP1 optical/numerical validation\nP2 inverse/correction research\nP3 fused-silica application", "#f7f7f7"),
        (0.70, 0.34, 0.26, 0.18, "Boundaries\nFused silica template has unknown/null values\nCrZnSe proxies retained as legacy\nNo material decision from legacy branch", "#ffe5e5"),
    ]
    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333", lw=1.2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", transform=ax.transAxes, fontsize=9)
    for x0, x1 in ((0.22, 0.29), (0.47, 0.54), (0.72, 0.79)):
        ax.annotate("", xy=(x1, 0.76), xytext=(x0, 0.76), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    ax.text(0.04, 0.17,
            "Claim rule: literature supports principle != numerical validation != bench calibration != fused-silica process demonstration.",
            transform=ax.transAxes, fontsize=11, weight="bold")
    ax.text(0.04, 0.10,
            "Next physical lab action: run the Stage 9A.1 Fourier-plane carrier sweep and record actual hardware/spec/bench metadata.",
            transform=ax.transAxes, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_stage9a2_artifacts(root: str | Path = ".") -> dict[str, Path]:
    root = Path(root)
    outputs = {
        "registry": root / "configs/evidence/project_claim_registry.json",
        "backlog": root / "configs/evidence/research_backlog.json",
        "search_plan": root / "configs/evidence/literature_search_plan.json",
        "manufacturer": root / "configs/evidence/manufacturer_evidence_register.json",
        "bench": root / "configs/evidence/bench_evidence_register.json",
        "fused_silica_template": root / "configs/materials/fused_silica_evidence_template.json",
        "doc": root / "docs/44_code_to_evidence_audit.md",
        "summary": root / "STAGE9A2_CODE_TO_EVIDENCE_AUDIT_SUMMARY.md",
        "references_readme": root / "references/README.md",
        "bib": root / "references/structured_beam_methods.bib",
        "figure": root / "outputs/figures/digital_twin/stage9a2_code_to_evidence_roadmap.png",
    }
    _write_json(outputs["registry"], build_project_claim_registry())
    _write_json(outputs["backlog"], build_research_backlog())
    _write_json(outputs["search_plan"], build_literature_search_plan())
    _write_json(outputs["manufacturer"], MANUFACTURER_EVIDENCE_REGISTER)
    _write_json(outputs["bench"], BENCH_EVIDENCE_REGISTER)
    _write_json(outputs["fused_silica_template"], FUSED_SILICA_TEMPLATE)
    outputs["doc"].parent.mkdir(parents=True, exist_ok=True)
    outputs["doc"].write_text(make_markdown_doc(), encoding="utf-8")
    outputs["summary"].write_text(make_summary(), encoding="utf-8")
    outputs["references_readme"].parent.mkdir(parents=True, exist_ok=True)
    outputs["references_readme"].write_text(make_references_readme(), encoding="utf-8")
    outputs["bib"].write_text(make_empty_bib(), encoding="utf-8")
    plot_roadmap(outputs["figure"])
    return outputs


if __name__ == "__main__":
    paths = write_stage9a2_artifacts(Path.cwd())
    for key, value in paths.items():
        print(f"{key}: {value}")
