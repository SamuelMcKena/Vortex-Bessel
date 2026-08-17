"""Full technical report and evidence pack for the Nathan source-scale branch."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import textwrap
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vbb_study.digital_twin.nathan_mode2u2_fix_strict_hexagon_optimisation import (
    OLD_BEST_COMPROMISE_ID,
)
from vbb_study.digital_twin.nathan_mode2u3_hardware_closure import (
    CANONICAL_OPERATING_POINT_ID,
    STRICT_COMPROMISE_ID,
)

REPORT_ROOT = Path("report")
REPORT_TEX = REPORT_ROOT / "nathan_hexagonal_bessel_full_report.tex"
REPORT_PDF = REPORT_ROOT / "nathan_hexagonal_bessel_full_report.pdf"
REPORT_README = REPORT_ROOT / "README.md"
REFERENCES_BIB = REPORT_ROOT / "references.bib"
EVIDENCE_ROOT = REPORT_ROOT / "evidence_pack"
EQUATION_REGISTRY = REPORT_ROOT / "equation_registry.csv"
CLAIM_REGISTRY = REPORT_ROOT / "claim_registry.csv"
NUMBER_REGISTRY = REPORT_ROOT / "number_registry.csv"
FIGURE_MANIFEST = REPORT_ROOT / "figure_manifest.csv"
SUPERSEDED_TABLE = REPORT_ROOT / "superseded_material.csv"
BUILD_SUMMARY = REPORT_ROOT / "build_report_summary.json"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [dict(row) for row in rows]
    fields: list[str] = []
    for row in data:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in data:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_ready(value), sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _latex_escape(text: Any) -> str:
    s = str(text)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def _tex_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def _engine_available() -> str | None:
    for candidate in ("latexmk", "pdflatex", "xelatex", "lualatex", "tectonic"):
        if shutil.which(candidate):
            return candidate
    return None


def equation_rows() -> list[dict[str, Any]]:
    return [
        {
            "equation_id": "EQ01",
            "name": "Gaussian source amplitude",
            "latex": r"A(r)=\exp(-r^2/w_0^2)",
            "meaning": "Defines the source-scale scalar envelope at the axicon input.",
            "variables": "r transverse radius; w0 1/e field radius",
            "units": "r,w0 in m",
            "code": "vbb_study/digital_twin/nathan_vector_hexagon.py::mode2n_source_target",
            "stage": "V0/M2N",
            "assumptions": "Axis-sampled Nathan source grid; source-scale beam radius.",
            "validation_status": "validated",
        },
        {
            "equation_id": "EQ02",
            "name": "Angular coordinate",
            "latex": r"\theta=\operatorname{atan2}(y,x)",
            "meaning": "Defines the local transverse azimuth used by the six-sector field.",
            "variables": "x,y transverse coordinates",
            "units": "rad",
            "code": "nathan_alpha_map / source_parity_grid",
            "stage": "V0/M2P",
            "assumptions": "Lab x/y coordinate frame.",
            "validation_status": "validated",
        },
        {
            "equation_id": "EQ03",
            "name": "Six-sector orientation angle",
            "latex": r"\alpha(\theta)=\theta+\phi_0(\theta),\quad \phi_0\in\{0,\pi/2\}",
            "meaning": "Alternates radial and azimuthal vector sectors.",
            "variables": "theta; phi0 sector offset",
            "units": "rad",
            "code": "nathan_alpha_map",
            "stage": "V0/M2P",
            "assumptions": "Three radial/azimuthal sector pairs.",
            "validation_status": "validated",
        },
        {
            "equation_id": "EQ04",
            "name": "Target Jones vector",
            "latex": r"\mathbf E_\mathrm{target}=A(r)[\cos\alpha,\ \sin\alpha]^T",
            "meaning": "The required transverse vector field before the axicon.",
            "variables": "A alpha",
            "units": "field amplitude",
            "code": "mode2p_target_arrays; mode2n_source_target",
            "stage": "M2P/M2N",
            "assumptions": "Linear x/y Jones basis.",
            "validation_status": "validated",
        },
        {
            "equation_id": "EQ05",
            "name": "Stokes parameters",
            "latex": r"S_0=|E_x|^2+|E_y|^2,\ S_1=|E_x|^2-|E_y|^2,\ S_2=2\Re(E_xE_y^*),\ S_3=-2\Im(E_xE_y^*)",
            "meaning": "Observable polarization diagnostics used to interpret the target field.",
            "variables": "Ex Ey",
            "units": "relative intensity",
            "code": "VectorField.stokes; stokes_from_linear_components",
            "stage": "M2P/M2W-FIX",
            "assumptions": "Project sign convention for S3.",
            "validation_status": "validated",
        },
        {
            "equation_id": "EQ06",
            "name": "HWP synthesis",
            "latex": r"J_\mathrm{HWP}(\beta)[1,0]^T=[\cos(2\beta),\sin(2\beta)]^T,\quad \beta=\alpha/2",
            "meaning": "A patterned HWP maps horizontal input to the target vector field.",
            "variables": "beta fast-axis angle",
            "units": "rad",
            "code": "route_patterned_hwp_ideal",
            "stage": "M2P/M2N",
            "assumptions": "Ideal lossless half-wave retardance.",
            "validation_status": "validated in simulation",
        },
        {
            "equation_id": "EQ07",
            "name": "QWP matrix",
            "latex": r"J(\delta,\psi)=R(-\psi)\operatorname{diag}(e^{-i\delta/2},e^{i\delta/2})R(\psi)",
            "meaning": "Uniform retarder convention used for the final QWP.",
            "variables": "delta retardance; psi axis angle",
            "units": "rad",
            "code": "linear_retarder; apply_uniform_jones",
            "stage": "M2P/M2N/M2W-FIX",
            "assumptions": "Ideal retarder until lab calibration.",
            "validation_status": "model-supported; mount sign calibration-required",
        },
        {
            "equation_id": "EQ08",
            "name": "Dual-SLM phase convention",
            "latex": r"\phi_H=+\alpha,\quad \phi_V=-\alpha+\pi/2",
            "meaning": "Phase masks recovered by ideal synthesis and inverse propagation.",
            "variables": "alpha",
            "units": "rad",
            "code": "route_dual_slm_linear_then_qwp_ideal; run_mode2q_backward_initialisation",
            "stage": "M2P/M2Q",
            "assumptions": "Carrier omitted; common piston handled separately.",
            "validation_status": "validated",
        },
        {
            "equation_id": "EQ09",
            "name": "Sequential SLM chain",
            "latex": r"E_4=A/\sqrt2[e^{i\phi_H},e^{i\phi_V}]^T",
            "meaning": "Collinear two-SLM route reproduces the abstract H/V channel field.",
            "variables": "phiH phiV A",
            "units": "field amplitude",
            "code": "sequential_jones_equivalence",
            "stage": "M2W-FIX",
            "assumptions": "Selective phase-only modulation; swap HWPs unitary when used.",
            "validation_status": "validated in simulation",
        },
        {
            "equation_id": "EQ10",
            "name": "Axicon transverse wavevector",
            "latex": r"k_r=(2\pi/\lambda)(n_\mathrm{axicon}-n_m)\tan\gamma",
            "meaning": "Source-scale axicon sets the radial Bessel fringe period.",
            "variables": "lambda n gamma",
            "units": "rad/m",
            "code": "NathanSourceParityConfig.k_r_m_inv",
            "stage": "V0/M2N",
            "assumptions": "Project tangent convention; source-scale axicon.",
            "validation_status": "validated against source branch",
        },
        {
            "equation_id": "EQ11",
            "name": "Angular spectrum propagation",
            "latex": r"\tilde{\mathbf E}(k_x,k_y,z)=\tilde{\mathbf E}(k_x,k_y,0)e^{ik_z z},\quad k_z=\sqrt{k^2-k_x^2-k_y^2}",
            "meaning": "Free-space propagation of the vector field after the axicon.",
            "variables": "kx ky kz z",
            "units": "rad/m and m",
            "code": "propagate_vector_asm",
            "stage": "V0/M2N/M2Q",
            "assumptions": "Propagating and decaying evanescent components; FFT grid sampling.",
            "validation_status": "validated in simulation",
        },
        {
            "equation_id": "EQ12",
            "name": "Inverse propagation",
            "latex": r"\tilde{\mathbf E}(k_x,k_y,0)=\tilde{\mathbf E}(k_x,k_y,z)e^{-ik_z z}",
            "meaning": "Backward target reconstruction used by M2Q.",
            "variables": "kx ky kz z",
            "units": "rad/m and m",
            "code": "mode2q_backpropagate_vector",
            "stage": "M2Q",
            "assumptions": "No evanescent amplification; propagating modes only.",
            "validation_status": "validated in simulation",
        },
        {
            "equation_id": "EQ13",
            "name": "4F carrier hologram",
            "latex": r"\phi_\mathrm{display}=\phi_\mathrm{target}+2\pi\nu x",
            "meaning": "Linear carrier shifts the desired order in the Fourier plane.",
            "variables": "nu carrier frequency; x panel coordinate",
            "units": "cycles/m and m",
            "code": "run_mode2n_dual_slm_4f_route; mode2s_apply_4f",
            "stage": "M2N/M2S",
            "assumptions": "Shared carrier sign; common 4F iris.",
            "validation_status": "validated in simulation",
        },
        {
            "equation_id": "EQ14",
            "name": "Fourier-order displacement",
            "latex": r"x_{+1}=\lambda f\nu",
            "meaning": "Physical +1 order offset at the 4F Fourier plane.",
            "variables": "lambda f nu",
            "units": "m",
            "code": "physical_4f_rows; fourf_final_design",
            "stage": "M2U3/M2V",
            "assumptions": "Paraxial grating relation.",
            "validation_status": "model-supported; bench calibration-required",
        },
        {
            "equation_id": "EQ15",
            "name": "Power observable",
            "latex": r"I=|E_x|^2+|E_y|^2+|E_z|^2",
            "meaning": "Total intensity compared to the V0 reference after vector propagation.",
            "variables": "Ex Ey Ez",
            "units": "relative intensity",
            "code": "VectorField.intensity",
            "stage": "V0/M2N/M2S",
            "assumptions": "Detector insensitive to vector component phase.",
            "validation_status": "validated simulation observable",
        },
        {
            "equation_id": "EQ16",
            "name": "Ring-count relation",
            "latex": r"N_\mathrm{rings}=R k_r/(2\pi)",
            "meaning": "Plane-labelled estimate of effective ring count.",
            "variables": "R radius; kr transverse wavevector",
            "units": "dimensionless",
            "code": "effective_ring_count_for_plane",
            "stage": "M1B/M2U",
            "assumptions": "Radius and kr must refer to the same physical plane.",
            "validation_status": "audited after plane-mixing correction",
        },
        {
            "equation_id": "EQ17",
            "name": "Strict symmetry diagnostics",
            "latex": r"\{c_{60},c_{90},c_{120},h_3,h_4,h_6,D_\mathrm{core}\}",
            "meaning": "Classifier features used to reject triangular and fourfold false positives.",
            "variables": "rotational correlations; angular harmonics; dark-core ratio",
            "units": "dimensionless",
            "code": "mode2q_strict_hexagon_gate; evaluate_strict_hexagon_metrics",
            "stage": "M2U2-FIX",
            "assumptions": "Project-calibrated thresholds, not universal definitions.",
            "validation_status": "validated against truth table",
        },
    ]


def claim_rows() -> list[dict[str, Any]]:
    return [
        {
            "claim_id": "CL01",
            "claim": "V0 literal source and project implementation reproduce the same source-scale field.",
            "evidence_files": "outputs/figures/digital_twin/nathan_visual_ladder/nathan_visual_ladder_v0_reference_vs_reproduction.png",
            "data_source": "nathan_visual_ladder_v0_status_report.json",
            "figure_or_table": "F01",
            "numerical_result": "source parity validated; Nathan axis-sampled grid required",
            "status": "validated",
            "limitations": "Simulation parity, not an experimental measurement.",
        },
        {
            "claim_id": "CL02",
            "claim": "The ideal patterned-HWP and dual-SLM/QWP Jones routes reproduce the target vector field.",
            "evidence_files": "outputs/figures/digital_twin/nathan_mode2_preflight_jones/jones_synthesis_summary.csv",
            "data_source": "M2P preflight",
            "figure_or_table": "F04",
            "numerical_result": "component overlaps equal to 1 within numerical precision",
            "status": "validated",
            "limitations": "Ideal component model before carrier, 4F and axicon propagation.",
        },
        {
            "claim_id": "CL03",
            "claim": "The realistic source-scale dual-SLM + carrier + common 4F route preserves the strict hexagon.",
            "evidence_files": "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/03_ideal_vs_realistic/mode2w_fix_ideal_vs_realistic_metrics.csv",
            "data_source": "M2W-FIX",
            "figure_or_table": "F07",
            "numerical_result": "realistic sequential z60 correlation approx 0.9936209493; strict class visual_hexagonal_field",
            "status": "validated in simulation",
            "limitations": "Experimental replication remains unverified.",
        },
        {
            "claim_id": "CL04",
            "claim": "M2Q inverse propagation independently recovers the same H/V mask convention.",
            "evidence_files": "outputs/figures/digital_twin/nathan_mode2q_backward_mask_synthesis/mode2q_backward_diagnostics.json",
            "data_source": "M2Q",
            "figure_or_table": "F06",
            "numerical_result": "complex vector overlap approx 0.99999997; alpha RMS approx 1.1e-4 rad",
            "status": "validated in simulation",
            "limitations": "Inverse uses the simulated complex V0 target.",
        },
        {
            "claim_id": "CL05",
            "claim": "The source-scale system is sensitive primarily to hologram/mask centre versus axicon-axis registration.",
            "evidence_files": "outputs/figures/digital_twin/nathan_mode2s_lab_realism_tolerance/mode2s_compensation_results.csv",
            "data_source": "M2S",
            "figure_or_table": "F10",
            "numerical_result": "0.5 mm decentre recovered from corr approx 0.7888 to approx 0.9762",
            "status": "model-supported",
            "limitations": "Finite perturbation set; bench correction still required.",
        },
        {
            "claim_id": "CL06",
            "claim": "N=1536 publication-selected outputs are used for the primary hero comparison where available.",
            "evidence_files": "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/01_source_audit/mode2w_fix_numerical_source_audit.csv",
            "data_source": "M2U/M2W-FIX source audit",
            "figure_or_table": "F07",
            "numerical_result": "Fig. 3A native N=1536; SAS zoom dx approx 3.087 um",
            "status": "validated",
            "limitations": "Some broad propagation context remains N=1024 by design.",
        },
        {
            "claim_id": "CL07",
            "claim": "The old M2U2 best-compromise optimiser candidate is forbidden.",
            "evidence_files": "outputs/figures/digital_twin/nathan_mode2u2_fix_strict_hexagon_optimisation/old_optima_strict_audit.csv",
            "data_source": "M2U2-FIX",
            "figure_or_table": "S01",
            "numerical_result": OLD_BEST_COMPROMISE_ID,
            "status": "forbidden",
            "limitations": "Forbidden within this project branch and gate definition.",
        },
        {
            "claim_id": "CL08",
            "claim": "The original MODE 2W split-arm presentation is superseded and must not be used as canonical architecture.",
            "evidence_files": "docs/82_nathan_mode2w_annotated_master_figure_pack.md; docs/83_nathan_mode2w_fix_sequential_architecture.md",
            "data_source": "M2W-FIX",
            "figure_or_table": "S02",
            "numerical_result": "M2W superseded; M2WF-A accepted",
            "status": "superseded",
            "limitations": "Historical split-arm files retained only as provenance.",
        },
        {
            "claim_id": "CL09",
            "claim": "Exact SLM phase stroke and per-panel LUT remain calibration-required.",
            "evidence_files": "docs/75_nathan_mode2u3_slm_phase_calibration.md",
            "data_source": "M2U3 hardware closure",
            "figure_or_table": "T03",
            "numerical_result": "phase stroke unresolved at 1029 nm",
            "status": "calibration-required",
            "limitations": "Bench calibration required before hardware masks are final.",
        },
        {
            "claim_id": "CL10",
            "claim": "Microfabrication/sample-plane success is not claimed by this report.",
            "evidence_files": "docs/83_nathan_mode2w_fix_sequential_architecture.md",
            "data_source": "M2W-FIX scope statement",
            "figure_or_table": "conclusion",
            "numerical_result": "no microfabrication/sample-plane success claim",
            "status": "validated scope boundary",
            "limitations": "Separate branch remains unresolved.",
        },
    ]


def number_rows() -> list[dict[str, Any]]:
    return [
        {"number_id": "N001", "quantity": "Physical bench wavelength", "value": 1029, "units": "nm", "scope": "M2U3/M2V/M2W-FIX", "provenance": "PHAROS / Digital Twin physical binding", "status": "resolved", "source_file": "docs/78_nathan_mode2u3_final_hardware_closure.md"},
        {"number_id": "N002", "quantity": "Nathan parity wavelength", "value": 1030, "units": "nm", "scope": "V0 source parity", "provenance": "Nathan source rounding", "status": "different scope", "source_file": "docs/78_nathan_mode2u3_final_hardware_closure.md"},
        {"number_id": "N003", "quantity": "Source beam radius", "value": 2.0, "units": "mm", "scope": "source scale", "provenance": "NathanSourceParityConfig.beam_radius_m", "status": "validated", "source_file": "vbb_study/digital_twin/nathan_vector_hexagon.py"},
        {"number_id": "N004", "quantity": "SLM panel width", "value": 1920, "units": "px", "scope": "native panel", "provenance": "PLUTO-2.1 family/project config", "status": "repository-confirmed geometry", "source_file": "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/02_target_masks/mode2w_fix_slm_mask_metadata.json"},
        {"number_id": "N005", "quantity": "SLM panel height", "value": 1080, "units": "px", "scope": "native panel", "provenance": "PLUTO-2.1 family/project config", "status": "repository-confirmed geometry", "source_file": "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/02_target_masks/mode2w_fix_slm_mask_metadata.json"},
        {"number_id": "N006", "quantity": "SLM pitch", "value": 8, "units": "um", "scope": "native panel", "provenance": "project config/manual family page", "status": "repository-confirmed geometry", "source_file": "docs/78_nathan_mode2u3_final_hardware_closure.md"},
        {"number_id": "N007", "quantity": "SLM active area", "value": "15.36 x 8.64", "units": "mm", "scope": "native panel", "provenance": "1920x1080 at 8 um", "status": "repository-confirmed geometry", "source_file": "docs/78_nathan_mode2u3_final_hardware_closure.md"},
        {"number_id": "N008", "quantity": "Carrier frequency", "value": 6.25, "units": "lp/mm", "scope": "source-scale display carrier", "provenance": "M2N/M2Q/M2S/M2U validated runs", "status": "validated", "source_file": "docs/78_nathan_mode2u3_final_hardware_closure.md"},
        {"number_id": "N009", "quantity": "Carrier period", "value": 20, "units": "SLM px", "scope": "8 um panel", "provenance": "6.25 lp/mm", "status": "validated", "source_file": "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/02_target_masks/mode2w_fix_slm_mask_metadata.json"},
        {"number_id": "N010", "quantity": "4F focal length", "value": 300, "units": "mm", "scope": "nominal bench", "provenance": "M2U3 F300 binding", "status": "routine bench confirmation", "source_file": "docs/78_nathan_mode2u3_final_hardware_closure.md"},
        {"number_id": "N011", "quantity": "+1 order displacement", "value": 1.929, "units": "mm", "scope": "1029 nm, f=300 mm, 6.25 lp/mm", "provenance": "x=lambda f nu", "status": "model-supported", "source_file": "docs/78_nathan_mode2u3_final_hardware_closure.md"},
        {"number_id": "N012", "quantity": "Iris diameter", "value": 1.54, "units": "mm", "scope": "0.40 carrier separation radius", "provenance": "M2U3 closure", "status": "routine bench calibration", "source_file": "docs/78_nathan_mode2u3_final_hardware_closure.md"},
        {"number_id": "N013", "quantity": "Axicon base angle", "value": 2, "units": "deg", "scope": "source scale", "provenance": "Nathan fused-silica source branch", "status": "validated", "source_file": "docs/78_nathan_mode2u3_final_hardware_closure.md"},
        {"number_id": "N014", "quantity": "Axicon refractive index", "value": 1.458, "units": "dimensionless", "scope": "source scale", "provenance": "Nathan fused silica", "status": "validated", "source_file": "docs/78_nathan_mode2u3_final_hardware_closure.md"},
        {"number_id": "N015", "quantity": "Reference z", "value": 60, "units": "mm", "scope": "source-scale Bessel zone", "provenance": "NathanSourceParityConfig.z_reference_m", "status": "validated", "source_file": "vbb_study/digital_twin/nathan_vector_hexagon.py"},
        {"number_id": "N016", "quantity": "Publication grid", "value": 1536, "units": "samples per side", "scope": "hero/high-resolution comparison", "provenance": "M2U publication sampling audit", "status": "publication-selected", "source_file": "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/01_source_audit/mode2w_fix_numerical_source_audit.csv"},
        {"number_id": "N017", "quantity": "N=1536 samples per radial fringe", "value": 9.89, "units": "samples/fringe", "scope": "native full 10 mm grid", "provenance": "M2U/M2W-FIX source audit", "status": "validated", "source_file": "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/01_source_audit/mode2w_fix_numerical_source_audit.csv"},
        {"number_id": "N018", "quantity": "Fig. 3A SAS zoom sampling", "value": 20.84, "units": "samples/fringe", "scope": "displayed focus crop", "provenance": "M2W-FIX SAS render audit", "status": "display provenance", "source_file": "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/01_source_audit/mode2w_fix_numerical_source_audit.csv"},
        {"number_id": "N019", "quantity": "Realistic sequential z60 correlation", "value": 0.9936209493, "units": "correlation", "scope": "M2W-FIX", "provenance": "sequential equivalence report", "status": "validated in simulation", "source_file": "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/10_final_status/m2wf_outcome_report.json"},
        {"number_id": "N020", "quantity": "Sequential pre-QWP overlap", "value": 1.0, "units": "overlap", "scope": "M2W-FIX", "provenance": "sequential_jones_equivalence", "status": "validated in simulation", "source_file": "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/00_architecture/mode2w_fix_sequential_equivalence.json"},
        {"number_id": "N021", "quantity": "Sequential post-QWP overlap", "value": 1.0, "units": "overlap", "scope": "M2W-FIX", "provenance": "sequential_jones_equivalence", "status": "validated in simulation", "source_file": "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/00_architecture/mode2w_fix_sequential_equivalence.json"},
        {"number_id": "N022", "quantity": "First-order efficiency", "value": 0.949, "units": "fraction", "scope": "realistic 4F route", "provenance": "M2N/M2W-FIX realistic route", "status": "validated in simulation", "source_file": "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/03_ideal_vs_realistic/mode2w_fix_ideal_vs_realistic_metrics.csv"},
        {"number_id": "N023", "quantity": "0.5 mm compensated axicon decentre correlation", "value": 0.9762, "units": "correlation", "scope": "M2S correction", "provenance": "mode2s compensation", "status": "model-supported", "source_file": "outputs/figures/digital_twin/nathan_mode2s_lab_realism_tolerance/mode2s_compensation_results.csv"},
        {"number_id": "N024", "quantity": "Old forbidden candidate full-field correlation", "value": 0.989, "units": "correlation", "scope": "superseded M2U2 optimum", "provenance": "M2U2-FIX audit", "status": "forbidden despite high correlation", "source_file": "outputs/figures/digital_twin/nathan_mode2u2_fix_strict_hexagon_optimisation/old_optima_strict_audit.csv"},
    ]


def figure_rows() -> list[dict[str, Any]]:
    return [
        _fig("F01", "V0 source parity and reference/reproduction comparison.", "outputs/figures/digital_twin/nathan_visual_ladder/nathan_visual_ladder_v0_reference_vs_reproduction.png", "V0", 1024, "10 mm", "", "display", True, "validated", "source_validation", True),
        _fig("F02", "Grid convention audit showing why Nathan axis-sampled centring is required.", "outputs/figures/digital_twin/nathan_visual_ladder/nathan_source_observable_audit_grid_convention.png", "V0", 1024, "10 mm", "", "display", True, "validated", "source_validation", False),
        _fig("F03", "High-resolution V0 reference crop and propagated structure.", "outputs/figures/digital_twin/nathan_mode2u_master_highres_audit/00_v0_reference/v0_reference_crop_highres.png", "M2U", 1536, "10 mm", "z=60 mm", "lanczos", True, "validated", "source_validation", True),
        _fig("F04", "M2P target alpha and sector map.", "outputs/figures/digital_twin/nathan_mode2u_master_highres_audit/01_m2p_preaxicon/m2p_target_alpha_sector_highres.png", "M2P/M2U", 1536, "10 mm", "pre-axicon", "lanczos", True, "validated", "equations", True),
        _fig("F05", "M2P pre-axicon Stokes comparison.", "outputs/figures/digital_twin/nathan_mode2u_master_highres_audit/01_m2p_preaxicon/m2p_stokes_comparison_highres.png", "M2P/M2U", 1536, "10 mm", "pre-axicon", "lanczos", True, "validated", "equations", False),
        _fig("F06", "M2Q recovered phase-only masks and inverse design diagnostics.", "outputs/figures/digital_twin/nathan_mode2u_master_highres_audit/03_m2q_inverse_masks/m2q_phase_only_masks_highres.png", "M2Q/M2U", 1536, "10 mm", "pre-axicon", "lanczos", True, "validated", "inverse_design", True),
        _fig("F07", "M2W-FIX V0 versus ideal sequential versus realistic sequential comparison.", "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/03_ideal_vs_realistic/fig3A_v0_ideal_realistic_sequential.png", "M2W-FIX", 1536, "SAS-scaled focus crop", "z=60 mm", "lanczos", True, "validated", "sequential_architecture", True),
        _fig("F08", "Corrected sequential single-beam optical system.", "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/00_architecture/fig1_sequential_optical_architecture.png", "M2W-FIX", "diagram", "bench", "", "vector schematic", True, "validated architecture", "sequential_architecture", True),
        _fig("F09", "Native SLM1/SLM2 masks and target vector-field diagnostics.", "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/02_target_masks/fig2_target_and_sequential_masks.png", "M2W-FIX", "1920x1080 masks; 1536 target grid", "native panel + source grid", "pre-axicon", "nearest/lanczos", True, "validated display package", "hardware", True),
        _fig("F10", "Realism degradation: clean, moderate and bad cases.", "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/04_realism/fig3B_clean_moderate_bad_realism.png", "M2W-FIX/M2S", 1024, "10 mm", "z=60 mm", "lanczos", True, "model-supported", "realism", True),
        _fig("F11", "Corrected recovery from a 0.5 mm axicon/mask offset.", "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/05_correction/fig3C_degraded_corrected_recovery.png", "M2W-FIX/M2S/M2V", 1024, "10 mm", "z=60 mm", "lanczos", True, "model-supported", "correction", True),
        _fig("F12", "Transverse propagation with physical context and SAS-scaled focus crops.", "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/06_propagation/fig4A_transverse_evolution.png", "M2W-FIX", 1024, "SAS-scaled focus windows", "0.1-200 mm", "lanczos", True, "validated display provenance", "source_validation", True),
        _fig("F13", "x-z and y-z propagation maps plus z diagnostics.", "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/06_propagation/fig4B_xz_yz_z_diagnostics.png", "M2W-FIX", 1024, "10 mm", "0.1-200 mm", "bilinear", True, "validated", "source_validation", True),
        _fig("F14", "Sequential-route power flow.", "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/07_power/fig4C_sequential_power_flow.png", "M2W-FIX", "ledger", "bench", "", "bar chart", True, "model-supported", "hardware", True),
        _fig("F15", "Tolerance limits table/plot.", "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/08_tolerances/fig5A_tolerance_limits.png", "M2W-FIX/M2S", "mixed", "table", "", "table", True, "model-supported", "realism", False),
        _fig("F16", "Combined realism and correction results.", "outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/08_tolerances/fig5B_combined_correction_results.png", "M2W-FIX/M2S/M2V", "mixed", "table", "", "bar chart", True, "model-supported", "correction", False),
        _fig("F17", "Strict classifier truth table and gate calibration.", "outputs/figures/digital_twin/nathan_mode2u2_fix_strict_hexagon_optimisation/hexagon_classifier_truth_table_highres.png", "M2U2-FIX", "highres", "focus crop", "z=60 mm", "lanczos", True, "validated gate", "provenance", True),
        _fig("S01", "Forbidden old best-compromise audit figure.", "outputs/figures/digital_twin/nathan_mode2u2_fix_strict_hexagon_optimisation/01_old_optima_audit/best_compromise_m2u2_opt_003_c5.75_i0.32_q-0.25_r+0.0_p0.10_audit.png", "M2U2-FIX", "historical", "focus crop", "z=60 mm", "lanczos", True, "forbidden/superseded", "superseded", False),
    ]


def _fig(
    figure_id: str,
    caption: str,
    source_file: str,
    stage: str,
    n: Any,
    physical_window: str,
    z_plane_range: str,
    display_interpolation: str,
    native_metrics: bool,
    validation_status: str,
    evidence_subfolder: str,
    primary_hero: bool,
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "report_caption": caption,
        "source_file": source_file,
        "source_stage": stage,
        "N": n,
        "dx": "see source audit" if figure_id in {"F07", "F12"} else "",
        "physical_window": physical_window,
        "z_plane_or_range": z_plane_range,
        "display_interpolation": display_interpolation,
        "native_metrics": bool(native_metrics),
        "validation_status": validation_status,
        "evidence_subfolder": evidence_subfolder,
        "primary_hero": bool(primary_hero),
    }


def superseded_rows() -> list[dict[str, Any]]:
    return [
        {
            "item_id": "S01",
            "material": "Original M2W split-arm optical diagram",
            "reason_superseded": "It depicted a PBS split/recombine interferometer as canonical; M2W-FIX accepts a sequential single-beam route.",
            "replacement": "F08 sequential optical architecture and docs/83",
            "status": "superseded",
        },
        {
            "item_id": "S02",
            "material": "Split-arm H/V/PBS power ledger",
            "reason_superseded": "Sequential route has no separate H/V spatial arms and no PBS recombination power bookkeeping.",
            "replacement": "F14 and mode2w_fix_sequential_power_ledger.csv",
            "status": "superseded",
        },
        {
            "item_id": "S03",
            "material": OLD_BEST_COMPROMISE_ID,
            "reason_superseded": "High full-field correlation hid fourfold/X-like non-hexagonal structure.",
            "replacement": CANONICAL_OPERATING_POINT_ID,
            "status": "forbidden",
        },
        {
            "item_id": "S04",
            "material": "Low-resolution hero montages",
            "reason_superseded": "High DPI/interpolation cannot replace numerical sampling.",
            "replacement": "N=1536 / SAS-audited M2W-FIX hero figures",
            "status": "superseded",
        },
    ]


def table_sources() -> list[tuple[str, str, str]]:
    return [
        ("outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/01_source_audit/mode2w_fix_numerical_source_audit.csv", "tables", "mode2w_fix_numerical_source_audit.csv"),
        ("outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/03_ideal_vs_realistic/mode2w_fix_ideal_vs_realistic_metrics.csv", "tables", "mode2w_fix_ideal_vs_realistic_metrics.csv"),
        ("outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/07_power/mode2w_fix_sequential_power_ledger.csv", "tables", "mode2w_fix_sequential_power_ledger.csv"),
        ("outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/08_tolerances/mode2w_fix_tolerance_limits.csv", "tables", "mode2w_fix_tolerance_limits.csv"),
        ("outputs/figures/digital_twin/nathan_mode2w_fix_sequential_master/08_tolerances/mode2w_fix_combined_correction_summary.csv", "tables", "mode2w_fix_combined_correction_summary.csv"),
        ("outputs/figures/digital_twin/nathan_mode2s_lab_realism_tolerance/mode2s_single_parameter_tolerances.csv", "realism", "mode2s_single_parameter_tolerances.csv"),
        ("outputs/figures/digital_twin/nathan_mode2s_lab_realism_tolerance/mode2s_combined_cases.csv", "realism", "mode2s_combined_cases.csv"),
        ("outputs/figures/digital_twin/nathan_mode2q_backward_mask_synthesis/mode2q_mask_candidates.csv", "inverse_design", "mode2q_mask_candidates.csv"),
        ("outputs/figures/digital_twin/nathan_mode2u2_fix_strict_hexagon_optimisation/old_optima_strict_audit.csv", "superseded", "old_optima_strict_audit.csv"),
        ("outputs/figures/digital_twin/nathan_mode2u2_fix_strict_hexagon_optimisation/strict_optima_summary.csv", "provenance", "strict_optima_summary.csv"),
    ]


def _copy_evidence(figures: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    copied: list[dict[str, Any]] = []
    for sub in (
        "equations", "figures", "tables", "claims", "provenance", "sequential_architecture",
        "source_validation", "inverse_design", "realism", "correction", "hardware", "superseded",
    ):
        (EVIDENCE_ROOT / sub).mkdir(parents=True, exist_ok=True)
    for row in figures:
        src = Path(str(row["source_file"]))
        if not src.exists():
            copied.append({**dict(row), "copied_to": "", "copy_status": "missing_source"})
            continue
        sub = str(row["evidence_subfolder"])
        dest = EVIDENCE_ROOT / sub / f"{row['figure_id']}_{src.name}"
        shutil.copy2(src, dest)
        copied.append({**dict(row), "copied_to": str(dest), "copy_status": "copied"})
    for src_text, sub, name in table_sources():
        src = Path(src_text)
        if src.exists():
            shutil.copy2(src, EVIDENCE_ROOT / sub / name)
            json_src = src.with_suffix(".json")
            if json_src.exists():
                shutil.copy2(json_src, EVIDENCE_ROOT / sub / json_src.name)
    return copied


def _report_sections() -> list[tuple[str, str]]:
    return [
        ("Introduction", "The project asks whether Nathan's segmented radial/azimuthal source-scale vector Bessel field can be reproduced with a programmable laboratory route. The answer at this stage is yes in simulation for the source-scale branch. The aim is not to claim microfabrication or sample-plane success. It is to establish the mathematical field, the optical transformations, the realistic carrier/4F implementation, and the corrected sequential dual-SLM bench architecture."),
        ("Physical and Mathematical Background", "The source field begins as a Gaussian amplitude A(r)=exp(-r^2/w0^2). The six-sector structure is not a scalar intensity mask. It lives in the local polarization angle alpha(theta)=theta+phi0(theta), where phi0 alternates between radial and azimuthal sectors. Thus the pre-axicon target is E=A[cos alpha, sin alpha]^T, and S0 remains Gaussian before the axicon. Stokes fields reveal the polarization structure; intensity alone does not."),
        ("V0 Source-Model Reproduction", "V0 established the reference observable and grid convention. The Nathan-style axis-sampled grid is required because a zero-straddling grid creates a central artefact at the angular singularity. The literal source port and the project implementation reproduce the same source-scale field. This is the reference used by later route comparisons."),
        ("Vector Axicon Propagation", "After the target vector field reaches the axicon, the local radial and azimuthal components see the source-scale conical phase and p/s response. The propagated observable is total intensity |Ex|^2+|Ey|^2+|Ez|^2. The vector angular-spectrum propagator enforces transversality and advances each spectral component by exp(i kz z). The hexagonal structure is therefore a downstream propagation result, not merely a pre-axicon drawing."),
        ("Ideal Jones-Chain Synthesis", "M2P showed two analytic routes. A patterned HWP with beta=alpha/2 maps horizontal input to the target Jones vector. The dual-SLM/QWP route uses phase masks phi_H=+alpha and phi_V=-alpha+pi/2, followed by a QWP in the selected project convention. The derivation is power preserving; relative piston rotates the local polarization state and remains visible to polarimetry even when total intensity is insensitive."),
        ("Forward Source-Scale Bench Replication", "M2N propagated the ideal patterned-HWP, ideal dual-SLM/QWP, and realistic dual-SLM + carrier + 4F + QWP routes through the same source axicon. The ideal routes reproduce V0 at z=60 mm. The realistic carrier/4F route keeps the strict visual hexagon with z=60 correlation around 0.9936 and first-order efficiency around 0.949."),
        ("Backward / Adjoint Mask Synthesis", "M2Q starts from the full complex V0 target at z=60 mm, backpropagates it, applies an inverse axicon operation, and removes the QWP. The recovered H/V phase solution independently returns phi_H=+alpha and phi_V=-alpha+pi/2. This is important because it confirms the forward analytic mask convention from the inverse problem."),
        ("Realistic 4F Filtering", "A displayed phase mask includes the desired phase plus a carrier 2*pi*nu*x. In the Fourier plane the +1 order is displaced by x=lambda f nu. With lambda=1029 nm, f=300 mm and nu=6.25 lp/mm, the displacement is about 1.929 mm and the recommended iris diameter is about 1.54 mm. The common 4F route filters both polarization channels together."),
        ("Lab Realism and Tolerance", "M2S introduced quantisation, fill factor, H/V imbalance, piston, QWP errors, iris errors, registration, axicon decentre and z offsets. Within tested ranges, most representative imperfections were relatively forgiving. The dominant practical sensitivity is hologram/mask centre relative to the axicon axis. A 0.5 mm axicon/mask offset is not acceptable blindly but can be substantially recovered by digital recentring."),
        ("High-Resolution and Sampling Audit", "M2U separated numerical resolution from display resolution. N=512 is Nyquist-safe but not publication-preferred. N=1024 is useful confirmation. N=1536 is publication-selected for the hero comparison with about 9.89 native samples per radial fringe. M2W-FIX adds SAS-scaled focus crops so close-up panels are rendered on finer output grids while classifier metrics remain tied to native validated arrays."),
        ("Optimisation Integrity and Strict Hexagon Gate", "M2U2-FIX repaired an important failure: high full-field correlation can be dominated by dark background and can hide fourfold or triangular structures. The old best compromise is permanently forbidden. The strict gate uses class, focus support, angular correlation, c60/c120 behaviour, dark-core limits and fourfold vetoes. Its 0.997 reference-correlation floor is a project-calibrated eligibility threshold, not a universal definition of a hexagon."),
        ("Native-Panel and Hardware Provenance", "The panel raster is 1920 x 1080 at 8 um pitch, giving a 15.36 x 8.64 mm active area. The exact PLUTO-2.1 NIR-149 identity is externally supplied until physical labels/manuals are read. Phase stroke and LUT are calibration-required. Camera scale, z-stage and QWP mount sign also remain bench calibrations."),
        ("Sequential Single-Beam Dual-SLM Architecture", "M2W-FIX supersedes the original split-arm diagram. The accepted implementation is collinear and sequential: equal H/V preparation, SLM1, optional swap HWP, SLM2, optional swap-back, common 4F, QWP, axicon and camera. The sequential Jones derivation gives E4=A/sqrt(2)[exp(i phi_H), exp(i phi_V)]^T. The pre-QWP and post-QWP overlaps are 1.0, the ideal z=60 correlation is 1.0, and the realistic sequential route remains strict-eligible."),
        ("Sequential Power / Energy Ledger", "The corrected power model has no split-arm H/V or PBS recombination rows. It tracks one beam through preparation, SLM1, optional swap, SLM2, common 4F selection, QWP, axicon, z=60 integration and useful-region power. The 1 W and 10 W examples are linear model examples only and are not damage-threshold claims."),
        ("Closed-Loop Correction", "The camera observes final intensity, centre, symmetry, dark core, lobe balance and z structure. Shack-Hartmann measurements address low-order common wavefront errors. Polarimetry checks pre-axicon vector field, H/V balance, relative phase and QWP sign. The bounded correction loop improves several degraded cases, but not every failure is fully recovered to strict eligibility."),
        ("Final Lab Build", "The recommended bench is a 1029 nm Gaussian source, POL/HWP preparation, sequential SLM1/SLM2 masks, common 300 mm 4F with 6.25 lp/mm carrier and 1.54 mm iris, QWP at nominal code -45 deg, 2 deg n=1.458 axicon, and a camera z scan around z=60 mm. The six-piece segmented optic is not required for this programmable source-scale trial."),
        ("Conclusions", "Nathan's source-scale field has been reproduced in simulation; the propagated result is a genuine strict-gate hexagonal field; the vector field can be synthesised analytically; inverse propagation recovers the same masks; realistic common-4F filtering preserves the beam; the main practical sensitivity is mask/axicon centring; the sequential single-beam dual-SLM architecture is valid in simulation. Source-scale lab trial is justified. Microfabrication/sample-plane success is not established."),
        ("Limitations", "The exact SLM phase stroke and per-panel LUT are not measured here. Camera scale and z-stage calibration remain unresolved. Exact LC-director orientation is unverified. The 4F geometry is nominal until bench measurement. Simulations do not prove experimental success. Strict thresholds are project-calibrated. No microfabrication/sample-plane success claim is made."),
        ("Next Steps", "Read physical SLM labels and determine LC directors; measure phase LUTs; deploy native masks; calibrate 4F displacement and iris; validate pre-axicon Stokes fields; insert the axicon; perform camera z scans; digitally recentre; add Shack-Hartmann correction; then use a bounded camera-driven correction loop before revisiting microfabrication adaptation."),
    ]


def _latex_table_from_rows(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], caption: str, label: str) -> str:
    col_spec = "p{0.18\\textwidth}" * len(fields)
    out = [r"\begin{longtable}{" + col_spec + "}", rf"\caption{{{_latex_escape(caption)}}}\label{{{label}}}\\", r"\toprule"]
    out.append(" & ".join(_latex_escape(f) for f in fields) + r"\\")
    out.append(r"\midrule")
    out.append(r"\endfirsthead")
    out.append(r"\toprule")
    out.append(" & ".join(_latex_escape(f) for f in fields) + r"\\")
    out.append(r"\midrule")
    out.append(r"\endhead")
    for row in rows:
        out.append(" & ".join(_latex_escape(row.get(f, "")) for f in fields) + r"\\")
    out.append(r"\bottomrule")
    out.append(r"\end{longtable}")
    return "\n".join(out)


def _build_latex(figures: Sequence[Mapping[str, Any]], copied: Sequence[Mapping[str, Any]]) -> str:
    today = date.today().isoformat()
    figure_by_id = {row["figure_id"]: row for row in copied if row.get("copy_status") == "copied"}
    lines: list[str] = [
        r"\documentclass[11pt,a4paper]{report}",
        r"\usepackage[margin=22mm]{geometry}",
        r"\usepackage{amsmath,amssymb,bm}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs,longtable,array}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{float}",
        r"\usepackage{caption}",
        r"\setlength{\parskip}{0.5em}",
        r"\setlength{\parindent}{0pt}",
        r"\title{Inverse Design and Sequential Dual-SLM Realisation of a Source-Scale Hexagonal Vector Bessel Beam}",
        rf"\author{{Nathan source-scale branch technical evidence draft\\Generated by Codex on {today}}}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{center}\textbf{Scope statement:} This report covers the source-scale branch. It does not claim experimental replication or microfabrication/sample-plane success.\end{center}",
        r"\tableofcontents",
        r"\listoffigures",
        r"\chapter*{Abstract}",
        r"\addcontentsline{toc}{chapter}{Abstract}",
        "This technical evidence draft derives the Nathan source-scale hexagonal vector Bessel beam model, traces the ideal and realistic synthesis routes, documents inverse reconstruction, audits sampling and strict-hexagon classification, and specifies the corrected sequential single-beam dual-SLM laboratory architecture. The current conclusion is that the source-scale sequential architecture is validated in simulation and sufficiently specified to justify a source-scale laboratory trial. Experimental replication and microfabrication/sample-plane success remain separate unresolved stages.",
        r"\chapter*{Notation and Abbreviations}",
        r"\addcontentsline{toc}{chapter}{Notation and Abbreviations}",
        r"\begin{longtable}{p{0.18\textwidth}p{0.72\textwidth}}",
        r"\toprule",
        r"Symbol or term & Meaning\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Symbol or term & Meaning\\",
        r"\midrule",
        r"\endhead",
        r"\(A(r)\) & Gaussian source amplitude envelope.\\",
        r"\(w_0\) & Source-scale beam radius used by the Nathan parity branch.\\",
        r"\(\theta\) & Transverse-plane angular coordinate, \(\operatorname{atan2}(y,x)\).\\",
        r"\(\alpha(\theta)\) & Local linear-polarisation angle after the radial/azimuthal sector offset.\\",
        r"SLM & Spatial light modulator; here a phase-only LCOS panel.\\",
        r"HWP & Half-wave plate; a retarder used for polarisation rotation or channel swapping.\\",
        r"QWP & Quarter-wave plate; the final retarder that maps the dual phase channels into the target linear vector field.\\",
        r"PBS & Polarising beamsplitter. It appears only as superseded split-arm context here, not in the accepted canonical architecture.\\",
        r"ASM & Angular-spectrum method for free-space propagation.\\",
        r"4F & Fourier filtering relay with two lenses separated by focal lengths, used to select the carrier-shifted +1 diffraction order.\\",
        r"\(S_0,S_1,S_2,S_3\) & Stokes observables used to inspect intensity and polarisation state.\\",
        r"\bottomrule",
        r"\end{longtable}",
    ]
    for idx, (title, body) in enumerate(_report_sections(), start=1):
        lines.append(rf"\chapter{{{_latex_escape(title)}}}")
        lines.append(_latex_escape(body))
        if idx == 1:
            lines.append(r"\textbf{Key rule:} the final architecture is sequential and collinear. No canonical PBS split/recombine interferometer arms are used.")
        if idx == 2:
            lines.append(r"\begin{align}A(r)&=\exp(-r^2/w_0^2)\\ \alpha(\theta)&=\theta+\phi_0(\theta)\\ \mathbf E_\mathrm{target}&=A(r)[\cos\alpha,\sin\alpha]^T\end{align}")
        if title == "Sequential Single-Beam Dual-SLM Architecture":
            lines.append(r"\begin{align}E_0&=\frac{A}{\sqrt2}[1,1]^T\\E_1&=\frac{A}{\sqrt2}[e^{i\phi_H},1]^T\\E_2&=\frac{A}{\sqrt2}[1,e^{i\phi_H}]^T\\E_3&=\frac{A}{\sqrt2}[e^{i\phi_V},e^{i\phi_H}]^T\\E_4&=\frac{A}{\sqrt2}[e^{i\phi_H},e^{i\phi_V}]^T.\end{align}")
            lines.append(r"The masks are \(\phi_H=+\alpha+\) carrier and \(\phi_V=-\alpha+\pi/2+\) carrier.")
        for fid in _figures_for_section(title):
            row = figure_by_id.get(fid)
            if row:
                rel = Path(str(row["copied_to"])).relative_to(REPORT_ROOT)
                lines.extend([
                    r"\begin{figure}[H]",
                    r"\centering",
                    rf"\includegraphics[width=0.92\textwidth]{{\detokenize{{{_tex_path(rel)}}}}}",
                    rf"\caption{{{_latex_escape(row['report_caption'])}}}",
                    rf"\label{{fig:{_latex_escape(fid)}}}",
                    r"\end{figure}",
                ])
    lines.extend([
        r"\appendix",
        r"\chapter{Equation Registry}",
        _latex_table_from_rows(equation_rows(), ["equation_id", "name", "latex", "code", "validation_status"], "Machine-readable equation registry excerpt.", "tab:eq-registry"),
        r"\chapter{Claim Registry}",
        _latex_table_from_rows(claim_rows(), ["claim_id", "claim", "numerical_result", "status", "limitations"], "Claim and evidence registry excerpt.", "tab:claim-registry"),
        r"\chapter{Number Registry}",
        _latex_table_from_rows(number_rows(), ["number_id", "quantity", "value", "units", "status"], "Master numerical registry excerpt.", "tab:number-registry"),
        r"\chapter{Superseded and Forbidden Material}",
        _latex_table_from_rows(superseded_rows(), ["item_id", "material", "reason_superseded", "replacement", "status"], "Superseded and invalidated material.", "tab:superseded"),
        r"\chapter{Figure Manifest}",
        _latex_table_from_rows(figures, ["figure_id", "report_caption", "source_stage", "N", "validation_status"], "Figure provenance manifest excerpt.", "tab:fig-manifest"),
        r"\end{document}",
    ])
    return "\n\n".join(lines)


def _figures_for_section(title: str) -> list[str]:
    return {
        "V0 Source-Model Reproduction": ["F01", "F02", "F03"],
        "Physical and Mathematical Background": ["F04", "F05"],
        "Ideal Jones-Chain Synthesis": ["F04", "F05"],
        "Backward / Adjoint Mask Synthesis": ["F06"],
        "Forward Source-Scale Bench Replication": ["F07"],
        "Realistic 4F Filtering": ["F07", "F09"],
        "Lab Realism and Tolerance": ["F10", "F11"],
        "High-Resolution and Sampling Audit": ["F03", "F07", "F12"],
        "Optimisation Integrity and Strict Hexagon Gate": ["F17", "S01"],
        "Native-Panel and Hardware Provenance": ["F09"],
        "Sequential Single-Beam Dual-SLM Architecture": ["F08", "F07"],
        "Sequential Power / Energy Ledger": ["F14"],
        "Closed-Loop Correction": ["F11", "F16"],
        "Final Lab Build": ["F08", "F13"],
    }.get(title, [])


def _write_references() -> Path:
    REFERENCES_BIB.write_text(
        """@misc{project-digital-twin,
  title = {Nathan hexagonal Bessel source-scale digital twin evidence pack},
  note = {Internal project repository outputs and generated registries}
}
""",
        encoding="utf-8",
    )
    return REFERENCES_BIB


def _write_readme(summary: Mapping[str, Any]) -> Path:
    text = f"""# Nathan Hexagonal Bessel Full Report Pack

Generated by `vbb_study.digital_twin.nathan_full_report_pack.build_nathan_full_report_pack`.

## Outputs

- LaTeX source: `{REPORT_TEX}`
- PDF: `{REPORT_PDF}`
- Equation registry: `{EQUATION_REGISTRY}`
- Claim registry: `{CLAIM_REGISTRY}`
- Number registry: `{NUMBER_REGISTRY}`
- Figure manifest: `{FIGURE_MANIFEST}`
- Evidence pack: `{EVIDENCE_ROOT}`

## Build

Preferred LaTeX command:

```powershell
pdflatex nathan_hexagonal_bessel_full_report.tex
pdflatex nathan_hexagonal_bessel_full_report.tex
```

Detected PDF build method in this environment: `{summary['pdf_build_method']}`.

Recorded build command/status: `{summary['compile_command']}`.

If `pdflatex`, `latexmk`, `xelatex`, `lualatex` or `tectonic` is unavailable, the generator creates a fallback PDF with Matplotlib so the evidence pack still contains a viewable report. The LaTeX source remains the authoritative editable source.

Required LaTeX packages:

- `geometry`
- `amsmath`
- `amssymb`
- `bm`
- `graphicx`
- `booktabs`
- `longtable`
- `array`
- `hyperref`
- `float`
- `caption`

## Figure Source Mapping

The authoritative machine-readable mapping is in:

- `{FIGURE_MANIFEST}`
- `{FIGURE_MANIFEST.with_suffix('.json')}`

Each row records the figure ID, caption, source file, source stage, numerical grid `N`, physical window, interpolation status, validation status and copied evidence path.

Primary mapping excerpt:

"""
    for row in figure_rows():
        if row.get("primary_hero"):
            text += f"- `{row['figure_id']}`: `{row['source_file']}` -> stage `{row['source_stage']}`, N `{row['N']}`\n"
    text += f"""
## Notes

- The canonical architecture is the corrected sequential single-beam dual-SLM route.
- The old split-arm M2W architecture is superseded and appears only in the superseded-material registry.
- No microfabrication/sample-plane success claim is made.
"""
    REPORT_README.write_text(text, encoding="utf-8")
    return REPORT_README


def _compile_or_fallback_pdf(copied_figures: Sequence[Mapping[str, Any]], sections: Sequence[tuple[str, str]]) -> dict[str, Any]:
    engine = _engine_available()
    if engine:
        if engine == "latexmk":
            cmd = [engine, "-pdf", "-interaction=nonstopmode", REPORT_TEX.name]
        elif engine == "tectonic":
            cmd = [engine, REPORT_TEX.name]
        else:
            cmd = [engine, "-interaction=nonstopmode", REPORT_TEX.name]
        try:
            subprocess.run(cmd, cwd=REPORT_ROOT, check=True, capture_output=True, text=True, timeout=180)
            if engine not in {"latexmk", "tectonic"}:
                subprocess.run(cmd, cwd=REPORT_ROOT, check=True, capture_output=True, text=True, timeout=180)
            return {
                "pdf_build_method": engine,
                "pdf_build_status": "compiled_with_latex_engine",
                "compile_command": " ".join(cmd),
                "pdf_path": str(REPORT_PDF),
            }
        except Exception as exc:  # pragma: no cover - engine unavailable in current CI.
            fallback = _write_pdf_fallback(copied_figures, sections, reason=f"{engine} failed: {exc}")
            return fallback
    return _write_pdf_fallback(copied_figures, sections, reason="No LaTeX engine found on PATH.")


def _write_pdf_fallback(copied_figures: Sequence[Mapping[str, Any]], sections: Sequence[tuple[str, str]], *, reason: str) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    REPORT_PDF.parent.mkdir(parents=True, exist_ok=True)
    page_count = 0
    with PdfPages(REPORT_PDF) as pdf:
        page_count += _pdf_text_page(
            pdf,
            "Inverse Design and Sequential Dual-SLM Realisation of a Source-Scale Hexagonal Vector Bessel Beam",
            "Source-scale technical evidence draft. Microfabrication/sample-plane success is not claimed.\n\n"
            "This PDF is a fallback rendering because the local environment lacks a LaTeX engine. "
            "The editable LaTeX source is still written to report/nathan_hexagonal_bessel_full_report.tex.",
            plt,
        )
        page_count += _pdf_text_page(pdf, "Abstract", "This report derives the source-scale branch from the Gaussian vector target through ideal Jones synthesis, inverse reconstruction, realistic carrier/4F filtering, tolerance audit, strict gate repair and the corrected sequential single-beam dual-SLM architecture.", plt)
        for title, body in sections:
            page_count += _pdf_text_page(pdf, title, body, plt)
        for row in copied_figures:
            if row.get("copy_status") != "copied":
                continue
            path = Path(str(row["copied_to"]))
            if path.suffix.lower() != ".png":
                continue
            try:
                img = mpimg.imread(path)
            except Exception:
                continue
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle(str(row["figure_id"]) + ": " + str(row["report_caption"]), fontsize=12, fontweight="bold", y=0.97)
            ax = fig.add_axes([0.08, 0.18, 0.84, 0.72])
            ax.imshow(img)
            ax.axis("off")
            fig.text(0.08, 0.08, "Source: " + str(row["source_file"]), fontsize=8, wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
            page_count += 1
    return {
        "pdf_build_method": "matplotlib_fallback",
        "pdf_build_status": "fallback_pdf_created",
        "compile_command": "pdflatex nathan_hexagonal_bessel_full_report.tex (not run: " + reason + ")",
        "pdf_path": str(REPORT_PDF),
        "page_count": page_count,
        "fallback_reason": reason,
    }


def _pdf_text_page(pdf: Any, title: str, body: str, plt: Any) -> int:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.94, title, fontsize=15, fontweight="bold", va="top")
    wrapped = "\n\n".join(textwrap.fill(p, width=88) for p in str(body).split("\n\n"))
    fig.text(0.07, 0.88, wrapped, fontsize=10.5, va="top", family="serif")
    fig.text(0.07, 0.04, "Nathan source-scale branch technical evidence draft", fontsize=8, color="0.35")
    pdf.savefig(fig)
    plt.close(fig)
    return 1


def build_nathan_full_report_pack(*, report_root: str | Path = REPORT_ROOT) -> dict[str, Any]:
    """Build the full report/evidence pack."""

    global REPORT_ROOT, REPORT_TEX, REPORT_PDF, REPORT_README, REFERENCES_BIB
    global EVIDENCE_ROOT, EQUATION_REGISTRY, CLAIM_REGISTRY, NUMBER_REGISTRY
    global FIGURE_MANIFEST, SUPERSEDED_TABLE, BUILD_SUMMARY

    REPORT_ROOT = Path(report_root)
    REPORT_TEX = REPORT_ROOT / "nathan_hexagonal_bessel_full_report.tex"
    REPORT_PDF = REPORT_ROOT / "nathan_hexagonal_bessel_full_report.pdf"
    REPORT_README = REPORT_ROOT / "README.md"
    REFERENCES_BIB = REPORT_ROOT / "references.bib"
    EVIDENCE_ROOT = REPORT_ROOT / "evidence_pack"
    EQUATION_REGISTRY = REPORT_ROOT / "equation_registry.csv"
    CLAIM_REGISTRY = REPORT_ROOT / "claim_registry.csv"
    NUMBER_REGISTRY = REPORT_ROOT / "number_registry.csv"
    FIGURE_MANIFEST = REPORT_ROOT / "figure_manifest.csv"
    SUPERSEDED_TABLE = REPORT_ROOT / "superseded_material.csv"
    BUILD_SUMMARY = REPORT_ROOT / "build_report_summary.json"

    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    figures = figure_rows()
    copied = _copy_evidence(figures)
    _write_csv(EQUATION_REGISTRY, equation_rows())
    _write_json(EQUATION_REGISTRY.with_suffix(".json"), equation_rows())
    _write_csv(CLAIM_REGISTRY, claim_rows())
    _write_json(CLAIM_REGISTRY.with_suffix(".json"), claim_rows())
    _write_csv(NUMBER_REGISTRY, number_rows())
    _write_json(NUMBER_REGISTRY.with_suffix(".json"), number_rows())
    _write_csv(FIGURE_MANIFEST, copied)
    _write_json(FIGURE_MANIFEST.with_suffix(".json"), copied)
    _write_csv(SUPERSEDED_TABLE, superseded_rows())
    _write_json(SUPERSEDED_TABLE.with_suffix(".json"), superseded_rows())
    shutil.copy2(EQUATION_REGISTRY, EVIDENCE_ROOT / "equations" / EQUATION_REGISTRY.name)
    shutil.copy2(CLAIM_REGISTRY, EVIDENCE_ROOT / "claims" / CLAIM_REGISTRY.name)
    shutil.copy2(NUMBER_REGISTRY, EVIDENCE_ROOT / "provenance" / NUMBER_REGISTRY.name)
    shutil.copy2(FIGURE_MANIFEST, EVIDENCE_ROOT / "provenance" / FIGURE_MANIFEST.name)
    shutil.copy2(SUPERSEDED_TABLE, EVIDENCE_ROOT / "superseded" / SUPERSEDED_TABLE.name)
    tex = _build_latex(figures, copied)
    REPORT_TEX.write_text(tex, encoding="utf-8")
    _write_references()
    pdf_summary = _compile_or_fallback_pdf(copied, _report_sections())
    summary = {
        "title": "Inverse Design and Sequential Dual-SLM Realisation of a Source-Scale Hexagonal Vector Bessel Beam",
        "report_root": str(REPORT_ROOT),
        "tex_path": str(REPORT_TEX),
        "pdf_path": str(REPORT_PDF),
        "equation_registry": str(EQUATION_REGISTRY),
        "claim_registry": str(CLAIM_REGISTRY),
        "number_registry": str(NUMBER_REGISTRY),
        "figure_manifest": str(FIGURE_MANIFEST),
        "superseded_table": str(SUPERSEDED_TABLE),
        "evidence_pack": str(EVIDENCE_ROOT),
        "section_count": len(_report_sections()),
        "figure_count": len([row for row in copied if row.get("copy_status") == "copied"]),
        "table_count": 5,
        "equation_count": len(equation_rows()),
        "claim_count": len(claim_rows()),
        "number_count": len(number_rows()),
        "microfabrication_sample_plane_claim": False,
        **pdf_summary,
    }
    _write_readme(summary)
    _write_json(BUILD_SUMMARY, summary)
    return summary


__all__ = [
    "REPORT_ROOT",
    "REPORT_TEX",
    "REPORT_PDF",
    "EVIDENCE_ROOT",
    "equation_rows",
    "claim_rows",
    "number_rows",
    "figure_rows",
    "superseded_rows",
    "build_nathan_full_report_pack",
]
