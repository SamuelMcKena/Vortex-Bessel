"""Vector and analyzer helpers for the vortex-Bessel-beam study.

The actual lab path is now Case 1: two phase-only SLMs share one director axis,
all folds are in-plane, and there are no waveplates. I therefore model only one
shaped H component plus an unshaped V reference as lab-realizable. The old
Baliyan-Nishchal waveplate chain remains available as a labelled paper-replica
simulation cross-check, not as the current bench model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy import special as sp

from . import vbb_style
from .equations import holography as holography_eq
from .equations import scalar_bessel as scalar_eq
from .equations import vector_jones as jones_eq
from vbb_study.config import EPS as BT_EPS, nm as BT_NM, um as BT_UM
from vbb_study.equations.propagation import make_bl_asm_propagator

EPS = 1.0e-15


@dataclass(frozen=True)
class VectorField:
	"""Two-component complex field in a fixed Jones basis."""

	Ex: np.ndarray
	Ey: np.ndarray


@dataclass(frozen=True)
class SLMVectorProgram:
	"""Explicit per-pass SLM phase program plus any fixed downstream optics."""

	hardware: str
	plus_phase_offset_rad: float
	minus_phase_offset_rad: float
	post_optics: str | Sequence[Any] | None
	post_optics_label: str
	notes: str = ""


@dataclass(frozen=True)
class LabVectorHardware:
	"""Actual Case-1 hardware flags for the current bench.

	I keep this separate from the paper Jones chain because the real train has
	two phase-only SLMs with the same director axis, in-plane folds, and no
	waveplates. That means only H is shaped and V is a fixed reference.
	"""

	future_element: str = "none"
	same_director_axis: bool = True
	fold_mirrors_in_plane: bool = True
	waveplates_present: bool = False
	shaped_axis: str = "H"
	reference_axis: str = "V"


def _complex_array(values: Any) -> np.ndarray:
	arr = np.asarray(values, dtype=complex)
	if arr.ndim != 2:
		raise ValueError("Expected one 2D complex field.")
	return arr


def _grid_xy(grid: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
	if "X" in grid and "Y" in grid:
		X = np.asarray(grid["X"], dtype=float)
		Y = np.asarray(grid["Y"], dtype=float)
	else:
		x = np.asarray(grid["x"], dtype=float)
		y = np.asarray(grid.get("y", x), dtype=float)
		X, Y = np.meshgrid(x, y, indexing="xy")
	if "x" in grid:
		x = np.asarray(grid["x"], dtype=float)
	else:
		x = np.asarray(X[0, :], dtype=float)
	if "y" in grid:
		y = np.asarray(grid["y"], dtype=float)
	else:
		y = np.asarray(Y[:, 0], dtype=float)
	dx = float(grid.get("dx", np.median(np.diff(x))))
	return X, Y, x, y, dx


def _extent_um(grid: Mapping[str, Any]) -> list[float]:
	_, _, x, y, _ = _grid_xy(grid)
	return [float(x[0] / BT_UM), float(x[-1] / BT_UM), float(y[0] / BT_UM), float(y[-1] / BT_UM)]


def _phase_angle(grid: Mapping[str, Any]) -> np.ndarray:
	X, Y, _, _, _ = _grid_xy(grid)
	return np.arctan2(Y, X)


def _wrap_headless(delta: np.ndarray) -> np.ndarray:
	return 0.5 * np.angle(np.exp(2j * np.asarray(delta, dtype=float)))


def _ensure_matrix(matrix: Any) -> np.ndarray:
	arr = np.asarray(matrix, dtype=complex)
	if arr.ndim < 2 or arr.shape[0] != 2 or arr.shape[1] != 2:
		raise ValueError("Jones matrices must have leading shape (2, 2).")
	return arr


def _post_optics_label(post_optics: str | Sequence[Any] | None) -> str:
	if post_optics is None:
		return "none"
	if isinstance(post_optics, str):
		label = post_optics.lower().strip()
		return label or "none"
	return "custom"


def compose_jones_matrices(*matrices: Any) -> np.ndarray:
	"""Compose Jones matrices in the same order as the physical optics chain."""

	return jones_eq.compose_jones_matrices(*matrices)


def apply_jones_matrix(vf: VectorField, matrix: Any) -> VectorField:
	"""Apply a Jones matrix, including spatially varying 2x2 matrices."""

	Ex, Ey = jones_eq.apply_jones_matrix(vf.Ex, vf.Ey, matrix)
	return VectorField(Ex=Ex, Ey=Ey)


def apply_hardware_chain(vf: VectorField, matrices: Sequence[Any] | None = None) -> VectorField:
	"""Apply an optional sequence of Jones matrices to one vector field."""

	out = vf
	for matrix in matrices or ():
		out = apply_jones_matrix(out, matrix)
	return out


def linear_polarizer_45_matrix() -> np.ndarray:
	"""Return the 45 degree linear-polarizer Jones matrix from paper eq. 10."""

	return jones_eq.linear_polarizer_45_matrix()


def first_slm_half_matrix(beta: np.ndarray) -> np.ndarray:
	"""Return the first SLM-half matrix from paper eq. 11."""

	return jones_eq.first_slm_half_matrix(beta)


def reflection_helicity_flip_matrix() -> np.ndarray:
	"""Return the reflection helicity-flip matrix from paper eq. 11.1."""

	return jones_eq.reflection_helicity_flip_matrix()


def hwp_45_swap_matrix() -> np.ndarray:
	"""Return the HWP-at-45-degree component-swap matrix from paper eq. 12."""

	return jones_eq.hwp_45_swap_matrix()


def second_slm_half_matrix(beta: np.ndarray) -> np.ndarray:
	"""Return the second SLM-half matrix from paper eq. 13."""

	return jones_eq.second_slm_half_matrix(beta)


def qwp1_matrix(mode: str) -> np.ndarray:
	"""Return QWP1 for radial or azimuthal generation from paper eqs. 14-15."""

	return jones_eq.qwp1_matrix(mode)


def qwp2_matrix() -> np.ndarray:
	"""Return QWP2 at 45 degrees to the circular basis from paper eq. 16."""

	return jones_eq.qwp2_matrix()


def resolve_post_optics(mode: str, post_optics: str | Sequence[Any] | None) -> tuple[Any, ...]:
	"""Resolve one named post-SLM optics option to an explicit Jones chain."""

	if post_optics is None:
		return ()
	if isinstance(post_optics, str):
		key = post_optics.lower().strip()
		if key in {"", "none", "identity"}:
			return ()
		if key in {"paper_qwp", "paper", "qwp"}:
			return (qwp1_matrix(mode), qwp2_matrix())
		raise ValueError("post_optics must be 'none', 'identity', or 'paper_qwp'.")
	return tuple(post_optics)


def circular_components_to_linear(E_sigma_plus: np.ndarray, E_sigma_minus: np.ndarray) -> VectorField:
	"""Convert circular-basis components into the linear x/y Jones basis."""

	plus = _complex_array(E_sigma_plus)
	minus = _complex_array(E_sigma_minus)
	if plus.shape != minus.shape:
		raise ValueError("Circular-basis component fields must share the same shape.")
	Ex, Ey = jones_eq.circular_to_linear(plus, minus)
	return VectorField(Ex=Ex, Ey=Ey)


def _mode_relative_phase(mode: str, relative_phase_rad: float | None = None) -> float:
	if relative_phase_rad is not None:
		return float(relative_phase_rad)
	mode_l = str(mode).lower().strip()
	if mode_l == "radial":
		return 0.0
	if mode_l == "azimuthal":
		return 0.5 * np.pi
	raise ValueError("mode must be 'radial' or 'azimuthal' when relative_phase_rad is omitted.")


def resolve_slm_encoder_program(
	mode: str,
	*,
	relative_phase_rad: float | None = None,
	hardware: str | Mapping[str, Any] | None = None,
	post_optics: str | Sequence[Any] | None = None,
) -> SLMVectorProgram:
	"""Resolve one explicit per-pass SLM vector program under a hardware constraint."""

	phase_shift = _mode_relative_phase(mode, relative_phase_rad)
	if hardware is None:
		hardware = "slm_only_symmetric"

	if isinstance(hardware, Mapping):
		resolved_post_optics = hardware.get("post_optics", post_optics)
		label = str(hardware.get("hardware", hardware.get("label", "custom"))).strip() or "custom"
		return SLMVectorProgram(
			hardware=label,
			plus_phase_offset_rad=float(hardware["plus_phase_offset_rad"]),
			minus_phase_offset_rad=float(hardware["minus_phase_offset_rad"]),
			post_optics=resolved_post_optics,
			post_optics_label=_post_optics_label(resolved_post_optics),
			notes=str(hardware.get("notes", "Custom SLM encoder program.")),
		)

	key = str(hardware).lower().strip()
	if key in {"slm_only", "slm_only_symmetric", "symmetric"}:
		resolved_post_optics = "none" if post_optics is None else post_optics
		return SLMVectorProgram(
			hardware="slm_only_symmetric",
			plus_phase_offset_rad=-phase_shift,
			minus_phase_offset_rad=phase_shift,
			post_optics=resolved_post_optics,
			post_optics_label=_post_optics_label(resolved_post_optics),
			notes="Two realistic SLM passes carry opposite uniform phase offsets; no fixed downstream optics are assumed unless explicitly requested.",
		)
	if key in {"slm_symmetric_fixed_qwp", "fixed_qwp", "paper_qwp_fixed"}:
		resolved_post_optics = "paper_qwp" if post_optics is None else post_optics
		return SLMVectorProgram(
			hardware="slm_symmetric_fixed_qwp",
			plus_phase_offset_rad=-phase_shift,
			minus_phase_offset_rad=phase_shift,
			post_optics=resolved_post_optics,
			post_optics_label=_post_optics_label(resolved_post_optics),
			notes="Two realistic SLM passes carry opposite uniform phase offsets and the fixed downstream optics are the paper QWP chain.",
		)
	raise ValueError(
		"hardware must be 'slm_only_symmetric', 'slm_symmetric_fixed_qwp', or a mapping with explicit phase offsets."
	)


def apply_slm_phase_program(
	component_plus_charge: np.ndarray,
	component_minus_charge: np.ndarray,
	program: SLMVectorProgram,
) -> tuple[np.ndarray, np.ndarray]:
	"""Apply one explicit per-pass SLM phase program to the ±ell component fields."""

	plus_charge = _complex_array(component_plus_charge)
	minus_charge = _complex_array(component_minus_charge)
	if plus_charge.shape != minus_charge.shape:
		raise ValueError("SLM-programmed component fields must share the same shape.")
	programmed_plus = plus_charge * np.exp(1j * float(program.plus_phase_offset_rad))
	programmed_minus = minus_charge * np.exp(1j * float(program.minus_phase_offset_rad))
	return programmed_plus, programmed_minus


def _slm_encoded_field_and_program(
	component_plus_charge: np.ndarray,
	component_minus_charge: np.ndarray,
	mode: str,
	*,
	relative_phase_rad: float | None = None,
	hardware: str | Mapping[str, Any] | None = None,
	post_optics: str | Sequence[Any] | None = None,
) -> tuple[VectorField, SLMVectorProgram]:
	program = resolve_slm_encoder_program(
		mode,
		relative_phase_rad=relative_phase_rad,
		hardware=hardware,
		post_optics=post_optics,
	)
	programmed_plus, programmed_minus = apply_slm_phase_program(
		component_plus_charge,
		component_minus_charge,
		program,
	)
	sigma_plus = programmed_minus
	sigma_minus = programmed_plus
	vf = circular_components_to_linear(sigma_plus, sigma_minus)
	return apply_hardware_chain(vf, resolve_post_optics(mode, program.post_optics)), program


def build_slm_encoded_vector_field(
	component_plus_charge: np.ndarray,
	component_minus_charge: np.ndarray,
	mode: str,
	*,
	relative_phase_rad: float | None = None,
	hardware: str | Mapping[str, Any] | None = None,
	post_optics: str | Sequence[Any] | None = None,
) -> VectorField:
	"""Build a vector field from two SLM-programmed ±ell component channels.

	The two scalar channels are treated as opposite circular-polarization basis
	states and then converted into the linear x/y Jones basis used by the Stokes
	and analyzer utilities. Any downstream optics are explicit and optional.
	"""

	vf, _ = _slm_encoded_field_and_program(
		component_plus_charge,
		component_minus_charge,
		mode,
		relative_phase_rad=relative_phase_rad,
		hardware=hardware,
		post_optics=post_optics,
	)
	return vf


def paper_dualpass_chain(beta: np.ndarray) -> tuple[np.ndarray, ...]:
	"""Return the explicit pre-QWP dual-pass chain from paper eqs. 10-13."""

	return (
		linear_polarizer_45_matrix(),
		first_slm_half_matrix(beta),
		reflection_helicity_flip_matrix(),
		hwp_45_swap_matrix(),
		second_slm_half_matrix(beta),
	)


def paper_full_chain(beta: np.ndarray, mode: str) -> tuple[np.ndarray, ...]:
	"""Return the full paper chain from eqs. 10-16."""

	return (*paper_dualpass_chain(beta), qwp1_matrix(mode), qwp2_matrix())


def scalar_bg_envelope(
	grid: Mapping[str, Any],
	ell: int,
	kr_m_inv: float,
	waist_m: float,
) -> np.ndarray:
	"""Return the scalar Bessel-Gauss envelope without the signed vortex phase."""

	X, Y, _, _, _ = _grid_xy(grid)
	R = np.hypot(X, Y)
	order = abs(int(ell))
	return sp.jv(order, float(kr_m_inv) * R) * np.exp(-(R**2) / max(float(waist_m) ** 2, EPS))


def signed_vortex_components(amplitude: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	"""Return the ±beta scalar components from the same combined hologram phase."""

	amp = np.asarray(amplitude, dtype=complex)
	phase = np.asarray(beta, dtype=float)
	return (amp * np.exp(1j * phase), amp * np.exp(-1j * phase))


def build_dualpass_seed(amplitude: np.ndarray, beta: np.ndarray) -> VectorField:
	"""Build the pre-QWP dual-pass seed by applying paper eqs. 10-13."""

	amp = np.asarray(amplitude, dtype=complex)
	vf = VectorField(Ex=amp, Ey=np.zeros_like(amp, dtype=complex))
	for matrix in paper_dualpass_chain(beta):
		vf = apply_jones_matrix(vf, matrix)
	return vf


def apply_paper_optics(vf: VectorField, mode: str) -> VectorField:
	"""Apply paper eqs. 14-16 to a pre-QWP dual-pass seed field."""

	return apply_hardware_chain(vf, resolve_post_optics(mode, "paper_qwp"))


def build_paper_vector_field(amplitude: np.ndarray, beta: np.ndarray, mode: str) -> VectorField:
	"""Build the corrected vector field by applying the full paper chain."""

	vf = build_dualpass_seed(amplitude, beta)
	return apply_paper_optics(vf, mode)


def build_vector_mode_from_seed(
	grid: Mapping[str, Any],
	seed: VectorField,
	ell: int,
	kr_m_inv: float,
	mode: str,
	*,
	post_optics: str | Sequence[Any] | None = "paper_qwp",
	metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
	"""Return vector diagnostics for one pre-QWP seed field."""

	vf = apply_hardware_chain(seed, resolve_post_optics(mode, post_optics))
	stokes = stokes_parameters(vf)
	ellipse = polarization_ellipse_parameters(stokes)
	analyzer = analyzer_maps(vf)
	case = {
		"grid": grid,
		"ell": int(ell),
		"kr_m_inv": float(kr_m_inv),
		"mode": str(mode),
		"seed": seed,
		"field": vf,
		"total_intensity": total_intensity(vf),
		"analyzer": analyzer,
		"stokes": stokes,
		"ellipse": ellipse,
	}
	if metadata:
		case.update(dict(metadata))
	return case


def build_slm_encoded_vector_mode(
	grid: Mapping[str, Any],
	component_plus_charge: np.ndarray,
	component_minus_charge: np.ndarray,
	ell: int,
	kr_m_inv: float,
	mode: str,
	*,
	relative_phase_rad: float | None = None,
	hardware: str | Mapping[str, Any] | None = None,
	post_optics: str | Sequence[Any] | None = None,
	metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
	"""Return diagnostics for an SLM-programmed vector mode."""

	vf, program = _slm_encoded_field_and_program(
		component_plus_charge,
		component_minus_charge,
		mode,
		relative_phase_rad=relative_phase_rad,
		hardware=hardware,
		post_optics=post_optics,
	)
	stokes = stokes_parameters(vf)
	ellipse = polarization_ellipse_parameters(stokes)
	analyzer = analyzer_maps(vf)
	case = {
		"grid": grid,
		"ell": int(ell),
		"kr_m_inv": float(kr_m_inv),
		"mode": str(mode),
		"field": vf,
		"total_intensity": total_intensity(vf),
		"analyzer": analyzer,
		"stokes": stokes,
		"ellipse": ellipse,
		"component_basis": "circular",
		"relative_phase_rad": 0.5 * float(program.minus_phase_offset_rad - program.plus_phase_offset_rad),
		"encoder_hardware": program.hardware,
		"plus_phase_offset_rad": float(program.plus_phase_offset_rad),
		"minus_phase_offset_rad": float(program.minus_phase_offset_rad),
		"post_optics": program.post_optics_label,
		"hardware_note": program.notes,
	}
	if metadata:
		case.update(dict(metadata))
	return case


def build_analytic_vector_mode(
	grid: Mapping[str, Any],
	ell: int,
	kr_m_inv: float,
	waist_m: float,
	mode: str,
) -> dict[str, Any]:
	"""Return one ideal cylindrical-vector BG mode and its diagnostics."""

	beta = abs(int(ell)) * _phase_angle(grid)
	amplitude = scalar_bg_envelope(grid, ell, kr_m_inv, waist_m)
	seed = build_dualpass_seed(amplitude, beta)
	case = build_vector_mode_from_seed(
		grid,
		seed,
		ell,
		kr_m_inv,
		mode,
		post_optics="paper_qwp",
		metadata={
			"waist_m": float(waist_m),
			"beta": beta,
			"amplitude": amplitude,
		},
	)
	return case


def actual_lab_hardware(future_element: str = "none") -> LabVectorHardware:
	"""Return the confirmed Case-1 hardware description.

	The only optional knob is ``future_element``. Any value other than ``none``
	is treated as a simulation-only branch because that element is not on the
	current bench.
	"""

	key = str(future_element).lower().strip()
	if key not in {"none", "qplate", "periscope", "waveplate"}:
		raise ValueError("future_element must be one of 'none', 'qplate', 'periscope', or 'waveplate'.")
	return LabVectorHardware(future_element=key)


def _lab_target_key(target: str) -> str:
	key = str(target).lower().strip().replace("-", "_").replace(" ", "_")
	if key in {"case1", "case_1", "lab", "actual_lab", "achievable", "achievable_sop", "sop", "phase_reference"}:
		return "achievable_sop"
	if key in {"scalar", "scalar_bg", "vortex", "vortex_reference"}:
		return "achievable_sop"
	if key in {"radial", "azimuthal"}:
		return key
	raise ValueError("target must be 'achievable_sop', 'radial', or 'azimuthal'.")


def lab_vector_realizability(target: str, hardware: LabVectorHardware | Mapping[str, Any] | None = None) -> dict[str, Any]:
	"""Return the hardware-honesty decision for one requested vector target."""

	hw = actual_lab_hardware() if hardware is None else (
		hardware if isinstance(hardware, LabVectorHardware) else LabVectorHardware(**dict(hardware))
	)
	target_key = _lab_target_key(target)
	if target_key in {"radial", "azimuthal"}:
		if hw.future_element == "none":
			return {
				"lab_realizable": False,
				"simulation_only": False,
				"requires_element": "qplate/periscope/waveplate",
				"reason": (
					"Radial/azimuthal beams require a second polarization axis or a polarization-converting element. "
					"The confirmed Case-1 bench has two phase-only SLMs with the same director axis and no waveplates."
				),
			}
		return {
			"lab_realizable": False,
			"simulation_only": True,
			"requires_element": hw.future_element,
			"reason": f"SIMULATION ONLY - requires {hw.future_element}, not present on the current bench.",
		}
	return {
		"lab_realizable": bool(hw.future_element == "none"),
		"simulation_only": bool(hw.future_element != "none"),
		"requires_element": None if hw.future_element == "none" else hw.future_element,
		"reason": "Case-1 achievable class: H is shaped and V is the fixed reference.",
	}


def _carrier_phase(grid: Mapping[str, Any], carrier_lpmm: float) -> np.ndarray:
	X, _, _, _, _ = _grid_xy(grid)
	return 2.0 * np.pi * float(carrier_lpmm) * 1.0e3 * X


def _normalised_abs(values: np.ndarray) -> np.ndarray:
	arr = np.abs(np.asarray(values, dtype=complex))
	peak = float(np.nanmax(arr)) if arr.size else 0.0
	if peak <= EPS:
		return np.zeros_like(arr, dtype=float)
	return np.clip(arr / peak, 0.0, 1.0)


def _lab_phase_map(grid: Mapping[str, Any], ell: int) -> np.ndarray:
	return int(ell) * _phase_angle(grid)


def _method_b_phase_proxy(target_amplitude: np.ndarray, target_phase: np.ndarray, carrier_phase: np.ndarray) -> tuple[np.ndarray, float]:
	"""Return a computed off-axis phase-only proxy for Method B.

	This is not a calibrated SLM diffraction-efficiency model. I use the
	``arccos(A)`` phase-depth proxy so the exported CGH records the desired
	amplitude map and the reported first-order power fraction is computed from
	``mean(A^2)`` over the grid.
	"""

	A = np.clip(np.asarray(target_amplitude, dtype=float), 0.0, 1.0)
	phase = np.asarray(target_phase, dtype=float) + np.asarray(carrier_phase, dtype=float) + np.arccos(A)
	return phase, float(np.mean(A**2)) if A.size else 0.0


def _case1_cgh(
	grid: Mapping[str, Any],
	ell: int,
	kr_m_inv: float,
	waist_m: float,
	*,
	method: str,
	carrier_lpmm: float,
) -> dict[str, Any]:
	phase = _lab_phase_map(grid, ell)
	envelope = _normalised_abs(scalar_bg_envelope(grid, ell, kr_m_inv, waist_m))
	carrier = _carrier_phase(grid, carrier_lpmm)
	method_key = str(method).upper().strip()
	if method_key in {"A", "PHASE", "PHASE_ONLY"}:
		phase_map = phase + carrier
		encoded_power_fraction = 1.0
		method_label = "A"
	elif method_key in {"B", "COMPLEX", "COMPLEX_AMPLITUDE"}:
		phase_map, encoded_power_fraction = _method_b_phase_proxy(envelope, phase, carrier)
		method_label = "B"
	else:
		raise ValueError("method must be 'A'/'phase_only' or 'B'/'complex_amplitude'.")
	wrapped = holography_eq.wrap_phase_rad(phase_map)
	return {
		"method": method_label,
		"phase_rad": phase_map,
		"phase_wrapped_rad": wrapped,
		"gray": holography_eq.phase_to_gray(wrapped, bits=8),
		"target_amplitude": envelope,
		"target_phase_rad": phase,
		"carrier_lpmm": float(carrier_lpmm),
		"encoded_power_fraction": float(encoded_power_fraction),
	}


def _future_element_field(
	grid: Mapping[str, Any],
	ell: int,
	kr_m_inv: float,
	waist_m: float,
	target_key: str,
) -> VectorField:
	phi = _phase_angle(grid)
	amp = _normalised_abs(scalar_bg_envelope(grid, max(1, abs(int(ell))), kr_m_inv, waist_m))
	if target_key == "radial":
		return VectorField(Ex=amp * np.cos(phi), Ey=amp * np.sin(phi))
	if target_key == "azimuthal":
		return VectorField(Ex=-amp * np.sin(phi), Ey=amp * np.cos(phi))
	raise ValueError("future-element field target must be 'radial' or 'azimuthal'.")


def _propagate_vector_field(
	field: VectorField,
	grid: Mapping[str, Any],
	*,
	z_m: float,
	wavelength_m: float,
	n_medium: float,
) -> VectorField:
	if abs(float(z_m)) <= 0.0:
		return field
	prop_x = make_bl_asm_propagator(field.Ex, grid, wavelength_m, n_medium=n_medium, bandlimit=True)
	prop_y = make_bl_asm_propagator(field.Ey, grid, wavelength_m, n_medium=n_medium, bandlimit=True)
	return VectorField(Ex=np.asarray(prop_x(float(z_m)), dtype=complex), Ey=np.asarray(prop_y(float(z_m)), dtype=complex))


def lab_observable_from_analyzer(analyzer: Mapping[int, np.ndarray]) -> dict[str, np.ndarray]:
	"""Return the lab-observable Stokes subset from 0/45/90/135 analyzer frames.

	The current bench has no QWP analyzer, so I deliberately return only
	``S0``, ``S1``, ``S2``, and ``psi``. ``S3`` and ``chi`` remain internal
	field diagnostics, not measured quantities.
	"""

	I0 = np.asarray(analyzer[0], dtype=float)
	I45 = np.asarray(analyzer[45], dtype=float)
	I90 = np.asarray(analyzer[90], dtype=float)
	I135 = np.asarray(analyzer[135], dtype=float)
	S0 = I0 + I90
	S1 = I0 - I90
	S2 = I45 - I135
	psi = 0.5 * np.arctan2(S2, S1)
	return {"S0": S0, "S1": S1, "S2": S2, "psi": psi}


def build_actual_lab_vector_case(
	grid: Mapping[str, Any],
	ell: int,
	kr_m_inv: float,
	waist_m: float,
	*,
	target: str = "achievable_sop",
	method: str = "A",
	future_element: str = "none",
	carrier_lpmm: float = 0.0,
	reference_phase_rad: float = 0.0,
	z_m: float = 0.0,
	wavelength_m: float = 1029.0 * BT_NM,
	n_medium: float = 1.0,
	require_lab_realizable: bool = False,
	export_dir: str | Path | None = None,
	label: str | None = None,
) -> dict[str, Any]:
	"""Build the actual Case-1 lab vector/SOP prediction.

	Default hardware is the confirmed bench: both SLMs share the H director axis,
	all folds are in-plane, and no waveplates exist. The lab-realizable class is
	therefore a shaped H component interfering with a fixed V reference. Requests
	for radial/azimuthal beams are refused unless ``future_element`` is set, in
	which case the result is explicitly tagged simulation-only and excluded from
	measured-vs-predicted lab comparisons.
	"""

	hardware = actual_lab_hardware(future_element)
	target_key = _lab_target_key(target)
	realizability = lab_vector_realizability(target_key, hardware)
	method_raw = str(method).upper().strip()
	if method_raw in {"A", "PHASE", "PHASE_ONLY"}:
		method_key = "A"
	elif method_raw in {"B", "COMPLEX", "COMPLEX_AMPLITUDE"}:
		method_key = "B"
	else:
		raise ValueError("method must be 'A'/'phase_only' or 'B'/'complex_amplitude'.")
	cgh = _case1_cgh(grid, ell, kr_m_inv, waist_m, method=method_key, carrier_lpmm=carrier_lpmm)
	base_meta = {
		"grid": grid,
		"ell": int(ell),
		"kr_m_inv": float(kr_m_inv),
		"waist_m": float(waist_m),
		"target": target_key,
		"mode": target_key,
		"method": method_key,
		"hardware": asdict(hardware),
		"shaped_components": ["H"],
		"reference_components": ["V"],
		"reference_component_constant": True,
		"uses_waveplates": False,
		"uses_two_slm": True,
		"uses_shared_director_axis": bool(hardware.same_director_axis),
		"post_optics": "none",
		"lab_realizable": bool(realizability["lab_realizable"]),
		"simulation_only": bool(realizability["simulation_only"]),
		"requires_element": realizability["requires_element"],
		"hardware_note": realizability["reason"],
		"encoded_power_fraction": float(cgh["encoded_power_fraction"]),
		"cgh": cgh,
	}
	if target_key in {"radial", "azimuthal"} and hardware.future_element == "none":
		if require_lab_realizable:
			raise ValueError(realizability["reason"])
		return {
			**base_meta,
			"field": None,
			"total_intensity": None,
			"analyzer": {},
			"lab_observable": {},
			"internal_diagnostic": {},
			"refused": True,
			"comparison_included": False,
		}

	if target_key in {"radial", "azimuthal"}:
		field0 = _future_element_field(grid, ell, kr_m_inv, waist_m, target_key)
	else:
		phase = np.asarray(cgh["target_phase_rad"], dtype=float)
		amp = np.ones_like(phase, dtype=float)
		field0 = VectorField(
			Ex=amp * np.exp(1j * phase),
			Ey=amp * np.exp(1j * float(reference_phase_rad)),
		)
	field = _propagate_vector_field(field0, grid, z_m=z_m, wavelength_m=wavelength_m, n_medium=n_medium)
	analyzer = analyzer_maps(field)
	stokes = stokes_parameters(field)
	ellipse = polarization_ellipse_parameters(stokes)
	lab_observable = lab_observable_from_analyzer(analyzer)
	s0 = np.asarray(stokes["S0"], dtype=float)
	s1 = np.asarray(stokes["S1"], dtype=float)
	s1_rms = float(np.sqrt(np.mean((s1 / (s0 + EPS)) ** 2))) if s0.size else 0.0
	case = {
		**base_meta,
		"field": field,
		"field_input": field0,
		"total_intensity": total_intensity(field),
		"analyzer": analyzer,
		"stokes": stokes,
		"ellipse": ellipse,
		"lab_observable": lab_observable,
		"internal_diagnostic": {
			"S3": stokes["S3"],
			"chi": ellipse["chi"],
			"lab_observable": False,
			"requires_qwp": True,
		},
		"s1_balance_rms": s1_rms,
		"refused": False,
		"comparison_included": bool(realizability["lab_realizable"] and not realizability["simulation_only"]),
		"z_m": float(z_m),
		"z_um": float(z_m / BT_UM),
	}
	if export_dir is not None:
		case["export_paths"] = export_lab_vector_cgh(case, export_dir, label=label)
	return case


def export_lab_vector_cgh(case: Mapping[str, Any], output_dir: str | Path, *, label: str | None = None) -> dict[str, Path]:
	"""Write one loadable 8-bit phase map and JSON sidecar for a vector target."""

	out = Path(output_dir)
	out.mkdir(parents=True, exist_ok=True)
	name = str(label or f"lab_vector_{case.get('target', 'target')}_l{case.get('ell', 0)}_method_{case.get('method', 'A')}").lower()
	name = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in name)
	gray = np.asarray(case["cgh"]["gray"], dtype=np.uint8)
	png_path = out / f"{name}.png"
	try:
		from PIL import Image

		Image.fromarray(gray, mode="L").save(png_path)
	except Exception:  # pragma: no cover - fallback for small environments
		plt.imsave(png_path, gray, cmap="gray", vmin=0, vmax=255)
	params = {
		"target": case.get("target"),
		"ell": int(case.get("ell", 0)),
		"method": case.get("method"),
		"future_element": case.get("hardware", {}).get("future_element"),
		"lab_realizable": bool(case.get("lab_realizable", False)),
		"simulation_only": bool(case.get("simulation_only", False)),
		"requires_element": case.get("requires_element"),
		"encoded_power_fraction": float(case.get("encoded_power_fraction", np.nan)),
		"carrier_lpmm": float(case.get("cgh", {}).get("carrier_lpmm", 0.0)),
		"shaped_components": list(case.get("shaped_components", [])),
		"reference_components": list(case.get("reference_components", [])),
		"uses_waveplates": bool(case.get("uses_waveplates", False)),
		"uses_two_slm": bool(case.get("uses_two_slm", False)),
		"uses_shared_director_axis": bool(case.get("uses_shared_director_axis", False)),
		"hardware_note": case.get("hardware_note", ""),
	}
	json_path = out / f"{name}.json"
	json_path.write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")
	return {"phase_png": png_path, "params_json": json_path}


def image_correlation(A: np.ndarray, B: np.ndarray, mask: np.ndarray | None = None) -> float:
	"""Return Pearson correlation after mean removal, optionally in a mask."""

	if mask is None:
		a = np.asarray(A, dtype=float).ravel()
		b = np.asarray(B, dtype=float).ravel()
	else:
		a = np.asarray(A, dtype=float)[mask].ravel()
		b = np.asarray(B, dtype=float)[mask].ravel()
	if a.size < 8:
		return 1.0
	a = a - float(np.mean(a))
	b = b - float(np.mean(b))
	denom = float(np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b)))
	if denom <= EPS:
		return 1.0
	return float(np.sum(a * b) / denom)


def _normalise_for_harness(values: np.ndarray) -> np.ndarray:
	arr = np.maximum(np.asarray(values, dtype=float), 0.0)
	peak = float(np.max(arr)) if arr.size else 0.0
	return arr / peak if peak > EPS else np.zeros_like(arr)


def _frame_centroid(values: np.ndarray) -> tuple[float, float]:
	arr = np.maximum(np.asarray(values, dtype=float), 0.0)
	total = float(np.sum(arr))
	if total <= EPS:
		return (0.5 * (arr.shape[0] - 1), 0.5 * (arr.shape[1] - 1))
	yy, xx = np.indices(arr.shape)
	return (float(np.sum(yy * arr) / total), float(np.sum(xx * arr) / total))


def _register_frame_to_prediction(measured: np.ndarray, predicted: np.ndarray) -> np.ndarray:
	my, mx = _frame_centroid(measured)
	py, px = _frame_centroid(predicted)
	return ndimage.shift(np.asarray(measured, dtype=float), shift=(py - my, px - mx), order=1, mode="nearest")


def compare_lab_measured_to_predicted(
	case: Mapping[str, Any],
	measured_frames: Mapping[int, np.ndarray] | None = None,
	*,
	mask: np.ndarray | None = None,
	register: bool = True,
) -> dict[str, Any]:
	"""Compare Case-1 predicted analyzer frames with measured frames.

	The harness intentionally accepts only the lab-realizable Case-1 class. It
	returns blank/NaN slots when data are absent, and it excludes future-element
	or refused radial/azimuthal predictions from measured-vs-predicted claims.
	"""

	if not bool(case.get("comparison_included", False)):
		return {
			"comparison_included": False,
			"status": "excluded_simulation_only_or_unreachable",
			"per_angle_correlation": {angle: np.nan for angle in (0, 45, 90, 135)},
			"structural_match": np.nan,
			"psi_rms_error_rad": np.nan,
			"observables": ["S0", "S1", "S2", "psi"],
		}
	predicted = {int(k): _normalise_for_harness(v) for k, v in case["analyzer"].items() if int(k) in {0, 45, 90, 135}}
	if measured_frames is None:
		return {
			"comparison_included": True,
			"status": "no_measured_data",
			"per_angle_correlation": {angle: np.nan for angle in (0, 45, 90, 135)},
			"structural_match": np.nan,
			"psi_rms_error_rad": np.nan,
			"observables": ["S0", "S1", "S2", "psi"],
		}
	measured_registered: dict[int, np.ndarray] = {}
	corr: dict[int, float] = {}
	for angle in (0, 45, 90, 135):
		if angle not in measured_frames:
			raise KeyError(f"Missing measured analyzer frame {angle} deg.")
		meas = _normalise_for_harness(np.asarray(measured_frames[angle], dtype=float))
		meas = _register_frame_to_prediction(meas, predicted[angle]) if register else meas
		meas = _normalise_for_harness(meas)
		measured_registered[angle] = meas
		corr[angle] = image_correlation(predicted[angle], meas, mask)
	pred_obs = lab_observable_from_analyzer(predicted)
	meas_obs = lab_observable_from_analyzer(measured_registered)
	if mask is None:
		mask = pred_obs["S0"] > 0.05 * float(np.max(pred_obs["S0"]))
	delta = _wrap_headless(np.asarray(meas_obs["psi"], dtype=float) - np.asarray(pred_obs["psi"], dtype=float))
	psi_rms = float(np.sqrt(np.mean(delta[mask] ** 2))) if np.any(mask) else np.nan
	return {
		"comparison_included": True,
		"status": "compared",
		"per_angle_correlation": corr,
		"structural_match": float(np.nanmean(list(corr.values()))),
		"psi_rms_error_rad": psi_rms,
		"observables": ["S0", "S1", "S2", "psi"],
		"measured_registered": measured_registered,
	}


def phase_winding_estimate(field: np.ndarray, grid: Mapping[str, Any], radius_m: float, n_samples: int = 720) -> float:
	"""Estimate vortex winding by sampling phase around a circular contour."""

	_, _, x, y, dx = _grid_xy(grid)
	U = np.asarray(field, dtype=complex)
	theta = np.linspace(0.0, 2.0 * np.pi, int(n_samples), endpoint=True)
	xs = float(radius_m) * np.cos(theta)
	ys = float(radius_m) * np.sin(theta)
	x0 = float(x[0])
	y0 = float(y[0])
	ix = np.clip(np.round((xs - x0) / dx).astype(int), 0, U.shape[1] - 1)
	iy = np.clip(np.round((ys - y0) / dx).astype(int), 0, U.shape[0] - 1)
	phase = np.unwrap(np.angle(U[iy, ix]))
	return float((phase[-1] - phase[0]) / (2.0 * np.pi))


def conj_match_report(vf: VectorField, mask: np.ndarray) -> tuple[complex, float]:
	"""Report how well ``Ey`` matches ``c*conj(Ex)`` inside a mask."""

	roi = np.asarray(mask, dtype=bool)
	ref = np.conj(np.asarray(vf.Ex, dtype=complex)[roi])
	tgt = np.asarray(vf.Ey, dtype=complex)[roi]
	c = np.vdot(ref, tgt) / (np.vdot(ref, ref) + EPS)
	err = np.linalg.norm(tgt - c * ref) / (np.linalg.norm(tgt) + EPS)
	return complex(c), float(err)


def total_intensity(vf: VectorField) -> np.ndarray:
	"""Return total intensity S0 for one vector field."""

	return np.abs(vf.Ex) ** 2 + np.abs(vf.Ey) ** 2


def stokes_parameters(vf: VectorField) -> dict[str, np.ndarray]:
	"""Return Stokes S0-S3 in the linear x/y basis."""

	return jones_eq.stokes_from_linear_components(_complex_array(vf.Ex), _complex_array(vf.Ey))


def polarization_ellipse_parameters(stokes: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
	"""Return orientation/ellipticity using the corrected paper eq. 17 convention."""

	S0 = np.asarray(stokes["S0"], dtype=float)
	angles = jones_eq.ellipse_angles_from_stokes(stokes)
	psi = angles["psi_rad"]
	# The paper prints sin(2 psi)=S3/S0, but the ellipse relation is
	# sin(2 chi)=S3/S0. I implement the corrected convention here.
	chi = angles["chi_rad"]
	S1 = np.asarray(stokes["S1"], dtype=float)
	S2 = np.asarray(stokes["S2"], dtype=float)
	S3 = np.asarray(stokes["S3"], dtype=float)
	dolp = np.sqrt(S1**2 + S2**2) / (S0 + EPS)
	docp = np.abs(S3) / (S0 + EPS)
	return {"psi": psi, "chi": chi, "dolp": dolp, "docp": docp}


def linear_analyzer_intensity(vf: VectorField, theta_deg: float) -> np.ndarray:
	"""Return intensity after an ideal linear analyzer at `theta_deg`."""

	return jones_eq.linear_analyzer_intensity(vf.Ex, vf.Ey, np.deg2rad(float(theta_deg)))


def analyzer_maps(vf: VectorField, angles_deg: Sequence[int] = (0, 45, 90, 135)) -> dict[int, np.ndarray]:
	"""Return the four analyzer maps used in Baliyan-Nishchal figs. 7-10."""

	return {int(angle): linear_analyzer_intensity(vf, float(angle)) for angle in angles_deg}


def predicted_ring_radius(ell: int, kr_m_inv: float) -> float:
	"""Return the first bright-ring radius used for ring-based diagnostics."""

	order = abs(int(ell))
	if order == 0:
		return scalar_eq.j0_first_null_radius_m(float(kr_m_inv))
	return scalar_eq.ring_radius_from_jprime_zero_m(order, float(kr_m_inv))


def ring_roi(grid: Mapping[str, Any], radius_m: float, rel_width: float = 0.18) -> np.ndarray:
	"""Return a broad annular mask around the predicted bright ring."""

	X, Y, _, _, dx = _grid_xy(grid)
	R = np.hypot(X, Y)
	r_lo = max(0.0, (1.0 - float(rel_width)) * float(radius_m))
	r_hi = (1.0 + float(rel_width)) * float(radius_m)
	roi = (R >= r_lo) & (R <= r_hi)
	if np.count_nonzero(roi) < 32:
		roi = (R >= max(0.0, 0.5 * float(radius_m))) & (R <= max(1.5 * float(radius_m), 3.0 * dx))
	return roi


def petal_modulation_metric(I: np.ndarray, roi: np.ndarray) -> float:
	"""Return the ring-ROI standard-deviation to mean modulation metric."""

	values = np.asarray(I[roi], dtype=float)
	mean = float(np.mean(values)) if values.size else 0.0
	if mean <= EPS:
		return 0.0
	return float(np.std(values) / mean)


def sample_angular_profile(
	image: np.ndarray,
	grid: Mapping[str, Any],
	radius_m: float,
	*,
	rel_width: float = 0.18,
	radial_samples: int = 5,
	angular_samples: int = 720,
) -> tuple[np.ndarray, np.ndarray]:
	"""Sample the mean ring intensity as a function of azimuth."""

	_, _, x, _, dx = _grid_xy(grid)
	image_f = np.asarray(image, dtype=float)
	phi = np.linspace(0.0, 2.0 * np.pi, int(angular_samples), endpoint=False)
	radii = np.linspace((1.0 - rel_width) * radius_m, (1.0 + rel_width) * radius_m, max(1, int(radial_samples)))
	samples = np.zeros((len(radii), len(phi)), dtype=float)
	x0 = float(x[0])
	for ridx, radius in enumerate(radii):
		xs = radius * np.cos(phi)
		ys = radius * np.sin(phi)
		ix = np.clip(np.round((xs - x0) / dx).astype(int), 0, image_f.shape[1] - 1)
		iy = np.clip(np.round((ys - x0) / dx).astype(int), 0, image_f.shape[0] - 1)
		samples[ridx, :] = image_f[iy, ix]
	return phi, np.mean(samples, axis=0)


def petal_count_and_orientation(
	image: np.ndarray,
	grid: Mapping[str, Any],
	radius_m: float,
	*,
	rel_width: float = 0.18,
	angular_samples: int = 720,
) -> dict[str, float]:
	"""Estimate petal count and principal orientation from the ring angular FFT."""

	phi, profile = sample_angular_profile(
		image,
		grid,
		radius_m,
		rel_width=rel_width,
		angular_samples=angular_samples,
	)
	centered = profile - float(np.mean(profile))
	coeffs = np.fft.rfft(centered)
	amps = np.abs(coeffs)
	if amps.size <= 1:
		return {"petal_count": 0.0, "harmonic": 0.0, "orientation_rad": 0.0}
	amps[0] = 0.0
	harmonic = int(np.argmax(amps[1:]) + 1)
	coeff = coeffs[harmonic]
	orientation = float((-np.angle(coeff) / harmonic) % (2.0 * np.pi / harmonic))
	return {
		"petal_count": float(harmonic),
		"harmonic": float(harmonic),
		"orientation_rad": orientation,
		"profile_mean": float(np.mean(profile)),
		"profile_std": float(np.std(profile)),
		"angular_samples": float(len(phi)),
	}


def mode_orientation_error(
	psi: np.ndarray,
	grid: Mapping[str, Any],
	roi: np.ndarray,
	*,
	target: str,
	order: int = 1,
) -> float:
	"""Return RMS headless-angle error against the radial or azimuthal target."""

	phi = _phase_angle(grid)
	order_f = float(abs(int(order)))
	target_l = str(target).lower().strip()
	if target_l == "radial":
		ref = order_f * phi
	elif target_l == "azimuthal":
		ref = order_f * phi + 0.5 * np.pi
	else:
		raise ValueError("target must be 'radial' or 'azimuthal'.")
	delta = _wrap_headless(psi - ref)
	masked = np.asarray(delta[roi], dtype=float)
	return float(np.sqrt(np.mean(masked**2))) if masked.size else 0.0


def circularity_residual(stokes: Mapping[str, np.ndarray], roi: np.ndarray) -> dict[str, float]:
	"""Return ring-ROI residual circularity statistics for |S3|/S0."""

	S0 = np.asarray(stokes["S0"], dtype=float)
	S3 = np.asarray(stokes["S3"], dtype=float)
	ratio = np.abs(S3) / (S0 + EPS)
	masked = np.asarray(ratio[roi], dtype=float)
	return {
		"mean_abs_s3_over_s0": float(np.mean(masked)) if masked.size else 0.0,
		"max_abs_s3_over_s0": float(np.max(masked)) if masked.size else 0.0,
		"median_abs_s3_over_s0": float(np.median(masked)) if masked.size else 0.0,
	}


def plot_total_and_analyzer_panel(
	field: VectorField,
	grid: Mapping[str, Any],
	*,
	title: str | None = None,
	gamma: float = 0.45,
) -> plt.Figure:
	"""Plot total intensity plus the four standard analyzer maps."""

	vbb_style.apply_style()
	analyzer = analyzer_maps(field)
	images = [total_intensity(field), analyzer[0], analyzer[45], analyzer[90], analyzer[135]]
	labels = ["total", "0 deg", "45 deg", "90 deg", "135 deg"]
	fig, axes = plt.subplots(1, 5, figsize=(16.8, 3.6), constrained_layout=True)
	extent = _extent_um(grid)
	artist = None
	for axis, image, label in zip(axes, images, labels):
		artist = axis.imshow(
			vbb_style.display_scale(image, gamma=gamma),
			origin="lower",
			extent=extent,
			cmap=vbb_style.INTENSITY_CMAP,
			vmin=0.0,
			vmax=1.0,
		)
		axis.set_title(label)
		axis.set_xlabel("x [um, sample plane]")
		axis.set_ylabel("y [um, sample plane]")
		axis.grid(False)
	if artist is not None:
		cbar = fig.colorbar(artist, ax=axes, shrink=0.94, pad=0.01)
		cbar.set_label("display intensity [a.u.]")
	if title:
		fig.suptitle(title, fontsize=12)
	return fig


def plot_analyzer_family_grid(
	cases: Sequence[Mapping[str, Any]],
	*,
	gamma: float = 0.45,
) -> plt.Figure:
	"""Plot one row per case with total intensity and four analyzer maps."""

	if not cases:
		raise ValueError("At least one case is required.")
	vbb_style.apply_style()
	fig, axes = plt.subplots(len(cases), 5, figsize=(16.8, 3.2 * len(cases)), constrained_layout=True)
	if len(cases) == 1:
		axes = np.asarray([axes])
	artist = None
	labels = ["total", "0 deg", "45 deg", "90 deg", "135 deg"]
	for row, case in enumerate(cases):
		grid = case["grid"]
		field = case["field"]
		analyzer = case["analyzer"]
		images = [case["total_intensity"], analyzer[0], analyzer[45], analyzer[90], analyzer[135]]
		extent = _extent_um(grid)
		row_label = f"{case['mode']} | l={case['ell']}"
		for col, (axis, image, label) in enumerate(zip(axes[row], images, labels)):
			artist = axis.imshow(
				vbb_style.display_scale(image, gamma=gamma),
				origin="lower",
				extent=extent,
				cmap=vbb_style.INTENSITY_CMAP,
				vmin=0.0,
				vmax=1.0,
			)
			axis.set_title(label if row == 0 else "")
			axis.set_xlabel("x [um, sample plane]")
			axis.grid(False)
			if col == 0:
				axis.set_ylabel(f"{row_label}\ny [um, sample plane]")
			else:
				axis.set_ylabel("y [um, sample plane]")
	if artist is not None:
		cbar = fig.colorbar(artist, ax=axes, shrink=0.96, pad=0.01)
		cbar.set_label("display intensity [a.u.]")
	fig.suptitle("Total intensity and analyzer maps", fontsize=12)
	return fig


def plot_polarization_quiver(
	case: Mapping[str, Any],
	*,
	step: int = 8,
	gamma: float = 0.45,
	intensity_floor: float = 0.18,
) -> plt.Figure:
	"""Plot a dense regular-grid polarization field on the bright ring ROI."""

	vbb_style.apply_style()
	grid = case["grid"]
	total = np.asarray(case["total_intensity"], dtype=float)
	psi = np.asarray(case["ellipse"]["psi"], dtype=float)
	_, _, x, y, _ = _grid_xy(grid)
	R = np.asarray(grid["R"], dtype=float)
	ring_source = np.where(total >= intensity_floor * float(np.max(total)), total, 0.0)
	if np.any(ring_source > 0.0):
		ring_radius = float(R.ravel()[int(np.argmax(ring_source.ravel()))])
		ring_width = max(3.0 * float(grid.get("dx", np.median(np.diff(x)))), 0.25 * max(ring_radius, BT_EPS))
		ring_mask = np.abs(R - ring_radius) <= ring_width
	else:
		ring_mask = np.ones_like(total, dtype=bool)
	Xs, Ys = np.meshgrid(x[::step] / BT_UM, y[::step] / BT_UM, indexing="xy")
	psi_s = psi[::step, ::step]
	mask = (total[::step, ::step] >= intensity_floor * float(np.max(total))) & ring_mask[::step, ::step]
	U = np.cos(psi_s)
	V = np.sin(psi_s)
	U = np.where(mask, U, np.nan)
	V = np.where(mask, V, np.nan)
	fig, ax = plt.subplots(1, 1, figsize=(5.1, 4.6), constrained_layout=True)
	image = ax.imshow(
		vbb_style.display_scale(total, gamma=gamma),
		origin="lower",
		extent=_extent_um(grid),
		cmap=vbb_style.INTENSITY_CMAP,
		vmin=0.0,
		vmax=1.0,
	)
	ax.quiver(
		Xs,
		Ys,
		U,
		V,
		color="#56B4E9",
		pivot="mid",
		headwidth=0.0,
		headlength=0.0,
		headaxislength=0.0,
		scale=18,
		width=0.0045,
	)
	ax.set_title(f"{case['mode']} | l={case['ell']} | polarization quiver")
	ax.set_xlabel("x [um, sample plane]")
	ax.set_ylabel("y [um, sample plane]")
	ax.grid(False)
	cbar = fig.colorbar(image, ax=ax, pad=0.02)
	cbar.set_label("display intensity [a.u.]")
	return fig


def export_figure(fig: plt.Figure, figure_path: str | Path, caption: str, *, metadata: Mapping[str, Any] | None = None) -> Path:
	"""Save one vector-physics figure with the shared caption flow."""

	return vbb_style.save_figure(fig, figure_path, caption, metadata=metadata)


__all__ = [
	"EPS",
	"LabVectorHardware",
	"SLMVectorProgram",
	"VectorField",
	"actual_lab_hardware",
	"analyzer_maps",
	"apply_slm_phase_program",
	"apply_paper_optics",
	"apply_jones_matrix",
	"apply_hardware_chain",
	"build_actual_lab_vector_case",
	"build_analytic_vector_mode",
	"build_vector_mode_from_seed",
	"build_slm_encoded_vector_field",
	"build_slm_encoded_vector_mode",
	"build_dualpass_seed",
	"build_paper_vector_field",
	"circular_components_to_linear",
	"circularity_residual",
	"compare_lab_measured_to_predicted",
	"compose_jones_matrices",
	"conj_match_report",
	"export_lab_vector_cgh",
	"export_figure",
	"first_slm_half_matrix",
	"hwp_45_swap_matrix",
	"image_correlation",
	"lab_observable_from_analyzer",
	"lab_vector_realizability",
	"linear_analyzer_intensity",
	"linear_polarizer_45_matrix",
	"mode_orientation_error",
	"paper_dualpass_chain",
	"paper_full_chain",
	"petal_count_and_orientation",
	"petal_modulation_metric",
	"phase_winding_estimate",
	"plot_analyzer_family_grid",
	"plot_polarization_quiver",
	"plot_total_and_analyzer_panel",
	"polarization_ellipse_parameters",
	"predicted_ring_radius",
	"qwp1_matrix",
	"qwp2_matrix",
	"resolve_slm_encoder_program",
	"resolve_post_optics",
	"reflection_helicity_flip_matrix",
	"ring_roi",
	"sample_angular_profile",
	"scalar_bg_envelope",
	"second_slm_half_matrix",
	"signed_vortex_components",
	"stokes_parameters",
	"total_intensity",
]
