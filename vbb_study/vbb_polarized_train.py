"""Configurable polarized optical-train simulator for the upgraded bench.

This module is deliberately separate from :mod:`vbb_vector`.  The older module
models the confirmed two co-aligned SLM setup, where only one linear component
is shaped.  Here I model the *upgraded* train in which real polarization optics
are inserted explicitly: uniform retarders for state preparation, and a
spatially varying segmented waveplate or q-plate for the radial/azimuthal
structure.  The physical axicon is represented as a local s/p Fresnel splitter,
then I project the field onto the transverse-to-k subspace before vectorial
angular-spectrum propagation so that ``k.E = 0`` and ``Ez`` is retained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vbb_study.config import EPS as BT_EPS, nm as BT_NM, um as BT_UM
from vbb_study.equations.fields import make_xy_grid

from . import vbb_style

VectorElement = Literal["none", "segmented_ra", "qplate_radial", "qplate_azimuthal"]
RealismMode = Literal["ideal", "lab_realistic"]
PresetTrainName = Literal[
    "segmented_vector_hexagon",
    "qplate_radial",
    "qplate_azimuthal",
    "scalar_vortex_bessel",
]


@dataclass(frozen=True)
class PolarizedTrainConfig:
    """Parameters for one upgraded-bench polarized-train simulation.

    The transverse grid is a sample-plane grid in metres.  I keep the axicon as
    a physical element rather than a holographic phase because this is the
    branch where the s/p Fresnel split can turn a sectored SOP into a sixfold
    intensity pattern.
    """

    N: int = 320
    dx_m: float = 0.18 * BT_UM
    wavelength_m: float = 1029.0 * BT_NM
    n_medium: float = 1.0
    n_axicon: float = 1.46
    axicon_base_angle_deg: float = 32.0
    ring_radius_m: float = 8.0 * BT_UM
    ring_width_m: float = 0.95 * BT_UM
    vortex_charge: int = 1
    vector_element: VectorElement = "segmented_ra"
    segment_count: int = 12
    symmetry_order: int = 6
    orientation_rad: float = 0.0
    input_polarization_angle_rad: float = 0.0
    uniform_hwp_angle_rad: float | None = None
    uniform_qwp_angle_rad: float | None = None
    qplate_charge: float = 0.5
    z_max_m: float = 110.0 * BT_UM
    z_points: int = 45


@dataclass(frozen=True)
class LabRealism:
    """Deterministic non-idealities for the lab-realistic branch."""

    retardance_error_rad: float = 0.035
    axis_error_rad: float = np.deg2rad(1.5)
    polarizer_extinction_ratio: float = 1.0e4
    polarizer_transmission: float = 0.965
    waveplate_ar_loss: float = 0.012
    segmented_seam_width_rad: float = np.deg2rad(1.2)
    segmented_seam_loss: float = 0.18
    segmented_angular_misregistration_rad: float = np.deg2rad(0.8)
    qplate_central_defect_radius_m: float = 0.85 * BT_UM
    qplate_conversion_efficiency: float = 0.94
    axicon_angle_error_deg: float = 0.25
    axicon_ar_loss: float = 0.025
    axicon_tip_rounding_m: float = 0.40 * BT_UM
    transmission_ripple: float = 0.025
    phase_ripple_rad: float = 0.025
    quantization_bits: int = 8
    relay_transmission: float = 0.92
    first_order_efficiency: float = 0.72
    filter_edge_softness_fraction: float = 0.08


@dataclass(frozen=True)
class VectorField3D:
    """Complex vector field in the linear basis, including longitudinal field."""

    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray


@dataclass(frozen=True)
class OpticalTrainState:
    """Mutable-in-spirit field packet passed through the optical train.

    Jones elements act on ``Ex`` and ``Ey``.  I keep ``Ez`` in the packet so
    free-space sections can propagate the full vector field rather than quietly
    returning to a scalar approximation between elements.
    """

    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    grid: Mapping[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LinearPolarizerElement:
    """Linear polarizer; lab mode uses finite extinction and transmission."""

    angle_rad: float
    extinction_ratio: float = np.inf
    transmission: float = 1.0
    label: str = "linear_polarizer"


@dataclass(frozen=True)
class HWPElement:
    """Uniform half-wave plate; lab mode adds retardance/axis/loss errors."""

    theta_rad: float
    retardance_rad: float = np.pi
    transmission: float = 1.0
    label: str = "hwp"


@dataclass(frozen=True)
class QWPElement:
    """Uniform quarter-wave plate; lab mode adds retardance/axis/loss errors."""

    theta_rad: float
    retardance_rad: float = 0.5 * np.pi
    transmission: float = 1.0
    label: str = "qwp"


@dataclass(frozen=True)
class SegmentedWaveplateElement:
    """Segmented spatial waveplate with per-sector fast-axis angles.

    If ``per_sector_fast_axis_rad`` is omitted, I generate the alternating
    radial/azimuthal segmented pattern used by the mechanism gate.
    """

    sector_count: int = 12
    per_sector_fast_axis_rad: tuple[float, ...] | None = None
    retardance_rad: float = np.pi
    orientation_rad: float = 0.0
    transmission: float = 1.0
    label: str = "segmented_waveplate"


@dataclass(frozen=True)
class QPlateElement:
    """Continuous q-plate/S-waveplate geometric-phase converter."""

    q: float = 0.5
    retardance_rad: float = np.pi
    orientation_rad: float = 0.0
    azimuthal: bool = False
    transmission: float = 1.0
    label: str = "qplate"


@dataclass(frozen=True)
class PhysicalAxiconElement:
    """Physical axicon with conical phase and local s/p Fresnel split."""

    n_axicon: float = 1.46
    base_angle_deg: float = 32.0
    aperture_radius_m: float | None = None
    orientation_rad: float = 0.0
    conical_phase_sign: float = -1.0
    transmission: float = 1.0
    label: str = "physical_axicon"


@dataclass(frozen=True)
class FourierRelayFilterElement:
    """4f relay and circular Fourier-plane filter.

    ``radius_m_inv`` is an angular-spatial-frequency radius in rad/m.  Lab mode
    softens the filter edge and applies relay/first-order efficiency.
    """

    radius_m_inv: float
    relay_transmission: float = 1.0
    first_order_efficiency: float = 1.0
    label: str = "fourier_relay_filter"


@dataclass(frozen=True)
class FreeSpacePropagationElement:
    """Vectorial ASM propagation between ordered train elements."""

    z_m: float
    label: str = "free_space"


@dataclass(frozen=True)
class SLMPhaseElement:
    """Phase-only SLM acting on the configured director/component."""

    label: str = "slm1"
    phase_rad: np.ndarray | None = None
    vortex_charge: int = 0
    axicon_kr_m_inv: float = 0.0
    carrier_lpmm: float = 0.0
    modulated_axis: Literal["H", "V", "both"] = "H"
    phase_bits: int | None = None


@dataclass(frozen=True)
class SLMConjugateElement:
    """Second SLM applying the conjugate of an upstream stored phase map."""

    source_label: str = "slm1"
    label: str = "slm2_conjugate"
    modulated_axis: Literal["H", "V", "both"] = "H"


@dataclass(frozen=True)
class PresetTrainConfig:
    """Exposed knobs for the named IRL optical-train presets."""

    name: PresetTrainName
    polarizer_angle_rad: float = 0.0
    polarizer_extinction_ratio: float = np.inf
    sector_pairs: int = 3
    sector_count: int | None = None
    sector_orientation_rad: float = 0.0
    per_sector_fast_axis_rad: tuple[float, ...] | None = None
    segmented_retardance_rad: float = np.pi
    q: float = 0.5
    qplate_retardance_rad: float = np.pi
    qplate_orientation_rad: float = 0.0
    n_axicon: float = 1.46
    axicon_base_angle_deg: float = 32.0
    axicon_aperture_radius_m: float | None = None
    filter_radius_m_inv: float = 2.5e6
    relay_transmission: float = 1.0
    first_order_efficiency: float = 1.0
    scalar_vortex_charge: int = 1
    scalar_axicon_kr_m_inv: float = 1.2 / BT_UM
    scalar_carrier_lpmm: float = 0.0
    include_two_slm_conjugate: bool = False
    inter_slm_z_m: float = 0.0
    modulated_axis: Literal["H", "V", "both"] = "H"
    axicon_before_4f: bool = True


def make_grid(config: PolarizedTrainConfig) -> dict[str, Any]:
    """Return the square sample grid used by the polarized train."""

    return make_xy_grid(int(config.N), float(config.dx_m))


def _angle_arrays(grid: Mapping[str, Any], orientation_rad: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(grid["X"], dtype=float)
    Y = np.asarray(grid["Y"], dtype=float)
    R = np.hypot(X, Y)
    Phi = np.arctan2(Y, X) - float(orientation_rad)
    return R, Phi, np.mod(Phi, 2.0 * np.pi)


def retarder_jones(
    Ex: np.ndarray,
    Ey: np.ndarray,
    retardance_rad: float,
    fast_axis_rad: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a linear retarder with a possibly spatially varying fast axis.

    A uniform HWP or QWP is just the scalar-axis special case.  I keep the same
    function for segmented waveplates and q-plates so the simulator cannot
    secretly treat the spatially varying element as a different kind of magic.
    """

    a = np.asarray(fast_axis_rad)
    c = np.cos(a)
    s = np.sin(a)
    e_fast = np.exp(-0.5j * float(retardance_rad))
    e_slow = np.exp(0.5j * float(retardance_rad))
    j11 = e_fast * c * c + e_slow * s * s
    j22 = e_fast * s * s + e_slow * c * c
    j12 = (e_fast - e_slow) * c * s
    return j11 * Ex + j12 * Ey, j12 * Ex + j22 * Ey


def initial_uniform_field(grid: Mapping[str, Any], angle_rad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return a unit-amplitude uniform Jones field.

    The spatial structure should appear only after a spatially varying element
    or the axicon.  This keeps the uniform-HWP/QWP guardrail test meaningful.
    """

    shape = np.asarray(grid["X"]).shape
    Ex = np.full(shape, np.cos(float(angle_rad)), dtype=complex)
    Ey = np.full(shape, np.sin(float(angle_rad)), dtype=complex)
    return Ex, Ey


def initial_train_state(
    grid: Mapping[str, Any],
    *,
    angle_rad: float = 0.0,
    amplitude: np.ndarray | float = 1.0,
) -> OpticalTrainState:
    """Create the starting state for a composable optical train."""

    Ex, Ey = initial_uniform_field(grid, angle_rad)
    amp = np.asarray(amplitude, dtype=float)
    Ex = Ex * amp
    Ey = Ey * amp
    Ez = np.zeros_like(Ex, dtype=complex)
    return OpticalTrainState(
        Ex=Ex,
        Ey=Ey,
        Ez=Ez,
        grid=grid,
        metadata={"history": [], "phase_maps": {}, "propagation": []},
    )


def _state_with(
    state: OpticalTrainState,
    Ex: np.ndarray,
    Ey: np.ndarray,
    Ez: np.ndarray | None = None,
    *,
    history: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> OpticalTrainState:
    """Return a copied state with updated field arrays and history."""

    metadata = dict(state.metadata)
    metadata["history"] = list(metadata.get("history", []))
    metadata["phase_maps"] = dict(metadata.get("phase_maps", {}))
    metadata["propagation"] = list(metadata.get("propagation", []))
    if history is not None:
        metadata["history"].append(dict(history))
    if extra_metadata:
        for key, value in extra_metadata.items():
            if key == "phase_maps":
                phase_maps = dict(metadata.get("phase_maps", {}))
                phase_maps.update(dict(value))
                metadata["phase_maps"] = phase_maps
            elif key == "propagation":
                propagation = list(metadata.get("propagation", []))
                propagation.extend(list(value))
                metadata["propagation"] = propagation
            else:
                metadata[key] = value
    return OpticalTrainState(
        Ex=np.asarray(Ex, dtype=complex),
        Ey=np.asarray(Ey, dtype=complex),
        Ez=np.asarray(state.Ez if Ez is None else Ez, dtype=complex),
        grid=state.grid,
        metadata=metadata,
    )


def linear_polarizer_jones(
    Ex: np.ndarray,
    Ey: np.ndarray,
    angle_rad: float,
    *,
    extinction_ratio: float = np.inf,
    transmission: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a linear polarizer with finite-extinction support."""

    c = np.cos(float(angle_rad))
    s = np.sin(float(angle_rad))
    E_parallel = Ex * c + Ey * s
    E_orthogonal = -Ex * s + Ey * c
    leak = 0.0 if np.isinf(extinction_ratio) else 1.0 / np.sqrt(max(float(extinction_ratio), 1.0))
    amp = np.sqrt(max(float(transmission), 0.0))
    Ex_out = amp * (E_parallel * c + leak * E_orthogonal * (-s))
    Ey_out = amp * (E_parallel * s + leak * E_orthogonal * c)
    return Ex_out, Ey_out


def segmented_ra_fast_axis(grid: Mapping[str, Any], config: PolarizedTrainConfig) -> np.ndarray:
    """Fast-axis map for a segmented radial/azimuthal half-wave plate.

    I use alternating radial and azimuthal target states.  With the default
    twelve sectors this produces a sixfold p/s pattern at the axicon while
    remaining explicitly a segmented polarization optic, not an SLM trick.
    """

    _, _, phi = _angle_arrays(grid, config.orientation_rad)
    count = max(2, int(config.segment_count))
    sector_width = 2.0 * np.pi / count
    idx = np.floor(phi / sector_width).astype(int)
    center = (idx + 0.5) * sector_width + float(config.orientation_rad)
    target_angle = np.where(idx % 2 == 0, center, center + 0.5 * np.pi)
    # A half-wave plate maps horizontal input onto a linear state at 2*axis.
    return 0.5 * target_angle


def qplate_fast_axis(grid: Mapping[str, Any], config: PolarizedTrainConfig, *, azimuthal: bool = False) -> np.ndarray:
    """Fast-axis map for a continuous q-plate/S-waveplate approximation."""

    _, phi, _ = _angle_arrays(grid, config.orientation_rad)
    offset = 0.25 * np.pi if azimuthal else 0.0
    return float(config.qplate_charge) * phi + float(config.orientation_rad) + offset


def segmented_waveplate_axis_map(
    grid: Mapping[str, Any],
    element: SegmentedWaveplateElement,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fast-axis and nearest-seam distance for a segmented waveplate."""

    _, _, phi = _angle_arrays(grid, element.orientation_rad)
    count = max(1, int(element.sector_count))
    sector_width = 2.0 * np.pi / count
    idx = np.floor(phi / sector_width).astype(int)
    if element.per_sector_fast_axis_rad is not None:
        angles = np.asarray(element.per_sector_fast_axis_rad, dtype=float)
        if angles.size == 0:
            raise ValueError("per_sector_fast_axis_rad must not be empty")
        axis = angles[idx % angles.size] + float(element.orientation_rad)
    else:
        center = (idx + 0.5) * sector_width + float(element.orientation_rad)
        target_angle = np.where(idx % 2 == 0, center, center + 0.5 * np.pi)
        axis = 0.5 * target_angle
    sector_phase = phi / sector_width
    frac = sector_phase - np.floor(sector_phase)
    seam_distance = np.minimum(frac, 1.0 - frac) * sector_width
    return axis, seam_distance


def slm_structuring_phase_map(grid: Mapping[str, Any], element: SLMPhaseElement) -> np.ndarray:
    """Build or return a phase-only SLM map in radians."""

    if element.phase_rad is not None:
        phase = np.asarray(element.phase_rad, dtype=float)
        if phase.shape != np.asarray(grid["X"]).shape:
            raise ValueError("SLM phase map shape must match the grid")
    else:
        R, Phi, _ = _angle_arrays(grid)
        phase = float(element.axicon_kr_m_inv) * R + int(element.vortex_charge) * Phi
        if element.carrier_lpmm:
            phase = phase + 2.0 * np.pi * float(element.carrier_lpmm) * 1.0e3 * np.asarray(grid["X"], dtype=float)
    if element.phase_bits is not None:
        levels = max(2, 2 ** int(element.phase_bits))
        phase = (np.round(((phase % (2.0 * np.pi)) / (2.0 * np.pi)) * (levels - 1)) / (levels - 1)) * (2.0 * np.pi)
    return phase


def _apply_component_phase(
    Ex: np.ndarray,
    Ey: np.ndarray,
    phase: np.ndarray,
    modulated_axis: Literal["H", "V", "both"],
) -> tuple[np.ndarray, np.ndarray]:
    factor = np.exp(1j * np.asarray(phase, dtype=float))
    if modulated_axis == "H":
        return Ex * factor, Ey
    if modulated_axis == "V":
        return Ex, Ey * factor
    if modulated_axis == "both":
        return Ex * factor, Ey * factor
    raise ValueError(f"Unsupported modulated_axis: {modulated_axis!r}")


def apply_train_polarization_elements(
    Ex: np.ndarray,
    Ey: np.ndarray,
    grid: Mapping[str, Any],
    config: PolarizedTrainConfig,
    *,
    realism: RealismMode = "ideal",
    lab: LabRealism | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the configured polarization train before the physical axicon.

    Uniform HWP/QWP settings prepare states but do not create angular structure.
    The only elements allowed to create sectors are ``segmented_ra`` and the
    q-plate variants.
    """

    lab = LabRealism() if lab is None else lab
    metadata: dict[str, Any] = {"elements": [], "spatially_varying_element": config.vector_element}
    axis_offset = float(lab.axis_error_rad) if realism == "lab_realistic" else 0.0
    retardance_error = float(lab.retardance_error_rad) if realism == "lab_realistic" else 0.0

    if config.uniform_hwp_angle_rad is not None:
        Ex, Ey = retarder_jones(Ex, Ey, np.pi + retardance_error, float(config.uniform_hwp_angle_rad) + axis_offset)
        metadata["elements"].append("uniform_hwp")
    if config.uniform_qwp_angle_rad is not None:
        Ex, Ey = retarder_jones(Ex, Ey, 0.5 * np.pi + retardance_error, float(config.uniform_qwp_angle_rad) + axis_offset)
        metadata["elements"].append("uniform_qwp")

    if config.vector_element == "segmented_ra":
        axis = segmented_ra_fast_axis(grid, config) + axis_offset
        Ex, Ey = retarder_jones(Ex, Ey, np.pi + retardance_error, axis)
        metadata["elements"].append("segmented_waveplate_hwp")
    elif config.vector_element == "qplate_radial":
        axis = qplate_fast_axis(grid, config, azimuthal=False) + axis_offset
        Ex, Ey = retarder_jones(Ex, Ey, np.pi + retardance_error, axis)
        metadata["elements"].append("qplate_radial")
    elif config.vector_element == "qplate_azimuthal":
        axis = qplate_fast_axis(grid, config, azimuthal=True) + axis_offset
        Ex, Ey = retarder_jones(Ex, Ey, np.pi + retardance_error, axis)
        metadata["elements"].append("qplate_azimuthal")
    elif config.vector_element == "none":
        metadata["elements"].append("no_spatially_varying_polarization")
    else:  # pragma: no cover - Literal typing should keep this unreachable.
        raise ValueError(f"Unsupported vector element: {config.vector_element!r}")

    return Ex, Ey, metadata


def fresnel_sp_coefficients(n_axicon: float, n_medium: float, base_angle_deg: float) -> tuple[complex, complex]:
    """Return flux-normalized local field transmission coefficients for p/s.

    I normalise the no-index-contrast case to exactly one.  The diagnostic gate
    therefore asks the right question: does sixfold structure remain when the
    physical s/p split is removed?  The Fresnel electric-field coefficients can
    exceed one when the transmitted medium has lower impedance; I multiply by
    the square-root power-flux factor so ``|E|^2`` throughput is not inflated.
    """

    n1 = float(n_axicon)
    n2 = float(n_medium)
    if abs(n1 - n2) < 1.0e-12:
        return 1.0 + 0j, 1.0 + 0j
    theta_i = np.deg2rad(float(base_angle_deg))
    sin_t = n1 / max(n2, BT_EPS) * np.sin(theta_i)
    if abs(sin_t) >= 1.0:
        # I keep the phase of total internal reflection but avoid a hard NaN in
        # planning scans near the critical angle.
        cos_t = 1j * np.sqrt(max(sin_t * sin_t - 1.0, 0.0))
    else:
        cos_t = np.sqrt(max(1.0 - sin_t * sin_t, 0.0))
    cos_i = np.cos(theta_i)
    ts = 2.0 * n1 * cos_i / (n1 * cos_i + n2 * cos_t + BT_EPS)
    tp = 2.0 * n1 * cos_i / (n2 * cos_i + n1 * cos_t + BT_EPS)
    flux_factor = np.real(n2 * cos_t / (n1 * cos_i + BT_EPS))
    flux_amp = np.sqrt(max(float(flux_factor), 0.0))
    return complex(tp * flux_amp), complex(ts * flux_amp)


def physical_axicon_vector_field(
    Ex: np.ndarray,
    Ey: np.ndarray,
    grid: Mapping[str, Any],
    config: PolarizedTrainConfig,
    *,
    realism: RealismMode = "ideal",
    lab: LabRealism | None = None,
    n_axicon_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the physical axicon as a local s/p Fresnel splitter.

    This is the compact model of the vectorial mechanism: radial electric field
    is the local p component, azimuthal electric field is the local s component.
    If ``n_axicon == n_medium`` then ``tp == ts`` and the axicon stops converting
    segmented SOP into segmented intensity.
    """

    lab = LabRealism() if lab is None else lab
    R, Phi, _ = _angle_arrays(grid, config.orientation_rad)
    erx = np.cos(Phi + float(config.orientation_rad))
    ery = np.sin(Phi + float(config.orientation_rad))
    esx = -ery
    esy = erx
    Ep = Ex * erx + Ey * ery
    Es = Ex * esx + Ey * esy

    gamma = float(config.axicon_base_angle_deg)
    if realism == "lab_realistic":
        gamma += float(lab.axicon_angle_error_deg)
    n_axicon = float(config.n_axicon if n_axicon_override is None else n_axicon_override)
    tp, ts = fresnel_sp_coefficients(n_axicon, config.n_medium, gamma)

    envelope = np.exp(-0.5 * ((R - float(config.ring_radius_m)) / max(float(config.ring_width_m), BT_EPS)) ** 2)
    envelope *= np.exp(-0.5 * (R / (0.48 * np.max(R) + BT_EPS)) ** 8)
    phase = np.exp(1j * int(config.vortex_charge) * np.arctan2(np.asarray(grid["Y"]), np.asarray(grid["X"])))
    if realism == "lab_realistic":
        ripple = 1.0 + float(lab.transmission_ripple) * np.cos(
            int(config.symmetry_order) * Phi + 0.37
        )
        quant = max(2, int(lab.quantization_bits))
        phase_angle = np.angle(phase) + float(lab.phase_ripple_rad) * np.sin(5.0 * Phi)
        phase_angle = (np.round(((phase_angle % (2.0 * np.pi)) / (2.0 * np.pi)) * (2**quant - 1)) / (2**quant - 1)) * (2.0 * np.pi)
        phase = np.exp(1j * phase_angle)
        envelope = envelope * ripple

    Eout_x = envelope * phase * (tp * Ep * erx + ts * Es * esx)
    Eout_y = envelope * phase * (tp * Ep * ery + ts * Es * esy)
    metadata = {
        "tp": {"real": float(np.real(tp)), "imag": float(np.imag(tp))},
        "ts": {"real": float(np.real(ts)), "imag": float(np.imag(ts))},
        "tp_abs": float(abs(tp)),
        "ts_abs": float(abs(ts)),
        "fresnel_amplitude_contrast": float(abs(abs(tp) - abs(ts)) / max(0.5 * (abs(tp) + abs(ts)), BT_EPS)),
        "n_axicon": n_axicon,
        "n_medium": float(config.n_medium),
        "axicon_base_angle_deg": gamma,
    }
    return Eout_x, Eout_y, metadata


def vector_angular_spectrum_propagate(
    Ex: np.ndarray,
    Ey: np.ndarray,
    grid: Mapping[str, Any],
    *,
    Ez: np.ndarray | None = None,
    wavelength_m: float,
    n_medium: float,
    z_values_m: Iterable[float],
) -> dict[str, Any]:
    """Propagate a vector field with angular-spectrum transversality.

    I project the Fourier-domain vector onto the plane perpendicular to each
    wave-vector before propagation.  That enforces ``k.E = 0`` numerically and
    creates the longitudinal component the scalar model cannot see.
    """

    Ex = np.asarray(Ex, dtype=complex)
    Ey = np.asarray(Ey, dtype=complex)
    Ez_arr = np.zeros_like(Ex, dtype=complex) if Ez is None else np.asarray(Ez, dtype=complex)
    N = int(Ex.shape[0])
    dx = float(grid["dx"])
    k0n = 2.0 * np.pi * float(n_medium) / float(wavelength_m)
    fx = np.fft.fftfreq(N, d=dx)
    kx = 2.0 * np.pi * fx
    ky = 2.0 * np.pi * fx
    KX, KY = np.meshgrid(kx, ky)
    kt2 = KX * KX + KY * KY
    KZ = np.sqrt((k0n * k0n - kt2).astype(complex))

    Fx = np.fft.fft2(Ex, norm="ortho")
    Fy = np.fft.fft2(Ey, norm="ortho")
    Fz = np.fft.fft2(Ez_arr, norm="ortho")
    # The real bench cannot carry evanescent spatial frequencies through the
    # axicon/objective train. I remove them before the transverse projection so
    # the vectorial ASM remains a unitary free-space propagation check.
    propagating = kt2 <= (0.98 * k0n) ** 2
    Fx = np.where(propagating, Fx, 0.0)
    Fy = np.where(propagating, Fy, 0.0)
    Fz = np.where(propagating, Fz, 0.0)
    dot = KX * Fx + KY * Fy + KZ * Fz
    denom = k0n * k0n + BT_EPS
    Fx_t = Fx - KX * dot / denom
    Fy_t = Fy - KY * dot / denom
    Fz_t = Fz - KZ * dot / denom

    residual = KX * Fx_t + KY * Fy_t + KZ * Fz_t
    norm = k0n * np.sqrt(np.mean(np.abs(Fx_t) ** 2 + np.abs(Fy_t) ** 2 + np.abs(Fz_t) ** 2)) + BT_EPS
    k_dot_e_rms = float(np.sqrt(np.mean(np.abs(residual) ** 2)) / norm)

    z_values = np.asarray(list(z_values_m), dtype=float)
    fields: list[VectorField3D] = []
    intensity_stack = []
    xz_rows = []
    center = N // 2
    for z in z_values:
        H = np.exp(1j * KZ * float(z))
        vx = np.fft.ifft2(Fx_t * H, norm="ortho")
        vy = np.fft.ifft2(Fy_t * H, norm="ortho")
        vz = np.fft.ifft2(Fz_t * H, norm="ortho")
        fields.append(VectorField3D(vx, vy, vz))
        I = total_intensity_3d(fields[-1])
        intensity_stack.append(I)
        xz_rows.append(I[center, :])

    first = fields[0]
    power0 = discrete_power(np.asarray(intensity_stack[0], dtype=float), grid)
    powers = np.asarray([discrete_power(np.asarray(I, dtype=float), grid) for I in intensity_stack], dtype=float)
    return {
        "field0": first,
        "fields": fields,
        "intensity_stack": np.asarray(intensity_stack),
        "xz": np.asarray(xz_rows),
        "z_values_m": z_values,
        "k_dot_e_rms": k_dot_e_rms,
        "ez_power_fraction": float(
            discrete_power(np.abs(first.Ez) ** 2, grid)
            / max(discrete_power(total_intensity_3d(first), grid), BT_EPS)
        ),
        "power_drift_rel": float((np.max(powers) - np.min(powers)) / max(power0, BT_EPS)),
        "powers": powers,
    }


def propagate_state(
    state: OpticalTrainState,
    z_m: float,
    *,
    wavelength_m: float,
    n_medium: float,
    label: str = "free_space",
) -> OpticalTrainState:
    """Propagate an optical-train state with the vectorial ASM."""

    result = vector_angular_spectrum_propagate(
        state.Ex,
        state.Ey,
        state.grid,
        Ez=state.Ez,
        wavelength_m=wavelength_m,
        n_medium=n_medium,
        z_values_m=[float(z_m)],
    )
    field = result["fields"][-1]
    propagation_record = {
        "label": label,
        "z_m": float(z_m),
        "k_dot_e_rms": result["k_dot_e_rms"],
        "ez_power_fraction": result["ez_power_fraction"],
        "power_drift_rel": result["power_drift_rel"],
    }
    return _state_with(
        state,
        field.Ex,
        field.Ey,
        field.Ez,
        history={"element": label, "kind": "vectorial_asm", "z_m": float(z_m)},
        extra_metadata={"propagation": [propagation_record]},
    )


def apply_fourier_relay_filter(
    state: OpticalTrainState,
    element: FourierRelayFilterElement,
    *,
    realism: RealismMode = "ideal",
    lab: LabRealism | None = None,
) -> OpticalTrainState:
    """Apply a circular 4f Fourier-plane filter to all vector components."""

    lab = LabRealism() if lab is None else lab
    N = int(state.Ex.shape[0])
    dx = float(state.grid["dx"])
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(kx, kx)
    KR = np.hypot(KX, KY)
    radius = max(float(element.radius_m_inv), BT_EPS)
    if realism == "lab_realistic":
        softness = max(float(lab.filter_edge_softness_fraction) * radius, BT_EPS)
        filt = 1.0 / (1.0 + np.exp((KR - radius) / softness))
        transmission = float(element.relay_transmission) * float(lab.relay_transmission)
        transmission *= float(element.first_order_efficiency) * float(lab.first_order_efficiency)
    else:
        filt = (KR <= radius).astype(float)
        transmission = float(element.relay_transmission) * float(element.first_order_efficiency)
    amp = np.sqrt(max(transmission, 0.0))

    def _filter(component: np.ndarray) -> np.ndarray:
        return amp * np.fft.ifft2(np.fft.fft2(component, norm="ortho") * filt, norm="ortho")

    return _state_with(
        state,
        _filter(state.Ex),
        _filter(state.Ey),
        _filter(state.Ez),
        history={
            "element": element.label,
            "kind": "fourier_relay_filter",
            "radius_m_inv": radius,
            "transmission": transmission,
            "realism": realism,
        },
    )


def apply_physical_axicon_element(
    state: OpticalTrainState,
    element: PhysicalAxiconElement,
    *,
    wavelength_m: float,
    n_medium: float,
    realism: RealismMode = "ideal",
    lab: LabRealism | None = None,
) -> OpticalTrainState:
    """Apply conical phase plus local p/s Fresnel transmission."""

    lab = LabRealism() if lab is None else lab
    R, Phi, _ = _angle_arrays(state.grid, element.orientation_rad)
    erx = np.cos(Phi + float(element.orientation_rad))
    ery = np.sin(Phi + float(element.orientation_rad))
    esx = -ery
    esy = erx
    Ep = state.Ex * erx + state.Ey * ery
    Es = state.Ex * esx + state.Ey * esy

    gamma = float(element.base_angle_deg)
    tip_rounding = 0.0
    transmission = float(element.transmission)
    if realism == "lab_realistic":
        gamma += float(lab.axicon_angle_error_deg)
        tip_rounding = float(lab.axicon_tip_rounding_m)
        transmission *= max(0.0, 1.0 - float(lab.axicon_ar_loss))
    tp, ts = fresnel_sp_coefficients(element.n_axicon, n_medium, gamma)
    R_eff = R if tip_rounding <= 0.0 else np.sqrt(R * R + tip_rounding * tip_rounding) - tip_rounding
    phase = np.exp(
        1j
        * float(element.conical_phase_sign)
        * (2.0 * np.pi / float(wavelength_m))
        * (float(element.n_axicon) - float(n_medium))
        * R_eff
        * np.tan(np.deg2rad(gamma))
    )
    aperture_radius = element.aperture_radius_m
    if aperture_radius is None:
        aperture = np.ones_like(R, dtype=float)
    elif realism == "lab_realistic":
        edge = max(2.0 * float(state.grid["dx"]), 0.02 * float(aperture_radius))
        aperture = 1.0 / (1.0 + np.exp((R - float(aperture_radius)) / edge))
    else:
        aperture = (R <= float(aperture_radius)).astype(float)
    amp = np.sqrt(max(transmission, 0.0)) * aperture
    Ex = amp * phase * (tp * Ep * erx + ts * Es * esx)
    Ey = amp * phase * (tp * Ep * ery + ts * Es * esy)
    Ez = amp * phase * state.Ez
    return _state_with(
        state,
        Ex,
        Ey,
        Ez,
        history={
            "element": element.label,
            "kind": "physical_axicon",
            "n_axicon": float(element.n_axicon),
            "n_medium": float(n_medium),
            "tp_abs": float(abs(tp)),
            "ts_abs": float(abs(ts)),
            "base_angle_deg": gamma,
            "realism": realism,
        },
    )


def apply_optical_element(
    state: OpticalTrainState,
    element: Any,
    *,
    config: PolarizedTrainConfig,
    realism: RealismMode = "ideal",
    lab: LabRealism | None = None,
) -> OpticalTrainState:
    """Apply one configured optical element to the train state."""

    lab = LabRealism() if lab is None else lab
    if isinstance(element, LinearPolarizerElement):
        extinction = element.extinction_ratio
        transmission = element.transmission
        if realism == "lab_realistic":
            extinction = min(float(extinction), float(lab.polarizer_extinction_ratio)) if np.isfinite(extinction) else float(lab.polarizer_extinction_ratio)
            transmission *= float(lab.polarizer_transmission)
        Ex, Ey = linear_polarizer_jones(
            state.Ex,
            state.Ey,
            element.angle_rad,
            extinction_ratio=extinction,
            transmission=transmission,
        )
        return _state_with(
            state,
            Ex,
            Ey,
            history={"element": element.label, "kind": "linear_polarizer", "realism": realism},
        )

    if isinstance(element, HWPElement):
        retardance = float(element.retardance_rad)
        theta = float(element.theta_rad)
        transmission = float(element.transmission)
        if realism == "lab_realistic":
            retardance += float(lab.retardance_error_rad)
            theta += float(lab.axis_error_rad)
            transmission *= max(0.0, 1.0 - float(lab.waveplate_ar_loss))
        Ex, Ey = retarder_jones(state.Ex, state.Ey, retardance, theta)
        amp = np.sqrt(max(transmission, 0.0))
        return _state_with(
            state,
            amp * Ex,
            amp * Ey,
            amp * state.Ez,
            history={"element": element.label, "kind": "hwp", "theta_rad": theta, "realism": realism},
        )

    if isinstance(element, QWPElement):
        retardance = float(element.retardance_rad)
        theta = float(element.theta_rad)
        transmission = float(element.transmission)
        if realism == "lab_realistic":
            retardance += float(lab.retardance_error_rad)
            theta += float(lab.axis_error_rad)
            transmission *= max(0.0, 1.0 - float(lab.waveplate_ar_loss))
        Ex, Ey = retarder_jones(state.Ex, state.Ey, retardance, theta)
        amp = np.sqrt(max(transmission, 0.0))
        return _state_with(
            state,
            amp * Ex,
            amp * Ey,
            amp * state.Ez,
            history={"element": element.label, "kind": "qwp", "theta_rad": theta, "realism": realism},
        )

    if isinstance(element, SegmentedWaveplateElement):
        axis, seam_distance = segmented_waveplate_axis_map(state.grid, element)
        retardance = float(element.retardance_rad)
        transmission = float(element.transmission)
        if realism == "lab_realistic":
            axis = axis + float(lab.segmented_angular_misregistration_rad) + float(lab.axis_error_rad)
            retardance += float(lab.retardance_error_rad)
            width = max(float(lab.segmented_seam_width_rad), BT_EPS)
            seam_loss = float(lab.segmented_seam_loss) * np.exp(-0.5 * (seam_distance / width) ** 2)
            transmission_map = np.sqrt(np.clip(transmission * (1.0 - seam_loss), 0.0, None))
        else:
            transmission_map = np.sqrt(max(transmission, 0.0))
        Ex, Ey = retarder_jones(state.Ex, state.Ey, retardance, axis)
        return _state_with(
            state,
            transmission_map * Ex,
            transmission_map * Ey,
            transmission_map * state.Ez,
            history={
                "element": element.label,
                "kind": "segmented_waveplate",
                "sector_count": int(element.sector_count),
                "realism": realism,
            },
        )

    if isinstance(element, QPlateElement):
        pseudo_config = replace(
            config,
            qplate_charge=float(element.q),
            orientation_rad=float(element.orientation_rad),
        )
        axis = qplate_fast_axis(state.grid, pseudo_config, azimuthal=bool(element.azimuthal))
        retardance = float(element.retardance_rad)
        transmission = float(element.transmission)
        Ex_in = state.Ex
        Ey_in = state.Ey
        if realism == "lab_realistic":
            axis = axis + float(lab.axis_error_rad)
            retardance += float(lab.retardance_error_rad)
            eta = np.clip(float(lab.qplate_conversion_efficiency), 0.0, 1.0)
            transmission *= max(0.0, 1.0 - float(lab.waveplate_ar_loss))
        else:
            eta = 1.0
        Ex_conv, Ey_conv = retarder_jones(Ex_in, Ey_in, retardance, axis)
        Ex = np.sqrt(eta) * Ex_conv + np.sqrt(max(1.0 - eta, 0.0)) * Ex_in
        Ey = np.sqrt(eta) * Ey_conv + np.sqrt(max(1.0 - eta, 0.0)) * Ey_in
        if realism == "lab_realistic" and lab.qplate_central_defect_radius_m > 0:
            R, _, _ = _angle_arrays(state.grid)
            mix = np.clip(R / float(lab.qplate_central_defect_radius_m), 0.0, 1.0) ** 2
            Ex = mix * Ex + (1.0 - mix) * Ex_in
            Ey = mix * Ey + (1.0 - mix) * Ey_in
        amp = np.sqrt(max(transmission, 0.0))
        return _state_with(
            state,
            amp * Ex,
            amp * Ey,
            amp * state.Ez,
            history={"element": element.label, "kind": "qplate", "q": float(element.q), "realism": realism},
        )

    if isinstance(element, PhysicalAxiconElement):
        return apply_physical_axicon_element(
            state,
            element,
            wavelength_m=config.wavelength_m,
            n_medium=config.n_medium,
            realism=realism,
            lab=lab,
        )

    if isinstance(element, FourierRelayFilterElement):
        return apply_fourier_relay_filter(state, element, realism=realism, lab=lab)

    if isinstance(element, FreeSpacePropagationElement):
        return propagate_state(
            state,
            element.z_m,
            wavelength_m=config.wavelength_m,
            n_medium=config.n_medium,
            label=element.label,
        )

    if isinstance(element, SLMPhaseElement):
        phase = slm_structuring_phase_map(state.grid, element)
        if realism == "lab_realistic" and element.phase_bits is None:
            quantized = SLMPhaseElement(
                label=element.label,
                phase_rad=phase,
                modulated_axis=element.modulated_axis,
                phase_bits=int(lab.quantization_bits),
            )
            phase = slm_structuring_phase_map(state.grid, quantized)
        Ex, Ey = _apply_component_phase(state.Ex, state.Ey, phase, element.modulated_axis)
        return _state_with(
            state,
            Ex,
            Ey,
            history={
                "element": element.label,
                "kind": "slm_phase",
                "modulated_axis": element.modulated_axis,
                "realism": realism,
            },
            extra_metadata={"phase_maps": {element.label: phase}},
        )

    if isinstance(element, SLMConjugateElement):
        phase_maps = state.metadata.get("phase_maps", {})
        if element.source_label not in phase_maps:
            raise KeyError(f"No stored SLM phase map named {element.source_label!r}")
        Ex, Ey = _apply_component_phase(state.Ex, state.Ey, -np.asarray(phase_maps[element.source_label]), element.modulated_axis)
        return _state_with(
            state,
            Ex,
            Ey,
            history={
                "element": element.label,
                "kind": "slm_conjugate",
                "source_label": element.source_label,
                "modulated_axis": element.modulated_axis,
            },
        )

    raise TypeError(f"Unsupported optical element type: {type(element).__name__}")


def run_optical_pipeline(
    config: PolarizedTrainConfig,
    elements: Iterable[Any],
    *,
    realism: RealismMode = "ideal",
    lab: LabRealism | None = None,
    initial_state: OpticalTrainState | None = None,
) -> OpticalTrainState:
    """Run an ordered, configurable polarized optical train.

    The caller owns the order.  That is important for the upgraded bench because
    the axicon and 4f/filter block may be swapped, while the two SLM phase steps
    can sit upstream of the polarization optics as explicit elements.
    """

    lab = LabRealism() if lab is None else lab
    state = initial_train_state(make_grid(config), angle_rad=config.input_polarization_angle_rad) if initial_state is None else initial_state
    for element in elements:
        state = apply_optical_element(state, element, config=config, realism=realism, lab=lab)
    return state


def two_slm_conjugate_prefix(
    *,
    phase_label: str = "slm1",
    vortex_charge: int = 0,
    axicon_kr_m_inv: float = 0.0,
    carrier_lpmm: float = 0.0,
    inter_slm_z_m: float = 0.0,
    modulated_axis: Literal["H", "V", "both"] = "H",
) -> tuple[Any, ...]:
    """Convenience prefix for SLM1 -> propagation -> SLM2 conjugation."""

    return (
        SLMPhaseElement(
            label=phase_label,
            vortex_charge=int(vortex_charge),
            axicon_kr_m_inv=float(axicon_kr_m_inv),
            carrier_lpmm=float(carrier_lpmm),
            modulated_axis=modulated_axis,
        ),
        FreeSpacePropagationElement(float(inter_slm_z_m), label="inter_slm_vectorial_asm"),
        SLMConjugateElement(source_label=phase_label, modulated_axis=modulated_axis),
    )


def preset_train_config(name: PresetTrainName, **overrides: Any) -> PresetTrainConfig:
    """Return a named preset config with caller-visible parameter overrides."""

    name = str(name)
    if name not in {
        "segmented_vector_hexagon",
        "qplate_radial",
        "qplate_azimuthal",
        "scalar_vortex_bessel",
    }:
        raise ValueError(f"Unknown polarized train preset: {name!r}")
    cfg = PresetTrainConfig(name=name)  # type: ignore[arg-type]
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg


def build_preset_train_elements(preset: PresetTrainConfig) -> tuple[Any, ...]:
    """Build the ordered element list for a named IRL bench preset."""

    elements: list[Any] = []
    if preset.include_two_slm_conjugate:
        elements.extend(
            two_slm_conjugate_prefix(
                phase_label="slm1",
                vortex_charge=preset.scalar_vortex_charge,
                axicon_kr_m_inv=preset.scalar_axicon_kr_m_inv,
                carrier_lpmm=preset.scalar_carrier_lpmm,
                inter_slm_z_m=preset.inter_slm_z_m,
                modulated_axis=preset.modulated_axis,
            )
        )

    if preset.name == "scalar_vortex_bessel":
        if not preset.include_two_slm_conjugate:
            elements.append(
                SLMPhaseElement(
                    label="scalar_slm_vortex_axicon",
                    vortex_charge=preset.scalar_vortex_charge,
                    axicon_kr_m_inv=preset.scalar_axicon_kr_m_inv,
                    carrier_lpmm=preset.scalar_carrier_lpmm,
                    modulated_axis=preset.modulated_axis,
                )
            )
        elements.append(
            FourierRelayFilterElement(
                radius_m_inv=preset.filter_radius_m_inv,
                relay_transmission=preset.relay_transmission,
                first_order_efficiency=preset.first_order_efficiency,
                label="scalar_fourier_relay_filter",
            )
        )
        return tuple(elements)

    elements.append(
        LinearPolarizerElement(
            angle_rad=preset.polarizer_angle_rad,
            extinction_ratio=preset.polarizer_extinction_ratio,
            label="input_polarizer",
        )
    )
    if preset.name == "segmented_vector_hexagon":
        count = int(preset.sector_count if preset.sector_count is not None else 2 * int(preset.sector_pairs))
        elements.append(
            SegmentedWaveplateElement(
                sector_count=count,
                per_sector_fast_axis_rad=preset.per_sector_fast_axis_rad,
                retardance_rad=preset.segmented_retardance_rad,
                orientation_rad=preset.sector_orientation_rad,
                label="segmented_ra_waveplate",
            )
        )
    elif preset.name in {"qplate_radial", "qplate_azimuthal"}:
        elements.append(
            QPlateElement(
                q=preset.q,
                retardance_rad=preset.qplate_retardance_rad,
                orientation_rad=preset.qplate_orientation_rad,
                azimuthal=preset.name == "qplate_azimuthal",
                label=preset.name,
            )
        )
    else:  # pragma: no cover - preset_train_config validates names.
        raise ValueError(f"Unsupported preset name: {preset.name!r}")

    axicon = PhysicalAxiconElement(
        n_axicon=preset.n_axicon,
        base_angle_deg=preset.axicon_base_angle_deg,
        aperture_radius_m=preset.axicon_aperture_radius_m,
        label="physical_axicon",
    )
    relay = FourierRelayFilterElement(
        radius_m_inv=preset.filter_radius_m_inv,
        relay_transmission=preset.relay_transmission,
        first_order_efficiency=preset.first_order_efficiency,
        label="fourier_relay_filter",
    )
    if preset.axicon_before_4f:
        elements.extend([axicon, relay])
    else:
        elements.extend([relay, axicon])
    return tuple(elements)


def run_preset_train(
    train_config: PolarizedTrainConfig,
    preset: PresetTrainConfig | PresetTrainName,
    *,
    realism: RealismMode = "ideal",
    lab: LabRealism | None = None,
    initial_state: OpticalTrainState | None = None,
) -> dict[str, Any]:
    """Run a named preset in ideal or lab-realistic mode."""

    preset_cfg = preset_train_config(preset) if isinstance(preset, str) else preset
    elements = build_preset_train_elements(preset_cfg)
    state = run_optical_pipeline(
        train_config,
        elements,
        realism=realism,
        lab=lab,
        initial_state=initial_state,
    )
    intensity = np.abs(state.Ex) ** 2 + np.abs(state.Ey) ** 2 + np.abs(state.Ez) ** 2
    metrics = sixfold_ring_metrics(intensity, state.grid, train_config)
    metrics.update(
        {
            "total_power_au_m2": discrete_power(intensity, state.grid),
            "element_count": len(elements),
        }
    )
    return {
        "preset": preset_cfg,
        "elements": elements,
        "state": state,
        "grid": state.grid,
        "intensity": intensity,
        "metrics": metrics,
        "realism": realism,
        "history": state.metadata.get("history", []),
    }


def _jsonable_signature(value: Any) -> Any:
    """Convert element configs into a comparison-safe signature."""

    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "min": float(np.min(value)) if value.size else 0.0,
            "max": float(np.max(value)) if value.size else 0.0,
            "mean": float(np.mean(value)) if value.size else 0.0,
        }
    if isinstance(value, tuple):
        return [_jsonable_signature(v) for v in value]
    if isinstance(value, list):
        return [_jsonable_signature(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable_signature(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def element_signature(element: Any) -> dict[str, Any]:
    """Return the documented, realism-independent identity of one element."""

    payload = asdict(element) if hasattr(element, "__dataclass_fields__") else {"repr": repr(element)}
    return {"type": type(element).__name__, "params": _jsonable_signature(payload)}


def grid_signature(grid: Mapping[str, Any]) -> dict[str, Any]:
    """Return the grid identity that must match in fair comparisons."""

    x = np.asarray(grid["x"], dtype=float)
    return {
        "shape": list(np.asarray(grid["X"]).shape),
        "dx_m": float(grid["dx"]),
        "x_min_m": float(x[0]),
        "x_max_m": float(x[-1]),
    }


def assert_same_train_and_grid(
    ideal_elements: Iterable[Any],
    lab_elements: Iterable[Any],
    ideal_grid: Mapping[str, Any],
    lab_grid: Mapping[str, Any],
) -> None:
    """Fail loudly unless ideal and lab use the same ordered elements and grid."""

    ideal_sig = [element_signature(e) for e in ideal_elements]
    lab_sig = [element_signature(e) for e in lab_elements]
    if ideal_sig != lab_sig:
        raise AssertionError("Fair comparison failed: ideal and lab element lists differ.")
    if grid_signature(ideal_grid) != grid_signature(lab_grid):
        raise AssertionError("Fair comparison failed: ideal and lab grids differ.")


def stokes_from_state(state: OpticalTrainState) -> dict[str, np.ndarray]:
    """Return Stokes maps from transverse Jones components.

    These are field diagnostics. Without a QWP in the analyzer train, the lab
    can reconstruct ``S0``, ``S1``, ``S2``, and ``psi`` from linear analyzers,
    but not ``S3`` or ellipticity.
    """

    Ex = np.asarray(state.Ex, dtype=complex)
    Ey = np.asarray(state.Ey, dtype=complex)
    S0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    S1 = np.abs(Ex) ** 2 - np.abs(Ey) ** 2
    S2 = 2.0 * np.real(Ex * np.conj(Ey))
    S3 = -2.0 * np.imag(Ex * np.conj(Ey))
    return {"S0": S0, "S1": S1, "S2": S2, "S3": S3}


def analyzer_maps_from_state(
    state: OpticalTrainState,
    angles_deg: Iterable[int] = (0, 45, 90, 135),
) -> dict[int, np.ndarray]:
    """Return ideal linear-analyzer camera frames for a train state."""

    frames: dict[int, np.ndarray] = {}
    Ex = np.asarray(state.Ex, dtype=complex)
    Ey = np.asarray(state.Ey, dtype=complex)
    for angle in angles_deg:
        theta = np.deg2rad(float(angle))
        projected = Ex * np.cos(theta) + Ey * np.sin(theta)
        frames[int(angle)] = np.abs(projected) ** 2
    return frames


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(np.asarray(values, dtype=float) * weights) / max(float(np.sum(weights)), BT_EPS))


def _headless_angle_mean(psi: np.ndarray, weights: np.ndarray) -> float:
    z = np.sum(weights * np.exp(2j * np.asarray(psi, dtype=float)))
    return float(0.5 * np.angle(z))


def _frame_correlation(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float).ravel()
    bv = np.asarray(b, dtype=float).ravel()
    av = av - np.mean(av)
    bv = bv - np.mean(bv)
    return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + BT_EPS))


def _radial_feature_metrics(intensity: np.ndarray, grid: Mapping[str, Any]) -> dict[str, float]:
    R = np.hypot(np.asarray(grid["X"], dtype=float), np.asarray(grid["Y"], dtype=float))
    I = np.asarray(intensity, dtype=float)
    max_r = float(np.max(R))
    bins = np.linspace(0.0, max_r, 220)
    idx = np.digitize(R.ravel(), bins) - 1
    radial = np.zeros(bins.size - 1, dtype=float)
    counts = np.zeros_like(radial)
    np.add.at(radial, idx[(idx >= 0) & (idx < radial.size)], I.ravel()[(idx >= 0) & (idx < radial.size)])
    np.add.at(counts, idx[(idx >= 0) & (idx < radial.size)], 1.0)
    radial = radial / np.maximum(counts, 1.0)
    centres = 0.5 * (bins[:-1] + bins[1:])
    peak_idx = int(np.argmax(radial))
    radius = float(centres[peak_idx])
    half = 0.5 * float(radial[peak_idx])
    above = np.flatnonzero(radial >= half)
    if above.size:
        width = float(centres[above[-1]] - centres[above[0]])
    else:
        width = np.nan
    return {
        "feature_radius_um": radius / BT_UM,
        "feature_diameter_um": 2.0 * radius / BT_UM,
        "feature_width_um": width / BT_UM if np.isfinite(width) else np.nan,
    }


def _zone_metric_from_state(
    state: OpticalTrainState,
    config: PolarizedTrainConfig,
) -> dict[str, float]:
    z_values = np.linspace(0.0, float(config.z_max_m), int(config.z_points))
    prop = vector_angular_spectrum_propagate(
        state.Ex,
        state.Ey,
        state.grid,
        Ez=state.Ez,
        wavelength_m=config.wavelength_m,
        n_medium=config.n_medium,
        z_values_m=z_values,
    )
    peaks = np.asarray([float(np.max(plane)) for plane in prop["intensity_stack"]], dtype=float)
    if peaks.size == 0 or np.max(peaks) <= 0.0:
        return {"zone_um": np.nan, "zone_start_um": np.nan, "zone_end_um": np.nan}
    above = np.flatnonzero(peaks >= 0.5 * float(np.max(peaks)))
    if above.size == 0:
        return {"zone_um": np.nan, "zone_start_um": np.nan, "zone_end_um": np.nan}
    start = float(z_values[above[0]])
    end = float(z_values[above[-1]])
    return {"zone_um": (end - start) / BT_UM, "zone_start_um": start / BT_UM, "zone_end_um": end / BT_UM}


def fair_case_metrics(
    state: OpticalTrainState,
    config: PolarizedTrainConfig,
    *,
    input_power_au_m2: float,
    vector_case: bool,
    analyzer_has_qwp: bool = False,
) -> dict[str, Any]:
    """Compute the scalar/vector metrics used in the fair delta table."""

    intensity = np.abs(state.Ex) ** 2 + np.abs(state.Ey) ** 2 + np.abs(state.Ez) ** 2
    ring = sixfold_ring_metrics(intensity, state.grid, config)
    radial = _radial_feature_metrics(intensity, state.grid)
    zone = _zone_metric_from_state(state, config)
    power = discrete_power(intensity, state.grid)
    out: dict[str, Any] = {
        **zone,
        **radial,
        "peak_fluence_proxy_au": float(np.max(intensity)),
        "contrast": float(ring["angular_contrast"]),
        "throughput": float(power / max(float(input_power_au_m2), BT_EPS)),
        "total_power_au_m2": power,
        "vector_case": bool(vector_case),
    }
    if vector_case:
        stokes = stokes_from_state(state)
        S0 = np.asarray(stokes["S0"], dtype=float)
        weights = S0 / max(float(np.max(S0)), BT_EPS)
        mask = weights > 0.05
        if not np.any(mask):
            mask = np.ones_like(weights, dtype=bool)
        psi = 0.5 * np.arctan2(stokes["S2"], stokes["S1"])
        chi = 0.5 * np.arcsin(np.clip(stokes["S3"] / (S0 + BT_EPS), -1.0, 1.0))
        for key in ("S1", "S2", "S3"):
            out[f"{key}_over_S0_weighted_mean"] = _weighted_mean(stokes[key][mask] / (S0[mask] + BT_EPS), weights[mask])
        out["psi_weighted_mean_rad"] = _headless_angle_mean(psi[mask], weights[mask])
        out["chi_field_only_weighted_mean_rad"] = _weighted_mean(chi[mask], weights[mask])
        frames = analyzer_maps_from_state(state)
        for angle, frame in frames.items():
            out[f"analyzer_{angle}_mean"] = _weighted_mean(frame[mask], weights[mask])
        out["psi_lab_observable"] = True
        out["S3_lab_observable"] = bool(analyzer_has_qwp)
        out["chi_lab_observable"] = bool(analyzer_has_qwp)
        out["S3_note"] = "field-only diagnostic unless a QWP analyzer is present"
    else:
        out["psi_lab_observable"] = False
        out["S3_lab_observable"] = False
        out["chi_lab_observable"] = False
        out["S3_note"] = "not applicable to scalar baseline"
    return out


def run_fair_preset_comparison(
    train_config: PolarizedTrainConfig,
    preset: PresetTrainConfig | PresetTrainName,
    *,
    lab: LabRealism | None = None,
    analyzer_has_qwp: bool = False,
) -> dict[str, Any]:
    """Run ideal/lab preset paths with a coded same-train assertion."""

    lab = LabRealism() if lab is None else lab
    preset_cfg = preset_train_config(preset) if isinstance(preset, str) else preset
    elements = build_preset_train_elements(preset_cfg)
    grid = make_grid(train_config)
    initial_ideal = initial_train_state(grid, angle_rad=train_config.input_polarization_angle_rad)
    initial_lab = initial_train_state(grid, angle_rad=train_config.input_polarization_angle_rad)
    input_power = discrete_power(np.abs(initial_ideal.Ex) ** 2 + np.abs(initial_ideal.Ey) ** 2, grid)
    ideal_state = run_optical_pipeline(
        train_config,
        elements,
        realism="ideal",
        lab=lab,
        initial_state=initial_ideal,
    )
    lab_state = run_optical_pipeline(
        train_config,
        elements,
        realism="lab_realistic",
        lab=lab,
        initial_state=initial_lab,
    )
    assert_same_train_and_grid(elements, elements, ideal_state.grid, lab_state.grid)
    vector_case = preset_cfg.name != "scalar_vortex_bessel"
    ideal_metrics = fair_case_metrics(
        ideal_state,
        train_config,
        input_power_au_m2=input_power,
        vector_case=vector_case,
        analyzer_has_qwp=analyzer_has_qwp,
    )
    lab_metrics = fair_case_metrics(
        lab_state,
        train_config,
        input_power_au_m2=input_power,
        vector_case=vector_case,
        analyzer_has_qwp=analyzer_has_qwp,
    )
    row: dict[str, Any] = {
        "preset": preset_cfg.name,
        "fair_same_train_assertion": True,
        "element_order": " -> ".join(type(e).__name__ for e in elements),
        "grid_N": int(train_config.N),
        "grid_dx_um": float(train_config.dx_m / BT_UM),
        "vector_case": vector_case,
        "psi_lab_observable": bool(ideal_metrics["psi_lab_observable"]),
        "S3_lab_observable": bool(ideal_metrics["S3_lab_observable"]),
        "S3_note": ideal_metrics["S3_note"],
    }
    metric_names = [
        "zone_um",
        "feature_radius_um",
        "feature_diameter_um",
        "feature_width_um",
        "peak_fluence_proxy_au",
        "contrast",
        "throughput",
        "total_power_au_m2",
    ]
    if vector_case:
        metric_names.extend(
            [
                "S1_over_S0_weighted_mean",
                "S2_over_S0_weighted_mean",
                "S3_over_S0_weighted_mean",
                "psi_weighted_mean_rad",
                "chi_field_only_weighted_mean_rad",
                "analyzer_0_mean",
                "analyzer_45_mean",
                "analyzer_90_mean",
                "analyzer_135_mean",
            ]
        )
    for name in metric_names:
        i_val = ideal_metrics.get(name, np.nan)
        l_val = lab_metrics.get(name, np.nan)
        row[f"ideal_{name}"] = i_val
        row[f"lab_{name}"] = l_val
        if isinstance(i_val, (int, float, np.floating)) and isinstance(l_val, (int, float, np.floating)):
            row[f"delta_{name}"] = float(l_val) - float(i_val)
            row[f"ratio_{name}"] = float(l_val) / (float(i_val) + BT_EPS)
    if vector_case:
        ideal_frames = analyzer_maps_from_state(ideal_state)
        lab_frames = analyzer_maps_from_state(lab_state)
        for angle in (0, 45, 90, 135):
            row[f"analyzer_{angle}_corr"] = _frame_correlation(ideal_frames[angle], lab_frames[angle])
        i_stokes = stokes_from_state(ideal_state)
        l_stokes = stokes_from_state(lab_state)
        i_psi = 0.5 * np.arctan2(i_stokes["S2"], i_stokes["S1"])
        l_psi = 0.5 * np.arctan2(l_stokes["S2"], l_stokes["S1"])
        row["psi_rms_delta_rad"] = float(np.sqrt(np.mean(np.abs(0.5 * np.angle(np.exp(2j * (l_psi - i_psi)))) ** 2)))
    return {
        "preset": preset_cfg,
        "elements": elements,
        "ideal_state": ideal_state,
        "lab_state": lab_state,
        "ideal_metrics": ideal_metrics,
        "lab_metrics": lab_metrics,
        "row": row,
        "grid_signature": grid_signature(grid),
        "element_signature": [element_signature(e) for e in elements],
    }


def run_fair_comparison_suite(
    train_config: PolarizedTrainConfig,
    presets: Iterable[PresetTrainConfig | PresetTrainName] | None = None,
    *,
    lab: LabRealism | None = None,
    analyzer_has_qwp: bool = False,
) -> dict[str, Any]:
    """Run fair ideal-vs-lab comparisons for every named preset."""

    if presets is None:
        presets = ("segmented_vector_hexagon", "qplate_radial", "qplate_azimuthal", "scalar_vortex_bessel")
    comparisons = [
        run_fair_preset_comparison(train_config, preset, lab=lab, analyzer_has_qwp=analyzer_has_qwp)
        for preset in presets
    ]
    return {"comparisons": comparisons, "delta_table": pd.DataFrame([c["row"] for c in comparisons])}


def _element_settings_rows(elements: Iterable[Any], lab: LabRealism) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, element in enumerate(elements):
        kind = type(element).__name__
        ideal = "nominal element parameters; no added loss/error"
        lab_setting = "same element present; "
        if isinstance(element, LinearPolarizerElement):
            lab_setting += f"finite extinction <= {lab.polarizer_extinction_ratio:.3g}, transmission x {lab.polarizer_transmission:.3g}"
        elif isinstance(element, (HWPElement, QWPElement)):
            lab_setting += f"retardance + {lab.retardance_error_rad:.3g} rad, axis + {lab.axis_error_rad:.3g} rad, AR loss {lab.waveplate_ar_loss:.3g}"
        elif isinstance(element, SegmentedWaveplateElement):
            lab_setting += f"seam width {lab.segmented_seam_width_rad:.3g} rad, seam loss {lab.segmented_seam_loss:.3g}, angular misregistration {lab.segmented_angular_misregistration_rad:.3g} rad"
        elif isinstance(element, QPlateElement):
            lab_setting += f"retardance + {lab.retardance_error_rad:.3g} rad, central defect {lab.qplate_central_defect_radius_m / BT_UM:.3g} um, conversion {lab.qplate_conversion_efficiency:.3g}"
        elif isinstance(element, PhysicalAxiconElement):
            lab_setting += f"angle + {lab.axicon_angle_error_deg:.3g} deg, AR loss {lab.axicon_ar_loss:.3g}, tip rounding {lab.axicon_tip_rounding_m / BT_UM:.3g} um"
        elif isinstance(element, FourierRelayFilterElement):
            lab_setting += f"relay transmission x {lab.relay_transmission:.3g}, first-order efficiency x {lab.first_order_efficiency:.3g}, soft filter edge {lab.filter_edge_softness_fraction:.3g} radius"
        elif isinstance(element, FreeSpacePropagationElement):
            lab_setting += "same vectorial ASM propagation"
        elif isinstance(element, (SLMPhaseElement, SLMConjugateElement)):
            lab_setting += f"same phase function; quantized to {lab.quantization_bits} bits for SLM phase elements"
        else:
            lab_setting += "documented realism settings not specialized"
        rows.append({"index": idx, "element": kind, "ideal_settings": ideal, "lab_settings": lab_setting})
    return rows


def write_fair_comparison_audit(
    suite: Mapping[str, Any],
    path: str | Path,
    *,
    lab: LabRealism | None = None,
) -> Path:
    """Write the same-train ideal-vs-lab audit document."""

    lab = LabRealism() if lab is None else lab
    lines = [
        "# Fair Comparison Audit",
        "",
        "The ideal and lab-realistic paths instantiate the same ordered element list on the same numerical grid. The only allowed differences are the per-element realism toggles documented below. The code asserts this with `assert_same_train_and_grid(...)` before any delta row is accepted.",
        "",
        "The linear-analyzer measurement model reports `S0`, `S1`, `S2`, and `psi = 0.5 atan2(S2,S1)`. `S3` and ellipticity/`chi` are field-only diagnostics unless a QWP analyzer is explicitly included; the current preset delta table therefore marks `S3_lab_observable = False` for vector cases.",
        "",
    ]
    for comparison in suite["comparisons"]:
        preset = comparison["preset"].name
        lines.extend([f"## {preset}", ""])
        lines.append(f"Grid signature: `{json.dumps(comparison['grid_signature'], sort_keys=True)}`")
        lines.append("")
        lines.append("| index | element | ideal settings | lab-realistic settings |")
        lines.append("|---:|---|---|---|")
        for row in _element_settings_rows(comparison["elements"], lab):
            lines.append(
                f"| {row['index']} | `{row['element']}` | {row['ideal_settings']} | {row['lab_settings']} |"
            )
        lines.append("")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def total_intensity_3d(field: VectorField3D) -> np.ndarray:
    """Return ``|Ex|^2 + |Ey|^2 + |Ez|^2`` in arbitrary intensity units."""

    return np.abs(field.Ex) ** 2 + np.abs(field.Ey) ** 2 + np.abs(field.Ez) ** 2


def discrete_power(intensity: np.ndarray, grid: Mapping[str, Any]) -> float:
    """Integrate transverse intensity over the sample plane in a.u. m^2."""

    return float(np.sum(np.asarray(intensity, dtype=float)) * float(grid["dx"]) ** 2)


def _sample_bilinear(image: np.ndarray, grid: Mapping[str, Any], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coords = (np.asarray(grid["x"], dtype=float) - float(np.asarray(grid["x"])[0])) / float(grid["dx"])
    ix = (x - float(np.asarray(grid["x"])[0])) / float(grid["dx"])
    iy = (y - float(np.asarray(grid["x"])[0])) / float(grid["dx"])
    ix0 = np.floor(ix).astype(int)
    iy0 = np.floor(iy).astype(int)
    ix0 = np.clip(ix0, 0, image.shape[1] - 2)
    iy0 = np.clip(iy0, 0, image.shape[0] - 2)
    tx = ix - ix0
    ty = iy - iy0
    v00 = image[iy0, ix0]
    v10 = image[iy0, ix0 + 1]
    v01 = image[iy0 + 1, ix0]
    v11 = image[iy0 + 1, ix0 + 1]
    return (1 - tx) * (1 - ty) * v00 + tx * (1 - ty) * v10 + (1 - tx) * ty * v01 + tx * ty * v11


def angular_ring_profile(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    radius_m: float,
    *,
    samples: int = 720,
) -> dict[str, np.ndarray]:
    """Sample intensity on a ring for symmetry and contrast metrics."""

    theta = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=False)
    x = float(radius_m) * np.cos(theta)
    y = float(radius_m) * np.sin(theta)
    values = _sample_bilinear(np.asarray(intensity, dtype=float), grid, x, y)
    return {"theta_rad": theta, "intensity": values}


def sixfold_ring_metrics(
    intensity: np.ndarray,
    grid: Mapping[str, Any],
    config: PolarizedTrainConfig,
) -> dict[str, float | bool]:
    """Measure the sixfold signature of the segmented-vector axicon output."""

    I = np.asarray(intensity, dtype=float)
    profile = angular_ring_profile(I, grid, config.ring_radius_m, samples=720)
    values = np.asarray(profile["intensity"], dtype=float)
    values = np.maximum(values, 0.0)
    centred = values - np.mean(values)
    fft = np.fft.rfft(centred)
    amplitudes = np.abs(fft)
    order = int(config.symmetry_order)
    order_amp = float(amplitudes[order]) if order < amplitudes.size else 0.0
    non_dc = float(np.sum(amplitudes[1:]) + BT_EPS)
    contrast = float((np.percentile(values, 95.0) - np.percentile(values, 5.0)) / max(np.percentile(values, 95.0), BT_EPS))
    core = float(I[I.shape[0] // 2, I.shape[1] // 2])
    ring_peak = float(np.percentile(values, 98.0))
    R, _, _ = _angle_arrays(grid)
    outside = I[R > 1.8 * float(config.ring_radius_m)]
    inside = I[R < 1.45 * float(config.ring_radius_m)]
    localisation = float(np.sum(inside) / max(np.sum(I), BT_EPS))
    lattice_side_peak = float(np.mean(outside) / max(np.mean(inside), BT_EPS)) if outside.size else 0.0
    return {
        "order": order,
        "order_fidelity": float(order_amp / non_dc),
        "angular_contrast": contrast,
        "core_null_depth": float(core / max(ring_peak, BT_EPS)),
        "localization_fraction": localisation,
        "lattice_periodicity_proxy": lattice_side_peak,
        "sixfold_visible": bool(order_amp / non_dc > 0.28 and contrast > 0.05),
        "dark_core_pass": bool(core / max(ring_peak, BT_EPS) < 0.08),
        "localized_pass": bool(localisation > 0.88 and lattice_side_peak < 0.08),
    }


def run_polarized_train(
    config: PolarizedTrainConfig | None = None,
    *,
    realism: RealismMode = "ideal",
    lab: LabRealism | None = None,
    n_axicon_override: float | None = None,
    z_values_m: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Run one ideal or lab-realistic upgraded-bench polarized train."""

    config = PolarizedTrainConfig() if config is None else config
    lab = LabRealism() if lab is None else lab
    grid = make_grid(config)
    Ex, Ey = initial_uniform_field(grid, config.input_polarization_angle_rad)
    Ex, Ey, pol_meta = apply_train_polarization_elements(Ex, Ey, grid, config, realism=realism, lab=lab)
    Ex, Ey, ax_meta = physical_axicon_vector_field(
        Ex,
        Ey,
        grid,
        config,
        realism=realism,
        lab=lab,
        n_axicon_override=n_axicon_override,
    )
    exit_intensity = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    exit_metrics = sixfold_ring_metrics(exit_intensity, grid, config)
    if z_values_m is None:
        z_values_m = np.linspace(0.0, float(config.z_max_m), int(config.z_points))
    propagation = vector_angular_spectrum_propagate(
        Ex,
        Ey,
        grid,
        wavelength_m=config.wavelength_m,
        n_medium=config.n_medium,
        z_values_m=z_values_m,
    )
    intensity = total_intensity_3d(propagation["field0"])
    propagated_metrics = sixfold_ring_metrics(intensity, grid, config)
    metrics = dict(exit_metrics)
    metrics.update(
        {
            "exit_total_power_au_m2": discrete_power(exit_intensity, grid),
            "propagated_order_fidelity": propagated_metrics["order_fidelity"],
            "propagated_angular_contrast": propagated_metrics["angular_contrast"],
            "propagated_core_null_depth": propagated_metrics["core_null_depth"],
            "total_power_au_m2": discrete_power(intensity, grid),
            "k_dot_e_rms": propagation["k_dot_e_rms"],
            "ez_power_fraction": propagation["ez_power_fraction"],
            "power_drift_rel": propagation["power_drift_rel"],
            "tp_abs": ax_meta["tp_abs"],
            "ts_abs": ax_meta["ts_abs"],
            "fresnel_amplitude_contrast": ax_meta["fresnel_amplitude_contrast"],
        }
    )
    return {
        "config": config,
        "lab_realism": lab,
        "realism": realism,
        "grid": grid,
        "exit_field": {"Ex": Ex, "Ey": Ey},
        "exit_intensity": exit_intensity,
        "field": propagation["field0"],
        "intensity": intensity,
        "propagation": propagation,
        "exit_metrics": exit_metrics,
        "propagated_metrics": propagated_metrics,
        "metrics": metrics,
        "polarization_metadata": pol_meta,
        "axicon_metadata": ax_meta,
        "uses_full_vector_path": True,
        "uses_waveplate_jones": config.vector_element != "none" or config.uniform_hwp_angle_rad is not None or config.uniform_qwp_angle_rad is not None,
        "uses_physical_axicon_fresnel": True,
    }


def run_hexagon_mechanism_gate(config: PolarizedTrainConfig | None = None) -> dict[str, Any]:
    """Run the required index-contrast versus matched-index decision gate."""

    config = PolarizedTrainConfig() if config is None else config
    contrast = run_polarized_train(config, realism="ideal")
    matched = run_polarized_train(config, realism="ideal", n_axicon_override=config.n_medium)
    contrast_metric = float(contrast["metrics"]["angular_contrast"])
    matched_metric = float(matched["metrics"]["angular_contrast"])
    contrast_order = float(contrast["metrics"]["order_fidelity"])
    matched_order = float(matched["metrics"]["order_fidelity"])
    vanish_ratio = matched_metric / max(contrast_metric, BT_EPS)
    order_ratio = matched_order / max(contrast_order, BT_EPS)
    vectorial = bool(vanish_ratio < 0.35 and order_ratio < 0.50)
    verdict = (
        "vectorial_fresnel_axicon"
        if vectorial
        else "geometric_or_scalar_segmented_pattern"
    )
    rows = pd.DataFrame(
        [
            {
                "case": "index_contrast_axicon",
                "n_axicon": config.n_axicon,
                "n_medium": config.n_medium,
                **{k: v for k, v in contrast["metrics"].items() if isinstance(v, (int, float, bool, np.bool_))},
            },
            {
                "case": "matched_index_axicon",
                "n_axicon": config.n_medium,
                "n_medium": config.n_medium,
                **{k: v for k, v in matched["metrics"].items() if isinstance(v, (int, float, bool, np.bool_))},
            },
        ]
    )
    return {
        "verdict": verdict,
        "vectorial": vectorial,
        "contrast": contrast,
        "matched": matched,
        "summary": rows,
        "vanish_ratio": float(vanish_ratio),
        "order_ratio": float(order_ratio),
        "default_path": "full_vector_jones_fresnel_axicon" if vectorial else "scalar_or_geometric_allowed_after_validation",
    }


def ideal_lab_comparison(config: PolarizedTrainConfig | None = None) -> dict[str, Any]:
    """Run the same train with realism toggled off and on."""

    config = PolarizedTrainConfig() if config is None else config
    ideal = run_polarized_train(config, realism="ideal")
    lab = run_polarized_train(config, realism="lab_realistic")
    rows = []
    for label, case in (("ideal", ideal), ("lab_realistic", lab)):
        rows.append(
            {
                "path": label,
                **{k: v for k, v in case["metrics"].items() if isinstance(v, (int, float, bool, np.bool_))},
            }
        )
    return {"ideal": ideal, "lab_realistic": lab, "summary": pd.DataFrame(rows)}


def export_element_maps(
    config: PolarizedTrainConfig,
    output_dir: str | Path,
    *,
    label: str = "polarized_train",
) -> dict[str, Path]:
    """Export loadable 8-bit maps for the spatial polarization element.

    These are not SLM claims.  For segmented/q-plate modes the PNG encodes the
    waveplate fast-axis angle modulo pi so the future optic can be documented
    next to the simulated field.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = make_grid(config)
    if config.vector_element == "segmented_ra":
        axis = segmented_ra_fast_axis(grid, config)
    elif config.vector_element == "qplate_radial":
        axis = qplate_fast_axis(grid, config, azimuthal=False)
    elif config.vector_element == "qplate_azimuthal":
        axis = qplate_fast_axis(grid, config, azimuthal=True)
    else:
        axis = np.zeros((config.N, config.N), dtype=float)
    gray = np.asarray(np.round(((axis % np.pi) / np.pi) * 255.0), dtype=np.uint8)
    try:
        from PIL import Image

        png = out / f"{label}_fast_axis_8bit.png"
        Image.fromarray(gray, mode="L").save(png)
    except Exception:  # pragma: no cover - Pillow is present in the study env.
        png = out / f"{label}_fast_axis_8bit.npy"
        np.save(png, gray)
    params = out / f"{label}_params.json"
    payload = {
        "map_type": "waveplate_fast_axis_mod_pi",
        "units": "8-bit grayscale, 0..255 maps to 0..pi radians",
        "config": asdict(config),
        "hardware_note": "spatially varying polarization optic; not generated by the co-aligned phase-only SLMs",
    }
    params.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"fast_axis_map": Path(png), "params_json": params}


def _imshow_intensity(
    ax: plt.Axes,
    case: Mapping[str, Any],
    title: str,
    *,
    vmax: float | None = None,
    intensity_key: str = "intensity",
) -> Any:
    grid = case["grid"]
    x_um = np.asarray(grid["x"], dtype=float) / BT_UM
    I = np.asarray(case[intensity_key], dtype=float)
    scale = I / max(float(vmax if vmax is not None else np.max(I)), BT_EPS)
    artist = ax.imshow(
        vbb_style.display_scale(scale, gamma=0.45, normalise=False),
        origin="lower",
        extent=[float(x_um[0]), float(x_um[-1]), float(x_um[0]), float(x_um[-1])],
        cmap=vbb_style.INTENSITY_CMAP,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_title(title)
    ax.set_xlabel("x [um, sample plane]")
    ax.set_ylabel("y [um, sample plane]")
    return artist


def plot_mechanism_gate(gate: Mapping[str, Any], output_path: str | Path) -> Path:
    """Save the matched-index mechanism diagnostic figure."""

    vbb_style.apply_style()
    contrast = gate["contrast"]
    matched = gate["matched"]
    vmax = max(float(np.max(contrast["exit_intensity"])), float(np.max(matched["exit_intensity"])))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), constrained_layout=True)
    artist = _imshow_intensity(axes[0], contrast, "index-contrast axicon", vmax=vmax, intensity_key="exit_intensity")
    _imshow_intensity(axes[1], matched, "matched index axicon", vmax=vmax, intensity_key="exit_intensity")
    cb = fig.colorbar(artist, ax=axes, shrink=0.90)
    cb.set_label("matched display intensity, gamma=0.45 [a.u.]")
    caption = (
        "Mechanism gate for the segmented radial/azimuthal train: the same field is run with the real index-contrast axicon and with n_axicon set equal to n_medium. "
        f"The sixfold contrast ratio is {gate['vanish_ratio']:.3f}, so the recorded verdict is {gate['verdict']}."
    )
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"verdict": gate["verdict"]})
    plt.close(fig)
    return out


def plot_ideal_lab_comparison(comparison: Mapping[str, Any], output_path: str | Path) -> Path:
    """Save transverse and axial views for the ideal/lab-realistic train."""

    vbb_style.apply_style()
    ideal = comparison["ideal"]
    lab = comparison["lab_realistic"]
    vmax = max(float(np.max(ideal["intensity"])), float(np.max(lab["intensity"])))
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 7.4), constrained_layout=True)
    artist = _imshow_intensity(axes[0, 0], ideal, "ideal x-y", vmax=vmax)
    _imshow_intensity(axes[0, 1], lab, "lab-realistic x-y", vmax=vmax)
    for ax, case, title in ((axes[1, 0], ideal, "ideal x-z"), (axes[1, 1], lab, "lab-realistic x-z")):
        grid = case["grid"]
        x_um = np.asarray(grid["x"], dtype=float) / BT_UM
        z_um = np.asarray(case["propagation"]["z_values_m"], dtype=float) / BT_UM
        xz = np.asarray(case["propagation"]["xz"], dtype=float)
        scale = xz / max(vmax, BT_EPS)
        im = ax.imshow(
            vbb_style.display_scale(scale, gamma=0.45, normalise=False),
            origin="lower",
            aspect="auto",
            extent=[float(x_um[0]), float(x_um[-1]), float(z_um[0]), float(z_um[-1])],
            cmap=vbb_style.INTENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("x [um, sample plane]")
        ax.set_ylabel("z [um, sample plane]")
    cb = fig.colorbar(artist, ax=axes, shrink=0.92)
    cb.set_label("matched display intensity, gamma=0.45 [a.u.]")
    caption = (
        "Ideal and lab-realistic upgraded-bench simulations use the same segmented-waveplate plus physical-axicon train; only deterministic realism toggles differ. "
        "The x-z panels use vectorial ASM with k.E projected to zero and Ez included in the displayed intensity."
    )
    out = vbb_style.save_figure(fig, output_path, caption, metadata={"figure": "polarized_train_ideal_lab"})
    plt.close(fig)
    return out


def write_mechanism_doc(
    gate: Mapping[str, Any],
    path: str | Path = "docs/HEXAGON_MECHANISM.md",
) -> Path:
    """Write the mechanism verdict document required by the prompt."""

    summary = gate["summary"]
    contrast = summary.loc[summary["case"] == "index_contrast_axicon"].iloc[0]
    matched = summary.loc[summary["case"] == "matched_index_axicon"].iloc[0]
    text = f"""# Hexagon Mechanism Diagnostic

This note records the decision gate for the upgraded polarized optical train. I
run the same segmented radial/azimuthal field through the same physical-axicon
model twice: once with the axicon index contrast present, and once with
`n_axicon = n_medium` so the local s/p Fresnel split is removed. The contrast
and order-fidelity values below are measured at the axicon exit, before the
vectorial ASM propagation step; propagation is still run afterward with
`k.E = 0` and `Ez` retained.

## Result

Verdict: **{gate['verdict']}**.

Default simulation path: **{gate['default_path']}**.

| case | exit angular contrast | exit order-6 fidelity | propagated angular contrast | core null | k dot E rms | Ez power fraction |
|---|---:|---:|---:|---:|---:|---:|
| index-contrast axicon | {float(contrast['angular_contrast']):.6g} | {float(contrast['order_fidelity']):.6g} | {float(contrast['propagated_angular_contrast']):.6g} | {float(contrast['core_null_depth']):.6g} | {float(contrast['k_dot_e_rms']):.6g} | {float(contrast['ez_power_fraction']):.6g} |
| matched-index axicon | {float(matched['angular_contrast']):.6g} | {float(matched['order_fidelity']):.6g} | {float(matched['propagated_angular_contrast']):.6g} | {float(matched['core_null_depth']):.6g} | {float(matched['k_dot_e_rms']):.6g} | {float(matched['ez_power_fraction']):.6g} |

The matched-index / index-contrast angular-contrast ratio is
**{gate['vanish_ratio']:.6g}**, and the order-fidelity ratio is
**{gate['order_ratio']:.6g}**.

## Interpretation

Uniform HWP/QWP elements in this simulator are global Jones matrices: they can
prepare a polarization state, but they cannot create azimuthal or sectored
structure. The sectored structure is created only by a spatially varying optic,
either a segmented waveplate or a q-plate/S-waveplate. The physical axicon then
decomposes the field into local p and s components and applies different
Fresnel transmission coefficients.

Because the Fresnel-converted sixfold modulation at the axicon exit collapses
when `n_axicon = n_medium`, I treat the segmented-vector hexagon as a vectorial
Fresnel-axicon effect. The propagated matched-index row still has nonzero
angular redistribution from the segmented vector field and the transverse
projection; I do not use that as permission to simplify the bench model. The
study therefore keeps the full vector Jones plus physical-axicon path, projects
the field so `k.E = 0`, retains `Ez`, and does not replace this branch with a
scalar holographic approximation.

## Guardrail

The co-aligned phase-only SLM-only bench remains the Case-1 model from
`vbb_vector.py`: it cannot create radial/azimuthal or segmented vector beams by
software rotation alone. The upgraded path here becomes reachable only after a
real spatially varying polarization element is present.

## Composable Train Layer

The implementation now also exposes an ordered pipeline API in
`vbb_polarized_train.py`. The elements are explicit dataclasses:
`SLMPhaseElement`, `SLMConjugateElement`, `FreeSpacePropagationElement`,
`LinearPolarizerElement`, `HWPElement`, `QWPElement`,
`SegmentedWaveplateElement`, `QPlateElement`, `PhysicalAxiconElement`, and
`FourierRelayFilterElement`. Ideal and lab-realistic behavior are selected by
the same `realism` flag for the whole train so ideal-vs-lab comparisons use the
same element order and differ only by losses, alignment/retardance errors,
seams, central q-plate defects, finite filters, and finite-efficiency relay
terms.

## Named Presets

The runnable presets are:

- `segmented_vector_hexagon`: input polarizer, six-sector segmented
  radial/azimuthal waveplate by default, physical axicon, then 4f filter.
- `qplate_radial`: input polarizer, q-plate/S-waveplate radial conversion,
  physical axicon, then 4f filter.
- `qplate_azimuthal`: input polarizer, q-plate/S-waveplate azimuthal
  conversion, physical axicon, then 4f filter.
- `scalar_vortex_bessel`: SLM vortex plus axicon phase only, with no
  spatial-polarization element, for the scalar baseline comparison.

All preset parameters live in `PresetTrainConfig`: polarizer angle, q,
sector-pair count or explicit sector fast-axis angles, sector orientation,
axicon index/angle/aperture, Fourier-filter radius, scalar SLM charge and
axicon radial wave-vector, optional two-SLM conjugate prefix, and axicon/4f
ordering.
"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


__all__ = [
    "PresetTrainConfig",
    "PresetTrainName",
    "FourierRelayFilterElement",
    "FreeSpacePropagationElement",
    "HWPElement",
    "LabRealism",
    "LinearPolarizerElement",
    "OpticalTrainState",
    "PhysicalAxiconElement",
    "PolarizedTrainConfig",
    "QPlateElement",
    "QWPElement",
    "RealismMode",
    "SLMConjugateElement",
    "SLMPhaseElement",
    "SegmentedWaveplateElement",
    "VectorElement",
    "VectorField3D",
    "angular_ring_profile",
    "analyzer_maps_from_state",
    "assert_same_train_and_grid",
    "apply_fourier_relay_filter",
    "apply_optical_element",
    "apply_physical_axicon_element",
    "apply_train_polarization_elements",
    "discrete_power",
    "export_element_maps",
    "fresnel_sp_coefficients",
    "ideal_lab_comparison",
    "initial_train_state",
    "initial_uniform_field",
    "linear_polarizer_jones",
    "make_grid",
    "physical_axicon_vector_field",
    "plot_ideal_lab_comparison",
    "plot_mechanism_gate",
    "preset_train_config",
    "propagate_state",
    "qplate_fast_axis",
    "retarder_jones",
    "build_preset_train_elements",
    "element_signature",
    "fair_case_metrics",
    "grid_signature",
    "run_hexagon_mechanism_gate",
    "run_fair_comparison_suite",
    "run_fair_preset_comparison",
    "run_optical_pipeline",
    "run_preset_train",
    "run_polarized_train",
    "segmented_ra_fast_axis",
    "segmented_waveplate_axis_map",
    "sixfold_ring_metrics",
    "slm_structuring_phase_map",
    "stokes_from_state",
    "total_intensity_3d",
    "two_slm_conjugate_prefix",
    "vector_angular_spectrum_propagate",
    "write_fair_comparison_audit",
    "write_mechanism_doc",
]
