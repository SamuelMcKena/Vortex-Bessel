"""Configuration dataclasses and unit constants for the VBB study.

This module is intentionally a leaf: it owns data shapes and literals, while
the facade and design modules own behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np


# Units
m = 1.0
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9
fs = 1e-15
kHz = 1e3
uJ = 1e-6

TWOPI = 2.0 * np.pi
EPS = 1e-30

PathKind = Literal["realistic", "ideal"]
PropagationMethod = Literal["bl_asm", "sas"]
GenerationMethod = Literal["holographic", "physical"]
Slm2ConjugateMode = Literal["full", "lowpass", "zernike", "preserve_vortex"]
StudyKind = Literal["beam_to_surface", "through_sample", "full_source_to_sample"]
RegimeName = Literal["general", "limits"]
ValidityViolationAction = Literal["flag", "warn", "raise"]
OpticalMappingMode = Literal["target_matched_inverse_design", "fixed_physical_optics"]
SurfacePlacement = Literal["zone_center", "zone_start", "custom"]


@dataclass
class LaserConfig:
    """PHAROS PH2-like source configuration."""

    name: str = "PHAROS PH2"
    wavelength_m: float = 1029.0 * nm
    pulse_duration_s: float = 260.0 * fs
    max_pulse_energy_J: float = 400.0 * uJ
    max_average_power_W: float = 10.0
    rep_rate_Hz: float = 100.0 * kHz
    input_pulse_energy_J: float = 10.0 * uJ
    beam_radius_on_slm_m: float = 2.0 * mm
    beam_radius_definition: str = "1/e field amplitude radius"

    @property
    def k0(self) -> float:
        return TWOPI / self.wavelength_m

    @property
    def average_power_W(self) -> float:
        return self.input_pulse_energy_J * self.rep_rate_Hz


@dataclass
class SLMConfig:
    """HOLOEYE LCOS-NIR-like phase-only SLM configuration."""

    name: str = "HOLOEYE LCOS-NIR"
    resolution_x: int = 1920
    resolution_y: int = 1080
    pixel_pitch_m: float = 8.0 * um
    phase_bits: int = 8
    fill_factor: float = 0.93
    blaze_period_px: int = 20
    first_order: int = 1
    first_order_filter_radius_lpmm: float = 2.5
    flip_x: bool = False
    flip_y: bool = False
    rotate_180: bool = False
    invert_gray: bool = False

    @property
    def active_width_m(self) -> float:
        return self.resolution_x * self.pixel_pitch_m

    @property
    def active_height_m(self) -> float:
        return self.resolution_y * self.pixel_pitch_m

    @property
    def blaze_period_m(self) -> float:
        return self.blaze_period_px * self.pixel_pitch_m

    @property
    def carrier_cpm(self) -> float:
        return self.first_order / self.blaze_period_m

    @property
    def carrier_lpmm(self) -> float:
        return self.carrier_cpm / 1e3


@dataclass
class ObjectiveConfig:
    """Scalar ideal-lens objective model."""

    NA: float = 0.45
    f_eff_m: float = 4.0 * mm
    immersion_n: float = 1.0
    pupil_fill: float = 0.95
    transmission: float = 0.90

    @property
    def pupil_radius_m(self) -> float:
        return self.f_eff_m * self.NA / max(self.immersion_n, EPS)


@dataclass
class RelayConfig:
    """Relay metadata for inverse-design bookkeeping."""

    name: str = "ideal relay"
    transmission: float = 0.90
    magnification_to_sample: Optional[float] = None
    effective_relay_f_m: float = 495.59763610450985 * mm


@dataclass
class MaterialConfig:
    """Material and material-proxy settings.

    The threshold values are planning proxies, not calibrated Cr:ZnSe damage or
    refractive-index-change laws.
    """

    name: str = "Cr:ZnSe"
    refractive_index: float = 2.44
    write_depth_m: float = 300.0 * um
    single_pulse_threshold_J_cm2: float = 2.0
    incubation_exponent: float = 0.84
    static_or_scan: str = "scan"
    n_static_pulses: int = 1
    scan_speed_m_s: float = 1.0 * mm
    feature_width_m: float = 3.0 * um
    side_lobe_exclusion_radius_factor: float = 1.6

    @classmethod
    def cr_znse(cls, write_depth_m: float = 300.0 * um) -> "MaterialConfig":
        return cls(name="Cr:ZnSe", refractive_index=2.44, write_depth_m=write_depth_m)

    @classmethod
    def fused_silica(cls, write_depth_m: float = 300.0 * um) -> "MaterialConfig":
        return cls(name="fused silica cross-check", refractive_index=1.45, write_depth_m=write_depth_m)

    def effective_pulses(self, rep_rate_Hz: float) -> float:
        mode = self.static_or_scan.lower().strip()
        if mode == "scan":
            return max(1.0, rep_rate_Hz * self.feature_width_m / max(self.scan_speed_m_s, EPS))
        return max(1.0, float(self.n_static_pulses))

    def incubated_threshold_J_cm2(self, rep_rate_Hz: float) -> float:
        n = self.effective_pulses(rep_rate_Hz)
        return self.single_pulse_threshold_J_cm2 * n ** (self.incubation_exponent - 1.0)


@dataclass
class EnergyBudget:
    """Pulse-energy throughput from PHAROS to the sample."""

    pulse_energy_in_J: float = 10.0 * uJ
    pre_slm_transmission: float = 1.0
    slm_reflectivity: float = 0.75
    first_order_efficiency: float = 0.45
    relay_transmission: float = 0.90
    focusing_transmission: float = 0.90
    sample_surface_transmission: float = 0.96
    user_extra_transmission: float = 1.0

    @property
    def total_transmission_without_first_order(self) -> float:
        return float(
            self.pre_slm_transmission
            * self.slm_reflectivity
            * self.relay_transmission
            * self.focusing_transmission
            * self.sample_surface_transmission
            * self.user_extra_transmission
        )

    @property
    def total_transmission(self) -> float:
        return self.total_transmission_without_first_order * self.first_order_efficiency

    @property
    def total_transmission_to_surface_air(self) -> float:
        return float(
            self.pre_slm_transmission
            * self.slm_reflectivity
            * self.first_order_efficiency
            * self.relay_transmission
            * self.focusing_transmission
            * self.user_extra_transmission
        )

    @property
    def pulse_energy_at_sample_J(self) -> float:
        return self.pulse_energy_in_J * self.total_transmission

    @property
    def pulse_energy_at_surface_air_J(self) -> float:
        return self.pulse_energy_in_J * self.total_transmission_to_surface_air


@dataclass
class BeamTarget:
    """Lab target for inverse design.

    ``target_core_diameter_m`` is retained for existing notebooks. In the
    current scalar design convention it means the equivalent ell=0 J0
    first-zero diameter. For ``ell > 0`` the actual bright vortex-ring diameter
    is derived separately from the first zero of ``J'_ell``.
    """

    ell: int = 3
    target_core_diameter_m: float = 3.0 * um
    target_bessel_length_m: float = 150.0 * um
    n_axicon: float = 1.5
    hologram_medium_n: float = 1.0
    signum_pi_flip: bool = False


@dataclass
class BeamDesign:
    """Computed SLM/sample design."""

    ell: int
    target_core_diameter_m: float
    target_scale_definition: str
    target_equivalent_l0_core_diameter_m: float
    target_bessel_length_m: float
    n_axicon: float
    hologram_medium_n: float
    sample_medium_n: float
    kr_sample_m_inv: float
    kr_slm_m_inv: float
    gamma_slm_rad: float
    gamma_slm_deg: float
    magnification_to_sample: float
    mapping_mode: OpticalMappingMode
    objective_map_source: str
    objective_map_demag: float
    w0_sample_m: float
    predicted_bessel_length_m: float
    equivalent_l0_core_radius_m: float
    equivalent_l0_core_diameter_m: float
    equivalent_l0_first_zero_radius_m: float
    equivalent_l0_first_zero_diameter_m: float
    vortex_main_ring_radius_m: float
    vortex_main_ring_diameter_m: float
    signum_pi_flip: bool = False


@dataclass
class GridConfig:
    """Numerical grids for the realistic and ideal paths."""

    N: int = 512
    device_downsample: int = 4
    axial_range_m: float = 120.0 * um
    axial_points: int = 41
    axial_target_factor: float = 1.8
    crop_pixels: int = 192
    coarse_scan_factor: float = 3.0
    coarse_scan_points: int = 25
    ideal_N: int = 512
    ideal_dx_m: float = 0.25 * um
    label: str = "fast"


@dataclass
class PropagationConfig:
    """Propagation-method selection and SAS numerical options."""

    method: PropagationMethod = "bl_asm"
    sas_pad_factor: int = 2
    sas_bandlimit: bool = True
    sas_skip_final_phase: bool = True
    sas_allow_invalid: bool = False


@dataclass
class PhysicalAxiconConfig:
    """Physical-axicon generation-stage settings.

    The defaults are deliberately conservative and deterministic. The physical
    path still reports the realised radial wavevector through ``AxiconResult``.
    """

    inter_slm_z_m: float = 25.0 * um
    inter_slm_n: float = 1.0
    slm2_conjugate_mode: Slm2ConjugateMode = "preserve_vortex"
    allow_vortex_removal: bool = False
    slm2_stroke_levels: Optional[int] = 256
    slm1_vortex_charge: Optional[int] = None
    n_axicon: Optional[float] = None
    axicon_medium_n: float = 1.0
    axicon_base_angle_deg: Optional[float] = None
    axicon_aperture_radius_m: Optional[float] = None
    slm1_transmission: float = 1.0
    slm2_transmission: float = 1.0
    axicon_transmission: float = 1.0


@dataclass
class SimulationPreset:
    """Named speed/quality preset."""

    name: str
    grid: GridConfig


@dataclass
class TwinConfig:
    """Complete case configuration."""

    laser: LaserConfig = field(default_factory=LaserConfig)
    slm: SLMConfig = field(default_factory=SLMConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    relay: RelayConfig = field(default_factory=RelayConfig)
    mapping_mode: OpticalMappingMode = "target_matched_inverse_design"
    material: MaterialConfig = field(default_factory=MaterialConfig.cr_znse)
    energy: EnergyBudget = field(default_factory=EnergyBudget)
    target: BeamTarget = field(default_factory=BeamTarget)
    grid: GridConfig = field(default_factory=GridConfig)
    propagation: PropagationConfig = field(default_factory=PropagationConfig)
    physical_axicon: PhysicalAxiconConfig = field(default_factory=PhysicalAxiconConfig)
    generation_method: GenerationMethod = "holographic"
    study_kind: StudyKind = "beam_to_surface"
    regime: RegimeName = "general"
    validity_on_violation: ValidityViolationAction = "flag"
    surface_placement: SurfacePlacement = "zone_center"
    surface_z_m: Optional[float] = None
    air_scan_half_span_factor: float = 1.25
    air_scan_coarse_span_factor: float = 4.0
    apply_interface: bool = True
    correct_interface: bool = False
    include_blaze: bool = True
    include_quantization: bool = True
    include_fill_factor: bool = True
    include_active_aperture: bool = True
    include_first_order_isolation: bool = True
    random_seed: int = 12345


__all__ = [
    "BeamDesign",
    "BeamTarget",
    "EnergyBudget",
    "EPS",
    "GenerationMethod",
    "GridConfig",
    "LaserConfig",
    "MaterialConfig",
    "ObjectiveConfig",
    "OpticalMappingMode",
    "PathKind",
    "PhysicalAxiconConfig",
    "PropagationConfig",
    "PropagationMethod",
    "RegimeName",
    "RelayConfig",
    "SLMConfig",
    "SimulationPreset",
    "Slm2ConjugateMode",
    "StudyKind",
    "SurfacePlacement",
    "TWOPI",
    "TwinConfig",
    "ValidityViolationAction",
    "cm",
    "fs",
    "kHz",
    "m",
    "mm",
    "nm",
    "uJ",
    "um",
]
