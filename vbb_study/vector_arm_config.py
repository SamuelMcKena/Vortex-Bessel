"""Configuration for the collinear dual-SLM vector-arm model.

This module is intentionally a leaf: it owns only data shapes and rig literals
for the vector arm. Behavioural modules are responsible for pulling the
physical axicon, relay, and objective mappings from the existing study config.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SLMPanelConfig:
    """HOLOEYE PLUTO-2.1-like phase-only panel settings for one arm panel.

    The physical-bench carrier is canonically a 20-pixel blaze period on the
    8 um panel.  Therefore ``carrier_lp_per_mm = 6.25``.  Older vector-route
    defaults used 6.94 lp/mm; that value was a route-local legacy setting and
    is not the physical-bench binding.
    """

    n_x: int = 1920
    n_y: int = 1080
    pitch_m: float = 8e-6
    phase_levels: int = 256
    fill_factor: float = 0.93
    carrier_lp_per_mm: float = 6.25
    carrier_sign: int = +1

    def __post_init__(self) -> None:
        if int(self.n_x) <= 0 or int(self.n_y) <= 0:
            raise ValueError("SLMPanelConfig dimensions must be positive.")
        if float(self.pitch_m) <= 0.0:
            raise ValueError("SLMPanelConfig.pitch_m must be positive.")
        if int(self.phase_levels) < 2:
            raise ValueError("SLMPanelConfig.phase_levels must be at least 2.")
        if not (0.0 <= float(self.fill_factor) <= 1.0):
            raise ValueError("SLMPanelConfig.fill_factor must be in [0, 1].")
        if float(self.carrier_lp_per_mm) <= 0.0:
            raise ValueError("SLMPanelConfig.carrier_lp_per_mm must be positive.")
        if int(self.carrier_sign) not in {-1, +1}:
            raise ValueError("SLMPanelConfig.carrier_sign must be -1 or +1.")

    @property
    def active_width_m(self) -> float:
        """Active panel width in metres."""

        return int(self.n_x) * float(self.pitch_m)

    @property
    def active_height_m(self) -> float:
        """Active panel height in metres."""

        return int(self.n_y) * float(self.pitch_m)

    @property
    def carrier_lp_per_m(self) -> float:
        """Configured carrier in line-pairs per metre, including sign."""

        return int(self.carrier_sign) * float(self.carrier_lp_per_mm) * 1e3

    @property
    def carrier_period_m(self) -> float:
        """Physical blaze period in metres."""

        return 1.0 / abs(self.carrier_lp_per_m)

    @property
    def carrier_period_px(self) -> float:
        """Physical blaze period in SLM pixels."""

        return self.carrier_period_m / float(self.pitch_m)


@dataclass(frozen=True)
class VectorArmConfig:
    """Parameters for the segmented radial/azimuthal vector-arm generator."""

    wavelength_m: float = 1029e-9
    pulse_duration_s: float = 260e-15
    waist_m: float = 2e-3
    n_pairs: int = 3
    sector_duty: float = 0.5
    sector_rotation_rad: float = 0.0
    piston_delta_rad: float = 0.0
    slm1: SLMPanelConfig = field(default_factory=SLMPanelConfig)
    slm2: SLMPanelConfig = field(default_factory=lambda: SLMPanelConfig(carrier_sign=-1))
    quantise: bool = True
    apply_fill_factor: bool = True
    fill_factor_model: str = "coherent_unmodulated_deadspace"
    apply_carrier: bool = True
    iris_radius_frac: float = 0.45
    ideal_components: bool = False
    hwp_retardance_error_rad: float = 0.0
    qwp_retardance_error_rad: float = 0.0

    def __post_init__(self) -> None:
        if float(self.wavelength_m) <= 0.0:
            raise ValueError("VectorArmConfig.wavelength_m must be positive.")
        if float(self.pulse_duration_s) <= 0.0:
            raise ValueError("VectorArmConfig.pulse_duration_s must be positive.")
        if float(self.waist_m) <= 0.0:
            raise ValueError("VectorArmConfig.waist_m must be positive.")
        if int(self.n_pairs) <= 0:
            raise ValueError("VectorArmConfig.n_pairs must be positive.")
        if not (0.0 < float(self.sector_duty) < 1.0):
            raise ValueError("VectorArmConfig.sector_duty must be strictly between 0 and 1.")
        if not (0.0 < float(self.iris_radius_frac) <= 1.0):
            raise ValueError("VectorArmConfig.iris_radius_frac must be in (0, 1].")
        if self.fill_factor_model not in {
            "throughput_only",
            "resolved_pixel_aperture",
            "coherent_unmodulated_deadspace",
        }:
            raise ValueError("VectorArmConfig.fill_factor_model is not a supported SLM model.")

    @property
    def effective_slm1(self) -> SLMPanelConfig:
        """SLM1 settings after ideal-component overrides."""

        if not self.ideal_components:
            return self.slm1
        return SLMPanelConfig(
            n_x=self.slm1.n_x,
            n_y=self.slm1.n_y,
            pitch_m=self.slm1.pitch_m,
            phase_levels=self.slm1.phase_levels,
            fill_factor=1.0,
            carrier_lp_per_mm=self.slm1.carrier_lp_per_mm,
            carrier_sign=self.slm1.carrier_sign,
        )

    @property
    def effective_slm2(self) -> SLMPanelConfig:
        """SLM2 settings after ideal-component overrides."""

        if not self.ideal_components:
            return self.slm2
        return SLMPanelConfig(
            n_x=self.slm2.n_x,
            n_y=self.slm2.n_y,
            pitch_m=self.slm2.pitch_m,
            phase_levels=self.slm2.phase_levels,
            fill_factor=1.0,
            carrier_lp_per_mm=self.slm2.carrier_lp_per_mm,
            carrier_sign=self.slm2.carrier_sign,
        )


__all__ = ["SLMPanelConfig", "VectorArmConfig"]
