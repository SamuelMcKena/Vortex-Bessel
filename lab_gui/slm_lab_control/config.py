from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal

from .hardware_profiles import profile_for

Axis = Literal["x", "y"]
TransferMode = Literal["png_file", "phase_file", "direct_gray_array", "direct_phase_array", "auto"]


@dataclass
class SlmGeometry:
    """Physical/pixel geometry used by the phase generator.

    In the lab GUI these values are sourced from the locked hardware profile and
    are deliberately not exposed as normal experiment controls.
    """

    width_px: int = 1920
    height_px: int = 1080
    pixel_pitch_um: float = 8.0
    wavelength_nm: float = 1030.0

    @property
    def pixel_pitch_m(self) -> float:
        return self.pixel_pitch_um * 1e-6

    @property
    def wavelength_m(self) -> float:
        return self.wavelength_nm * 1e-9

    @property
    def wavelength_um(self) -> float:
        return self.wavelength_nm * 1e-3


@dataclass
class TermSwitches:
    wavefront: bool = False
    blaze: bool = True
    focus: bool = False
    axicon: bool = False
    vortex: bool = False
    spherical_interface: bool = False
    zernike_z40: bool = False  # kept for preset compatibility; now enables low-order Zernikes
    n_fold: bool = False
    circular_pupil: bool = False
    retrieved_correction: bool = False
    custom_phase: bool = False


@dataclass
class SlmPhaseConfig:
    """Complete phase state for one SLM.

    Hardware-specific fields remain in the dataclass so the numerical backend and
    historic presets remain compatible, but the GUI locks them to the installed
    hardware profile on load/regeneration.
    """

    name: str = "SLM"
    serial: str = ""
    geometry: SlmGeometry = field(default_factory=SlmGeometry)
    switches: TermSwitches = field(default_factory=TermSwitches)

    # Base / compensation
    background_gray: float = 0.0
    wavefront_path: str = ""
    wavefront_gain: float = 1.0

    # Alignment and pupil
    center_x_px: float = 960.0
    center_y_px: float = 540.0
    pupil_diameter_mm: float = 9.0
    term_rotation_deg: float = 0.0

    # Locked carrier / grating values (applied from hardware profile in lab GUI)
    blaze_period_px: float = 20.0
    blaze_axis: Axis = "y"
    blaze_sign: float = -1.0

    # Lens / focus
    focus_focal_length_mm: float = 0.0
    focus_sign: float = 1.0

    # Digital axicon / Bessel conical phase
    axicon_period_px: float = 80.0
    axicon_sign: float = 1.0

    # Vortex
    vortex_charge: int = 0

    # Spherical interface correction
    interface_NA: float = 0.4
    interface_depth_um: float = 0.0
    interface_n1: float = 1.0
    interface_n2: float = 1.45
    interface_sign: float = -1.0

    # Low-order Zernike correction coefficients, in waves.
    # z40_amp_waves existed in v0.2 and is retained unchanged.
    z20_amp_waves: float = 0.0
    z22_cos_amp_waves: float = 0.0
    z22_sin_amp_waves: float = 0.0
    z31_cos_amp_waves: float = 0.0
    z31_sin_amp_waves: float = 0.0
    z40_amp_waves: float = 0.0

    # Simple symmetry perturbation, useful for polygonal/hexagonal experiments
    n_fold_order: int = 6
    n_fold_amp_waves: float = 0.0

    # Retrieved/inverse correction map. Preferred format is a native 2-D .npy
    # array in radians. It is added to the existing phase stack before the one
    # final wrap. Gain can be negative for a sign/convention test.
    retrieved_correction_path: str = ""
    retrieved_correction_gain: float = 0.05

    # Arbitrary extra grayscale phase overlay. Useful for experimental masks that
    # do not yet deserve their own coded phase term.
    custom_phase_path: str = ""
    custom_phase_gain: float = 1.0

    # Output
    global_phase_gain: float = 1.0
    output_bit_depth: int = 8

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SlmPhaseConfig":
        data = dict(data)
        if isinstance(data.get("geometry"), dict):
            data["geometry"] = SlmGeometry(**data["geometry"])
        if isinstance(data.get("switches"), dict):
            switch_data = dict(data["switches"])
            # Forward compatibility with v0.2 preset files.
            allowed = TermSwitches.__dataclass_fields__.keys()
            data["switches"] = TermSwitches(**{k: v for k, v in switch_data.items() if k in allowed})
        allowed_fields = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in allowed_fields})

    def apply_locked_hardware(self, preserve_alignment_centre: bool = True) -> None:
        """Overwrite non-experimental values from the installed lab profile."""

        profile = profile_for(self.name)
        old_default_x = self.geometry.width_px / 2.0
        old_default_y = self.geometry.height_px / 2.0
        centre_was_default = (
            abs(self.center_x_px - old_default_x) < 1e-9
            and abs(self.center_y_px - old_default_y) < 1e-9
        )

        self.serial = profile.serial
        self.geometry = SlmGeometry(
            width_px=profile.width_px,
            height_px=profile.height_px,
            pixel_pitch_um=profile.pixel_pitch_um,
            wavelength_nm=profile.wavelength_nm,
        )
        self.blaze_period_px = profile.blaze_period_px
        self.blaze_axis = profile.blaze_axis
        self.blaze_sign = profile.blaze_sign
        self.output_bit_depth = profile.output_bit_depth
        self.switches.blaze = profile.carrier_enabled

        # Preserve a deliberately measured beam centre, but migrate an old preset's
        # mathematical default to the new profile centre if geometry ever changes.
        if not preserve_alignment_centre or centre_was_default:
            self.center_x_px = profile.centre_x_px
            self.center_y_px = profile.centre_y_px


@dataclass
class AppConfig:
    """Top-level GUI/backend state."""

    backend: Literal["dummy", "heds"] = "dummy"
    sdk_major: int = 4
    sdk_minor: int = 2
    slm1: SlmPhaseConfig = field(
        default_factory=lambda: SlmPhaseConfig(name="SLM1", serial="6010-2382")
    )
    slm2: SlmPhaseConfig = field(
        default_factory=lambda: SlmPhaseConfig(name="SLM2", serial="6010-2381")
    )
    output_root: str = "outputs/casts"
    preset_root: str = "presets"
    auto_regenerate: bool = True
    transfer_mode: TransferMode = "png_file"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        data = dict(data)
        if isinstance(data.get("slm1"), dict):
            data["slm1"] = SlmPhaseConfig.from_dict(data["slm1"])
        if isinstance(data.get("slm2"), dict):
            data["slm2"] = SlmPhaseConfig.from_dict(data["slm2"])
        allowed = cls.__dataclass_fields__.keys()
        cfg = cls(**{k: v for k, v in data.items() if k in allowed})
        cfg.apply_locked_hardware()
        return cfg

    def apply_locked_hardware(self) -> None:
        self.slm1.name = "SLM1"
        self.slm2.name = "SLM2"
        self.slm1.apply_locked_hardware()
        self.slm2.apply_locked_hardware()

    def resolve_output_root(self, project_root: Path) -> Path:
        p = Path(self.output_root)
        return p if p.is_absolute() else project_root / p

    def resolve_preset_root(self, project_root: Path) -> Path:
        p = Path(self.preset_root)
        return p if p.is_absolute() else project_root / p
