from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

Axis = Literal["x", "y"]


@dataclass(frozen=True)
class SlmHardwareProfile:
    """Locked lab-specific values that should not be casually edited in the GUI.

    These values describe the installed panel / carrier convention rather than an
    experiment.  Change them here only when the hardware or calibrated carrier
    convention changes.
    """

    name: str
    serial: str
    width_px: int
    height_px: int
    pixel_pitch_um: float
    wavelength_nm: float
    blaze_period_px: float
    blaze_axis: Axis
    blaze_sign: float
    output_bit_depth: int = 8
    carrier_enabled: bool = True

    @property
    def centre_x_px(self) -> float:
        return self.width_px / 2.0

    @property
    def centre_y_px(self) -> float:
        return self.height_px / 2.0

    @property
    def short_summary(self) -> str:
        return (
            f"{self.serial}  •  {self.width_px}×{self.height_px}  •  "
            f"{self.pixel_pitch_um:g} µm  •  {self.wavelength_nm:g} nm"
        )

    @property
    def carrier_summary(self) -> str:
        state = "ON" if self.carrier_enabled else "OFF"
        return (
            f"Carrier {state}: {self.blaze_period_px:g} px, "
            f"axis {self.blaze_axis}, sign {self.blaze_sign:+g}"
        )


# Lab working convention captured from the current GUI screenshots / configuration.
# If the verified carrier sign or period changes, change it once here rather than
# exposing it as an experiment knob on every run.
SLM_HARDWARE_PROFILES: Dict[str, SlmHardwareProfile] = {
    "SLM1": SlmHardwareProfile(
        name="SLM1",
        serial="6010-2382",
        width_px=1920,
        height_px=1080,
        pixel_pitch_um=8.0,
        wavelength_nm=1030.0,
        blaze_period_px=20.0,
        blaze_axis="y",
        blaze_sign=-1.0,
    ),
    "SLM2": SlmHardwareProfile(
        name="SLM2",
        serial="6010-2381",
        width_px=1920,
        height_px=1080,
        pixel_pitch_um=8.0,
        wavelength_nm=1030.0,
        blaze_period_px=20.0,
        blaze_axis="y",
        blaze_sign=-1.0,
    ),
}


def profile_for(name: str) -> SlmHardwareProfile:
    try:
        return SLM_HARDWARE_PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"No locked hardware profile exists for {name!r}.") from exc
