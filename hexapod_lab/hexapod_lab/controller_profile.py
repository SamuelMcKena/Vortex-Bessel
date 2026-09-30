from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ActuatorConfig:
    positioner: str
    stage_name: str
    minimum_target_position_mm: float
    maximum_target_position_mm: float
    home_preset_mm: float
    maximum_velocity_mm_s: float
    maximum_acceleration_mm_s2: float
    encoder_resolution_mm: float
    backlash_mm: float


@dataclass(frozen=True, slots=True)
class HXPControllerProfile:
    host: str
    port: int
    timeout_s: float
    group: str
    coordinate_system: str
    work_in_world: tuple[float, float, float, float, float, float]
    base_in_world: tuple[float, float, float, float, float, float]
    tool_in_carriage: tuple[float, float, float, float, float, float]
    geometry: dict[str, Any]
    actuators: tuple[ActuatorConfig, ...]
    attenuator_gpio: str
    attenuator_raw_min: float
    attenuator_raw_max: float
    attenuator_transmission_min_percent: float
    attenuator_transmission_max_percent: float
    pockels_candidates: tuple[dict[str, Any], ...]

    def raw_to_transmission_percent(self, raw: float) -> float:
        raw = float(raw)
        span_raw = self.attenuator_raw_max - self.attenuator_raw_min
        span_pct = self.attenuator_transmission_max_percent - self.attenuator_transmission_min_percent
        if span_raw <= 0:
            raise ValueError("invalid attenuator raw calibration span")
        pct = self.attenuator_transmission_min_percent + ((raw - self.attenuator_raw_min) / span_raw) * span_pct
        return max(self.attenuator_transmission_min_percent, min(self.attenuator_transmission_max_percent, pct))

    def transmission_percent_to_raw(self, percent: float) -> float:
        percent = float(percent)
        lo = self.attenuator_transmission_min_percent
        hi = self.attenuator_transmission_max_percent
        if not lo <= percent <= hi:
            raise ValueError(f"attenuator transmission must be between {lo:g} and {hi:g} %")
        span_pct = hi - lo
        if span_pct <= 0:
            raise ValueError("invalid attenuator transmission calibration span")
        return self.attenuator_raw_min + ((percent - lo) / span_pct) * (self.attenuator_raw_max - self.attenuator_raw_min)


def load_hxp_controller_profile(path: str | Path) -> HXPControllerProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    network = data["network"]
    group = data["group"]
    frames = data["frames"]
    att = data["io"]["attenuator"]
    candidates = (data["io"]["labview_pockels_candidate"], data["io"]["tcl_pockels_candidate"])
    actuators = tuple(ActuatorConfig(positioner=str(item["positioner"]), stage_name=str(item["stage_name"]), minimum_target_position_mm=float(item["minimum_target_position_mm"]), maximum_target_position_mm=float(item["maximum_target_position_mm"]), home_preset_mm=float(item["home_preset_mm"]), maximum_velocity_mm_s=float(item["maximum_velocity_mm_s"]), maximum_acceleration_mm_s2=float(item["maximum_acceleration_mm_s2"]), encoder_resolution_mm=float(item["encoder_resolution_mm"]), backlash_mm=float(item["backlash_mm"])) for item in data["actuators"])
    def six(key: str) -> tuple[float, float, float, float, float, float]:
        vals = tuple(float(v) for v in frames[key])
        if len(vals) != 6:
            raise ValueError(f"{key} must contain six values")
        return vals  # type: ignore[return-value]
    return HXPControllerProfile(host=str(network["host"]), port=int(network["port"]), timeout_s=float(network["timeout_s"]), group=str(group["name"]), coordinate_system=str(group["coordinate_system"]), work_in_world=six("work_in_world"), base_in_world=six("base_in_world"), tool_in_carriage=six("tool_in_carriage"), geometry=dict(data["geometry"]), actuators=actuators, attenuator_gpio=str(att["gpio"]), attenuator_raw_min=float(att["raw_min"]), attenuator_raw_max=float(att["raw_max"]), attenuator_transmission_min_percent=float(att["transmission_min_percent"]), attenuator_transmission_max_percent=float(att["transmission_max_percent"]), pockels_candidates=tuple(dict(v) for v in candidates))
