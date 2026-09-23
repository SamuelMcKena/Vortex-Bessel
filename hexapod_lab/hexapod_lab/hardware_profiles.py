from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class PockelsCandidate:
    key: str
    label: str
    gpio_name: str
    mask: int
    open_value: int | None
    closed_value: int | None
    confidence: str
    source_note: str

    @property
    def polarity_known(self) -> bool:
        return (
            self.open_value is not None
            and self.closed_value is not None
            and self.open_value != self.closed_value
        )


@dataclass(frozen=True, slots=True)
class LegacyHardwareProfile:
    hxp_host: str
    hxp_port: int
    hxp_timeout_s: float
    hxp_group: str
    coordinate_system: str
    pockels_candidates: tuple[PockelsCandidate, ...]
    gate_marker_gpio: str
    gate_marker_mask: int
    analog_monitor_gpio: str
    serial_resource: str

    def candidate(self, key: str) -> PockelsCandidate:
        for item in self.pockels_candidates:
            if item.key == key:
                return item
        raise KeyError(key)


def load_legacy_hardware_profile(path: str | Path) -> LegacyHardwareProfile:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))

    hxp = data["hxp_legacy_labview"]
    lv = data["labview_front_panel_v3"]
    tcl = data["tcl_processing_scripts"]

    labview = lv["pockels_candidate"]
    tcl_gate = tcl["beam_control"]

    # The later LabVIEW VI proves GPIO3.DO/mask 1/states 0 and 1, but the
    # recoverable source does not prove which state is physically OPEN on the
    # present wiring. Keep polarity explicitly unknown.
    candidates = (
        PockelsCandidate(
            key="labview_v3",
            label="LabVIEW v3 candidate — GPIO3.DO / mask 1",
            gpio_name=str(labview["gpio_name"]),
            mask=int(labview["mask"]),
            open_value=None,
            closed_value=None,
            confidence="channel/mask high; polarity unresolved",
            source_note=str(labview["evidence"]),
        ),
        PockelsCandidate(
            key="tcl_legacy",
            label="TCL legacy writing map — GPIO4.DO / mask 1",
            gpio_name=str(tcl_gate["gpio_name"]),
            mask=int(tcl_gate["mask"]),
            open_value=int(tcl_gate["write_value"]),
            closed_value=int(tcl_gate["non_write_value"]),
            confidence="high historical software evidence; physical wiring unverified",
            source_note=str(tcl_gate["evidence"]),
        ),
    )

    marker = lv["gate_state_marker"]
    power = lv["power_control"]

    return LegacyHardwareProfile(
        hxp_host=str(hxp["host"]),
        hxp_port=int(hxp["port"]),
        hxp_timeout_s=float(hxp["timeout_ms"]) / 1000.0,
        hxp_group=str(hxp["group"]),
        coordinate_system=str(hxp["coordinate_system"]),
        pockels_candidates=candidates,
        gate_marker_gpio=str(marker["gpio_name"]),
        gate_marker_mask=int(marker["mask"]),
        analog_monitor_gpio=str(power["analog_read_gpio"]),
        serial_resource=str(power["serial_resource"]),
    )
