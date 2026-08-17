"""Stage 8C.3R.5.3 coordinate-convention contract.

Declares the coordinate frames and inter-frame transforms that a physically
located scalar 4F model (and any later camera comparison) would require.  This is
a *contract / readiness* layer only: it declares conventions and records which
transforms are modelled vs require calibration.  It does NOT implement any
optical transform, thin-lens propagation, or camera mapping.

The model's own grid (``lab_beam_frame`` / phase-map frames) uses an explicit,
declared convention.  The physical SLM pixel orientation, the Fourier-plane
physical-position mapping, and the camera mapping are *unknown* and must block
measured-bench readiness until calibrated.

Boundary unchanged: n = 1.0 free-space; ``fourier_filter_physics_available=False``;
``diagnostic_only``; ``final_export_allowed=False``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CoordinateUnit = Literal["pixel", "um", "mm", "m", "cycles_per_m", "cycles_per_mm"]

VALID_UNITS: frozenset[str] = frozenset(
    {"pixel", "um", "mm", "m", "cycles_per_m", "cycles_per_mm"}
)
VALID_CALIBRATION_STATUS: frozenset[str] = frozenset(
    {"declared_model_convention", "unknown", "estimated", "measured", "factory"}
)
# provenance vocabulary is shared with control_contract
VALID_PROVENANCE: frozenset[str] = frozenset(
    {"measured", "manufacturer_specification", "estimated",
     "diagnostic_placeholder", "unknown", "derived"}
)


@dataclass(frozen=True)
class CoordinateFrame:
    frame_id: str
    display_name: str
    transverse_units: CoordinateUnit
    axial_units: CoordinateUnit
    positive_x_definition: str
    positive_y_definition: str
    positive_z_definition: str
    pixel_origin: str
    x_axis_flip: bool | None
    y_axis_flip: bool | None
    rotation_deg: float | None
    handedness: str
    calibration_status: str
    provenance: str
    note: str

    @property
    def mapping_known(self) -> bool:
        """True if the orientation/scale convention is explicit (not unknown)."""
        return self.calibration_status != "unknown" and self.provenance != "unknown"

    def as_row(self) -> dict:
        return {
            "frame": self.frame_id,
            "transverse": self.transverse_units,
            "axial": self.axial_units,
            "+x": self.positive_x_definition,
            "+y": self.positive_y_definition,
            "+z": self.positive_z_definition,
            "origin": self.pixel_origin,
            "x_flip": self.x_axis_flip,
            "y_flip": self.y_axis_flip,
            "rotation_deg": self.rotation_deg,
            "handedness": self.handedness,
            "calibration": self.calibration_status,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class CoordinateTransformDeclaration:
    transform_id: str
    source_frame_id: str
    destination_frame_id: str
    transform_type: str
    active: bool
    calibration_required: bool
    modelled: bool
    note: str

    @property
    def usable(self) -> bool:
        """True if the transform is available for use (modelled or not needed)."""
        return self.modelled or not self.calibration_required

    def as_row(self) -> dict:
        return {
            "transform": self.transform_id,
            "source": self.source_frame_id,
            "destination": self.destination_frame_id,
            "type": self.transform_type,
            "active": self.active,
            "calibration_required": self.calibration_required,
            "modelled": self.modelled,
        }


_DECL = "declared_model_convention"
_UNK = "unknown"


def build_coordinate_frames() -> tuple[CoordinateFrame, ...]:
    """Declared coordinate frames for the CSLM-to-axicon route and camera comparison."""
    return (
        CoordinateFrame(
            "lab_beam_frame", "Lab beam frame (model)", "um", "um",
            "+x to the right (model)", "+y upward (model)", "+z along propagation",
            "array centre (N/2)", False, False, 0.0, "right_handed",
            _DECL, "derived",
            "The simulation's own free-space frame; centred grid, right-handed."),
        CoordinateFrame(
            "SLM1_pixel_frame", "SLM1 pixel frame", "pixel", "um",
            "unknown physical x", "unknown physical y", "n/a",
            "unknown (device corner?)", None, None, None, "unknown",
            _UNK, "unknown",
            "Physical SLM1 pixel grid orientation/scale relative to lab is not calibrated."),
        CoordinateFrame(
            "SLM1_phase_map_frame", "SLM1 phase-map frame (model)", "um", "um",
            "+x to the right (model grid)", "+y upward (model grid)", "n/a",
            "array centre", False, False, 0.0, "right_handed",
            _DECL, "derived",
            "The SLM1 phase is computed on the model grid = lab_beam_frame convention."),
        CoordinateFrame(
            "SLM2_pixel_frame", "SLM2 pixel frame", "pixel", "um",
            "unknown physical x", "unknown physical y", "n/a",
            "unknown (device corner?)", None, None, None, "unknown",
            _UNK, "unknown",
            "Physical SLM2 pixel grid orientation/scale relative to lab is not calibrated; "
            "this sets carrier-order x/y placement and is mandatory for 4F."),
        CoordinateFrame(
            "SLM2_phase_map_frame", "SLM2 phase-map frame (model)", "um", "um",
            "+x to the right (model grid)", "+y upward (model grid)", "n/a",
            "array centre", False, False, 0.0, "right_handed",
            _DECL, "derived",
            "The SLM2 phase/carrier is computed on the model grid = lab_beam_frame convention."),
        CoordinateFrame(
            "physical_axicon_local_frame", "Physical axicon local frame", "um", "um",
            "+x to the right (model)", "+y upward (model)", "+z along propagation",
            "axicon apex / array centre", False, False, 0.0, "right_handed",
            _DECL, "diagnostic_placeholder",
            "Benchmark places the axicon on-axis in the model frame; physical mount "
            "orientation relative to lab is unknown."),
        CoordinateFrame(
            "Fourier_plane_spatial_frequency_frame", "Fourier-plane spatial-frequency frame",
            "cycles_per_m", "m", "+fx (fftshift convention)", "+fy (fftshift convention)", "n/a",
            "DC at array centre (fftshift)", False, False, 0.0, "right_handed",
            _DECL, "derived",
            "The FFT spatial-frequency convention is declared by the model (fftshift)."),
        CoordinateFrame(
            "Fourier_plane_physical_position_frame", "Fourier-plane physical-position frame",
            "mm", "mm", "unknown physical x", "unknown physical y", "n/a",
            "unknown", None, None, None, "unknown",
            _UNK, "unknown",
            "Physical mm position in the Fourier plane requires lens focal length + wavelength; "
            "no lens model exists, so this mapping is unknown."),
        CoordinateFrame(
            "camera_sensor_pixel_frame", "Camera sensor pixel frame", "pixel", "mm",
            "unknown physical x", "unknown physical y", "n/a",
            "unknown (sensor corner?)", None, None, None, "unknown",
            _UNK, "unknown",
            "Camera sensor pixel orientation/flip/rotation relative to lab is not calibrated."),
        CoordinateFrame(
            "camera_object_plane_frame", "Camera object-plane frame", "um", "um",
            "unknown physical x", "unknown physical y", "+z along propagation",
            "unknown", None, None, None, "unknown",
            _UNK, "unknown",
            "Camera object-plane mapping requires magnification + alignment; not calibrated."),
    )


def build_coordinate_transforms() -> tuple[CoordinateTransformDeclaration, ...]:
    """Declared inter-frame transforms; only model-internal ones are modelled."""
    return (
        CoordinateTransformDeclaration(
            "SLM1_pixel_to_phase_map", "SLM1_pixel_frame", "SLM1_phase_map_frame",
            "pixel_to_physical_scale_and_orientation", False, True, False,
            "Model treats SLM1 as a continuous phase screen; pixel->physical scale/orientation "
            "not calibrated."),
        CoordinateTransformDeclaration(
            "SLM1_phase_map_to_lab", "SLM1_phase_map_frame", "lab_beam_frame",
            "identity_model_convention", True, False, True,
            "Identity in the model (phase map is on the lab grid)."),
        CoordinateTransformDeclaration(
            "SLM2_pixel_to_phase_map", "SLM2_pixel_frame", "SLM2_phase_map_frame",
            "pixel_to_physical_scale_and_orientation", False, True, False,
            "Mandatory for physical carrier-order placement; pixel->physical scale/orientation "
            "not calibrated."),
        CoordinateTransformDeclaration(
            "SLM2_phase_map_to_lab", "SLM2_phase_map_frame", "lab_beam_frame",
            "identity_model_convention", True, False, True,
            "Identity in the model (phase map is on the lab grid)."),
        CoordinateTransformDeclaration(
            "SLM2_pixel_to_lab", "SLM2_pixel_frame", "lab_beam_frame",
            "physical_slm_to_lab", False, True, False,
            "Required for measured-bench prediction; not calibrated."),
        CoordinateTransformDeclaration(
            "fourier_frequency_to_physical_position", "Fourier_plane_spatial_frequency_frame",
            "Fourier_plane_physical_position_frame", "lens_focal_length_and_wavelength_scaling",
            False, True, False,
            "Maps spatial frequency to physical Fourier-plane mm via f and lambda; no lens "
            "model -> not modelled. KEY 4F blocker."),
        CoordinateTransformDeclaration(
            "fourier_physical_to_lab", "Fourier_plane_physical_position_frame", "lab_beam_frame",
            "physical_fourier_to_lab", False, True, False,
            "Required for measured-bench prediction; not calibrated."),
        CoordinateTransformDeclaration(
            "camera_pixel_to_object_plane", "camera_sensor_pixel_frame", "camera_object_plane_frame",
            "magnification_and_pixel_pitch", False, True, False,
            "Requires camera magnification + pixel pitch; not calibrated."),
        CoordinateTransformDeclaration(
            "camera_object_plane_to_lab", "camera_object_plane_frame", "lab_beam_frame",
            "physical_camera_to_lab", False, True, False,
            "Requires camera alignment; not calibrated."),
    )


def coordinate_frame_rows(frames=None) -> list[dict]:
    frames = frames if frames is not None else build_coordinate_frames()
    return [f.as_row() for f in frames]


def coordinate_transform_rows(transforms=None) -> list[dict]:
    transforms = transforms if transforms is not None else build_coordinate_transforms()
    return [t.as_row() for t in transforms]


def validate_coordinate_contract(frames=None, transforms=None) -> list[str]:
    """Return validation issues (empty == valid)."""
    frames = frames if frames is not None else build_coordinate_frames()
    transforms = transforms if transforms is not None else build_coordinate_transforms()
    issues: list[str] = []
    frame_ids = {f.frame_id for f in frames}
    for f in frames:
        if f.transverse_units not in VALID_UNITS:
            issues.append(f"{f.frame_id}: invalid transverse_units {f.transverse_units!r}")
        if f.axial_units not in VALID_UNITS:
            issues.append(f"{f.frame_id}: invalid axial_units {f.axial_units!r}")
        if f.calibration_status not in VALID_CALIBRATION_STATUS:
            issues.append(f"{f.frame_id}: invalid calibration_status {f.calibration_status!r}")
        if f.provenance not in VALID_PROVENANCE:
            issues.append(f"{f.frame_id}: invalid provenance {f.provenance!r}")
    for t in transforms:
        if t.source_frame_id not in frame_ids:
            issues.append(f"{t.transform_id}: unknown source frame {t.source_frame_id!r}")
        if t.destination_frame_id not in frame_ids:
            issues.append(f"{t.transform_id}: unknown destination frame {t.destination_frame_id!r}")
    return issues


# ---------------------------------------------------------------------------
# readiness helpers
# ---------------------------------------------------------------------------


def _frame(frames, frame_id: str) -> CoordinateFrame | None:
    return next((f for f in frames if f.frame_id == frame_id), None)


def _transform(transforms, transform_id: str) -> CoordinateTransformDeclaration | None:
    return next((t for t in transforms if t.transform_id == transform_id), None)


def fourier_position_mapping_known(frames=None) -> bool:
    """True only if the physical Fourier-plane position mapping is declared (not unknown)."""
    frames = frames if frames is not None else build_coordinate_frames()
    f = _frame(frames, "Fourier_plane_physical_position_frame")
    return bool(f is not None and f.mapping_known)


def slm2_to_lab_mapping_known(frames=None, transforms=None) -> bool:
    """True only if the SLM2 pixel->lab mapping is declared/modelled."""
    frames = frames if frames is not None else build_coordinate_frames()
    transforms = transforms if transforms is not None else build_coordinate_transforms()
    f = _frame(frames, "SLM2_pixel_frame")
    t = _transform(transforms, "SLM2_pixel_to_lab")
    return bool(f is not None and f.mapping_known and t is not None and t.usable)


def camera_to_lab_mapping_known(frames=None, transforms=None) -> bool:
    frames = frames if frames is not None else build_coordinate_frames()
    transforms = transforms if transforms is not None else build_coordinate_transforms()
    f = _frame(frames, "camera_sensor_pixel_frame")
    t = _transform(transforms, "camera_object_plane_to_lab")
    return bool(f is not None and f.mapping_known and t is not None and t.usable)
