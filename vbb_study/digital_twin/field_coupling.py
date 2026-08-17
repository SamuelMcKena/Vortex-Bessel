"""
Canonical optical-field adapters for the beam-to-write digital twin (Stage 8C).

This module bridges the existing repository optical-field outputs
(``vbb_study.vbb_studies.SurfaceField`` and the ``propagate_volume`` volume dict)
to a small, explicit, canonical representation that the Stage 8C fluence layer
can consume without guessing sampling or array conventions.

Canonical conventions (explicit, non-negotiable):

    2D plane : intensity[y, x]
    3D stack : intensity[z, y, x]
    x_um     : x coordinate array (microns), strictly monotonic
    y_um     : y coordinate array (microns), strictly monotonic
    z_um     : z coordinate array (microns), strictly monotonic

Model status: this module produces *optical-field containers only*.  It does not
compute fluence, dose, absorption, or any material response.  It carries the
``source_status`` of the originating field so downstream layers can refuse to
treat fabricated arrays as real optical predictions.

Hard rules honoured here:
  - No analytic placeholder beams are fabricated.  Constructors accept arrays
    that the *caller* supplies; production notebook paths must not use
    ``source_status="synthetic_placeholder"`` (that string is reserved and
    flagged as non-governed).
  - If a real optical field is unavailable or unsupported, we fail loudly with a
    clear, specific error rather than silently inventing sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

STAGE = "stage8c_surfacefield_energy_scaled_cockpit"

# Source-status tokens with explicit governance meaning.
SOURCE_REAL_OPTICAL_FIELD = "real_optical_field"
SOURCE_UNIT_TEST_FIXTURE = "unit_test_fixture"
SOURCE_PROVIDED_ARRAY = "provided_array"
SOURCE_SYNTHETIC_PLACEHOLDER = "synthetic_placeholder"  # reserved; non-governed

# Source statuses that are NOT allowed to back a governed/saved Stage 8C output.
NON_GOVERNED_SOURCE_STATUSES = frozenset({SOURCE_SYNTHETIC_PLACEHOLDER})

_UM_PER_M = 1e6  # metres -> microns


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MissingOpticalFieldError(RuntimeError):
    """Raised when a real optical field is required but none is available."""


class InvalidOpticalFieldError(ValueError):
    """Raised when an optical-field array fails validation."""


class UnsupportedSurfaceFieldError(TypeError):
    """Raised when a field object cannot be adapted to a canonical container."""


# ---------------------------------------------------------------------------
# Canonical dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpticalFieldPlane:
    """A single transverse optical-intensity plane in canonical [y, x] form."""

    intensity: np.ndarray
    dx_um: float
    dy_um: float
    z_um: float | None
    field_label: str
    source_status: str
    x_um: np.ndarray | None = None
    y_um: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = validate_intensity_plane(self.intensity, name="intensity")
        dxv, dyv = _validate_positive_sampling(self.dx_um, self.dy_um)
        ny, nx = arr.shape
        if self.x_um is None:
            x = _centered_coords_um(nx, dxv)
        else:
            x = _validate_monotonic_coords(self.x_um, name="x_um")
            if x.size != nx:
                raise InvalidOpticalFieldError(
                    f"len(x_um)={x.size} does not match intensity.shape[1]={nx}."
                )
        if self.y_um is None:
            y = _centered_coords_um(ny, dyv)
        else:
            y = _validate_monotonic_coords(self.y_um, name="y_um")
            if y.size != ny:
                raise InvalidOpticalFieldError(
                    f"len(y_um)={y.size} does not match intensity.shape[0]={ny}."
                )
        object.__setattr__(self, "intensity", arr)
        object.__setattr__(self, "dx_um", dxv)
        object.__setattr__(self, "dy_um", dyv)
        object.__setattr__(self, "x_um", x)
        object.__setattr__(self, "y_um", y)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def is_governed_source(self) -> bool:
        """True if this plane may back a governed/saved output."""
        return self.source_status not in NON_GOVERNED_SOURCE_STATUSES


@dataclass(frozen=True)
class OpticalFieldStack:
    """A 3D optical-intensity stack in canonical [z, y, x] form."""

    intensity_zyx: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    z_um: np.ndarray
    field_label: str
    source_status: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_governed_source(self) -> bool:
        """True if this stack may back a governed/saved output."""
        return self.source_status not in NON_GOVERNED_SOURCE_STATUSES

    @property
    def dx_um(self) -> float:
        return _uniform_spacing(self.x_um, "x_um")

    @property
    def dy_um(self) -> float:
        return _uniform_spacing(self.y_um, "y_um")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_intensity_plane(intensity: np.ndarray, *, name: str = "intensity") -> np.ndarray:
    """Validate and return a 2D non-negative finite intensity plane [y, x]."""
    arr = np.asarray(intensity, dtype=float)
    if arr.size == 0:
        raise InvalidOpticalFieldError(f"{name} is empty.")
    if arr.ndim != 2:
        raise InvalidOpticalFieldError(
            f"{name} must be a 2D [y, x] array; got ndim={arr.ndim}, shape={arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise InvalidOpticalFieldError(f"{name} contains non-finite values (NaN/inf).")
    if np.any(arr < 0.0):
        raise InvalidOpticalFieldError(f"{name} contains negative values; intensity must be >= 0.")
    return arr


def validate_intensity_stack(intensity_zyx: np.ndarray, *, name: str = "intensity_zyx") -> np.ndarray:
    """Validate and return a 3D non-negative finite intensity stack [z, y, x]."""
    arr = np.asarray(intensity_zyx, dtype=float)
    if arr.size == 0:
        raise InvalidOpticalFieldError(f"{name} is empty.")
    if arr.ndim != 3:
        raise InvalidOpticalFieldError(
            f"{name} must be a 3D [z, y, x] array; got ndim={arr.ndim}, shape={arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise InvalidOpticalFieldError(f"{name} contains non-finite values (NaN/inf).")
    if np.any(arr < 0.0):
        raise InvalidOpticalFieldError(f"{name} contains negative values; intensity must be >= 0.")
    return arr


def _validate_positive_sampling(dx_um: float, dy_um: float) -> tuple[float, float]:
    if not np.isfinite(dx_um) or dx_um <= 0:
        raise InvalidOpticalFieldError(f"dx_um must be positive and finite; got {dx_um}.")
    if not np.isfinite(dy_um) or dy_um <= 0:
        raise InvalidOpticalFieldError(f"dy_um must be positive and finite; got {dy_um}.")
    return float(dx_um), float(dy_um)


def _centered_coords_um(n: int, spacing_um: float) -> np.ndarray:
    """Return coordinates centred on zero using the pixel-centre convention."""
    if n <= 0:
        raise InvalidOpticalFieldError(f"coordinate length must be positive; got {n}.")
    if not np.isfinite(spacing_um) or spacing_um <= 0:
        raise InvalidOpticalFieldError(f"spacing_um must be positive and finite; got {spacing_um}.")
    return (np.arange(int(n), dtype=float) - (int(n) - 1) / 2.0) * float(spacing_um)


def _validate_monotonic_coords(coords: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise InvalidOpticalFieldError(f"{name} must be a non-empty 1D array.")
    if not np.all(np.isfinite(arr)):
        raise InvalidOpticalFieldError(f"{name} contains non-finite values.")
    if arr.size >= 2:
        diffs = np.diff(arr)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            raise InvalidOpticalFieldError(f"{name} must be strictly monotonic.")
    return arr


def _uniform_spacing(coords: np.ndarray, name: str) -> float:
    """Return the mean absolute spacing of a 1D coordinate array (microns)."""
    arr = np.asarray(coords, dtype=float)
    if arr.size < 2:
        raise InvalidOpticalFieldError(
            f"{name} needs at least two samples to define a pixel pitch."
        )
    return float(np.mean(np.abs(np.diff(arr))))


# ---------------------------------------------------------------------------
# Canonical constructors (from raw arrays)
# ---------------------------------------------------------------------------


def plane_from_arrays(
    intensity: np.ndarray,
    dx_um: float,
    dy_um: float,
    z_um: float | None = None,
    field_label: str = "array_field",
    source_status: str = SOURCE_PROVIDED_ARRAY,
    metadata: Mapping[str, Any] | None = None,
    *,
    x_um: np.ndarray | None = None,
    y_um: np.ndarray | None = None,
) -> OpticalFieldPlane:
    """Build a canonical :class:`OpticalFieldPlane` from a raw intensity array.

    This is allowed for tests and for callers that already hold a real intensity
    array.  It does NOT fabricate a beam: it only wraps the array the caller
    supplies.  Production paths must not pass ``source_status='synthetic_placeholder'``.
    """
    arr = validate_intensity_plane(intensity, name="intensity")
    dxv, dyv = _validate_positive_sampling(dx_um, dy_um)
    if z_um is not None and not np.isfinite(z_um):
        raise InvalidOpticalFieldError(f"z_um must be finite or None; got {z_um}.")
    return OpticalFieldPlane(
        intensity=arr,
        dx_um=dxv,
        dy_um=dyv,
        z_um=None if z_um is None else float(z_um),
        field_label=str(field_label),
        source_status=str(source_status),
        x_um=x_um,
        y_um=y_um,
        metadata=dict(metadata or {}),
    )


def stack_from_arrays(
    intensity_zyx: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    z_um: np.ndarray,
    field_label: str = "array_stack",
    source_status: str = SOURCE_PROVIDED_ARRAY,
    metadata: Mapping[str, Any] | None = None,
) -> OpticalFieldStack:
    """Build a canonical :class:`OpticalFieldStack` from raw arrays.

    Enforces the [z, y, x] convention and coordinate/shape consistency.
    """
    arr = validate_intensity_stack(intensity_zyx, name="intensity_zyx")
    x = _validate_monotonic_coords(x_um, name="x_um")
    y = _validate_monotonic_coords(y_um, name="y_um")
    z = _validate_monotonic_coords(z_um, name="z_um")
    nz, ny, nx = arr.shape
    if (nz, ny, nx) != (z.size, y.size, x.size):
        raise InvalidOpticalFieldError(
            "intensity_zyx shape does not match coordinate lengths: "
            f"intensity[z,y,x]={arr.shape} but (len z, len y, len x)="
            f"({z.size}, {y.size}, {x.size}). Convention is intensity[z, y, x]."
        )
    return OpticalFieldStack(
        intensity_zyx=arr,
        x_um=x,
        y_um=y,
        z_um=z,
        field_label=str(field_label),
        source_status=str(source_status),
        metadata=dict(metadata or {}),
    )


# ---------------------------------------------------------------------------
# Inspection helpers for existing repository field objects
# ---------------------------------------------------------------------------


def _is_mapping(obj: Any) -> bool:
    return isinstance(obj, Mapping)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Attribute- or key-style access for objects and mappings."""
    if _is_mapping(obj):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _has(obj: Any, key: str) -> bool:
    if _is_mapping(obj):
        return key in obj
    return hasattr(obj, key)


def _intensity_2d_from_object(obj: Any) -> tuple[np.ndarray, str]:
    """Extract a 2D intensity plane and a label describing the components used.

    Supports, in priority order:
      - complex field components Ex/Ey/Ez  -> |Ex|^2 + |Ey|^2 + |Ez|^2
      - an existing 2D real intensity under intensity_xy / intensity / I
      - a single complex field under field / E -> |field|^2
    """
    # Vector / scalar complex field components (repo SurfaceField uses Ex/Ey/Ez).
    if _has(obj, "Ex") and _get(obj, "Ex") is not None:
        used = []
        total = None
        for comp in ("Ex", "Ey", "Ez"):
            val = _get(obj, comp)
            if val is None:
                continue
            a = np.asarray(val)
            contrib = np.abs(a) ** 2
            total = contrib if total is None else total + contrib
            used.append(comp)
        if total is None:
            raise UnsupportedSurfaceFieldError("Field object exposes Ex but all components are None.")
        return np.asarray(total, dtype=float), "|" + "|^2+|".join(used) + "|^2"

    for key in ("intensity_xy", "intensity", "I"):
        if _has(obj, key) and _get(obj, key) is not None:
            arr = np.asarray(_get(obj, key))
            if arr.ndim == 2:
                return np.asarray(arr, dtype=float), key
            # 3D under "intensity" is a stack, not a plane.
            if arr.ndim == 3:
                raise UnsupportedSurfaceFieldError(
                    f"'{key}' is 3D ({arr.shape}); use extract_stack_from_surfacefield for stacks."
                )

    for key in ("field", "E"):
        if _has(obj, key) and _get(obj, key) is not None:
            arr = np.asarray(_get(obj, key))
            if arr.ndim == 2:
                return np.abs(arr).astype(float) ** 2, f"|{key}|^2"

    raise UnsupportedSurfaceFieldError(
        "Could not find a 2D intensity in the field object. Looked for "
        "Ex/Ey/Ez, intensity_xy, intensity, I, field, E. "
        f"Available attributes: {_describe_keys(obj)}."
    )


def _sampling_um_from_object(obj: Any, n_y: int, n_x: int) -> tuple[float, float, float | None]:
    """Return (dx_um, dy_um, z_um) for a 2D field object, converting units.

    Supports, in priority order:
      - explicit micron sampling dx_um/dy_um
      - micron coordinate arrays x_um/y_um (pitch from spacing)
      - a repository ``grid`` dict with 'dx' and 'x' in metres
      - metre sampling dx_m / dx (assumed metres)
    z_um is taken from z_um, or converted from z_surface_m, if present.
    """
    dx_um = dy_um = None

    if _has(obj, "dx_um") and _get(obj, "dx_um") is not None:
        dx_um = float(_get(obj, "dx_um"))
        dy_um = float(_get(obj, "dy_um", dx_um))
    elif _has(obj, "x_um") and _get(obj, "x_um") is not None:
        dx_um = _uniform_spacing(np.asarray(_get(obj, "x_um"), dtype=float), "x_um")
        y_um = _get(obj, "y_um")
        dy_um = _uniform_spacing(np.asarray(y_um, dtype=float), "y_um") if y_um is not None else dx_um
    elif _has(obj, "grid") and _get(obj, "grid") is not None:
        grid = _get(obj, "grid")
        dx_m = _get(grid, "dx")
        if dx_m is None:
            x_m = _get(grid, "x")
            if x_m is None:
                raise UnsupportedSurfaceFieldError(
                    "grid has neither 'dx' nor 'x'; cannot determine sampling."
                )
            dx_m = _uniform_spacing(np.asarray(x_m, dtype=float), "grid['x']") / _UM_PER_M
        dx_um = dy_um = float(dx_m) * _UM_PER_M
    elif _has(obj, "dx_m") and _get(obj, "dx_m") is not None:
        dx_um = dy_um = float(_get(obj, "dx_m")) * _UM_PER_M
    elif _has(obj, "dx") and _get(obj, "dx") is not None:
        dx_um = dy_um = float(_get(obj, "dx")) * _UM_PER_M

    if dx_um is None or dy_um is None:
        raise UnsupportedSurfaceFieldError(
            "Could not determine transverse sampling (dx/dy). Looked for "
            "dx_um/dy_um, x_um/y_um, grid['dx'|'x'], dx_m, dx. "
            f"Available attributes: {_describe_keys(obj)}."
        )

    z_um: float | None = None
    if _has(obj, "z_um") and _get(obj, "z_um") is not None:
        z_um = float(_get(obj, "z_um"))
    elif _has(obj, "z_surface_m") and _get(obj, "z_surface_m") is not None:
        z_um = float(_get(obj, "z_surface_m")) * _UM_PER_M

    return float(dx_um), float(dy_um), z_um


def _xy_coords_um_from_object(
    obj: Any,
    n_y: int,
    n_x: int,
    dx_um: float,
    dy_um: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract transverse coordinate arrays for a 2D field object."""
    metadata: dict[str, Any] = {}
    x_um = y_um = None

    if _has(obj, "x_um") and _get(obj, "x_um") is not None:
        x_um = np.asarray(_get(obj, "x_um"), dtype=float)
    if _has(obj, "y_um") and _get(obj, "y_um") is not None:
        y_um = np.asarray(_get(obj, "y_um"), dtype=float)

    grid = _get(obj, "grid", None) if _has(obj, "grid") else None
    if grid is not None:
        if x_um is None and _get(grid, "x") is not None:
            x_um = np.asarray(_get(grid, "x"), dtype=float) * _UM_PER_M
        if y_um is None and _get(grid, "y") is not None:
            y_um = np.asarray(_get(grid, "y"), dtype=float) * _UM_PER_M
        if x_um is None and _get(grid, "X") is not None:
            X = np.asarray(_get(grid, "X"), dtype=float)
            if X.ndim == 2 and X.shape[1] == n_x:
                x_um = X[0, :] * _UM_PER_M
        if y_um is None and _get(grid, "Y") is not None:
            Y = np.asarray(_get(grid, "Y"), dtype=float)
            if Y.ndim == 2 and Y.shape[0] == n_y:
                y_um = Y[:, 0] * _UM_PER_M

    if x_um is None or x_um.size != n_x:
        x_um = _centered_coords_um(n_x, dx_um)
        metadata["reconstructed_x_from_spacing"] = True
    if y_um is None or y_um.size != n_y:
        y_um = _centered_coords_um(n_y, dy_um)
        metadata["reconstructed_y_from_spacing"] = True

    return x_um, y_um, metadata


def _describe_keys(obj: Any) -> str:
    if _is_mapping(obj):
        return ", ".join(sorted(str(k) for k in obj.keys())) or "<empty mapping>"
    names = [n for n in dir(obj) if not n.startswith("_")]
    return ", ".join(names) or "<no public attributes>"


# ---------------------------------------------------------------------------
# Extractors from real repository field objects
# ---------------------------------------------------------------------------


def extract_plane_from_surfacefield(
    surface_field: Any,
    *,
    preferred_z_um: float | None = None,
    source_status: str = SOURCE_REAL_OPTICAL_FIELD,
) -> OpticalFieldPlane:
    """Adapt a real repository field object to a canonical :class:`OpticalFieldPlane`.

    Handles ``vbb_study.vbb_studies.SurfaceField`` (Ex/Ey/Ez + grid dict in metres)
    and other objects exposing intensity + sampling.  Raises
    :class:`MissingOpticalFieldError` if ``surface_field`` is None and
    :class:`UnsupportedSurfaceFieldError` if it cannot be adapted.
    """
    if surface_field is None:
        raise MissingOpticalFieldError(
            "No optical field provided to extract_plane_from_surfacefield(). "
            "A real field is required; refusing to fabricate a placeholder beam."
        )

    intensity, components_used = _intensity_2d_from_object(surface_field)
    intensity = validate_intensity_plane(intensity, name="surface_field intensity")
    ny, nx = intensity.shape
    dx_um, dy_um, z_um = _sampling_um_from_object(surface_field, ny, nx)
    x_um, y_um, coord_metadata = _xy_coords_um_from_object(surface_field, ny, nx, dx_um, dy_um)
    if preferred_z_um is not None:
        z_um = float(preferred_z_um)

    label = str(_get(surface_field, "field_label", None) or type(surface_field).__name__)
    src_metadata = _get(surface_field, "metadata", None)
    metadata: dict[str, Any] = {
        "adapter": "extract_plane_from_surfacefield",
        "intensity_components": components_used,
        "origin_type": type(surface_field).__name__,
    }
    if isinstance(src_metadata, Mapping):
        metadata["origin_metadata"] = dict(src_metadata)
    if _has(surface_field, "medium_before"):
        metadata["medium_before"] = _get(surface_field, "medium_before")
    metadata.update(coord_metadata)

    return OpticalFieldPlane(
        intensity=intensity,
        dx_um=dx_um,
        dy_um=dy_um,
        z_um=z_um,
        field_label=label,
        source_status=str(source_status),
        x_um=x_um,
        y_um=y_um,
        metadata=metadata,
    )


def extract_stack_from_surfacefield(
    surface_field: Any,
    *,
    source_status: str = SOURCE_REAL_OPTICAL_FIELD,
) -> OpticalFieldStack:
    """Adapt a real repository volume/stack output to a canonical :class:`OpticalFieldStack`.

    Handles:
      - the ``propagate_volume`` result dict
        (``intensity_stack`` [z, y, x] + ``z`` [m] + ``crop_grid`` with 'x' [m]),
      - a ``result`` dict that nests such a volume under ``volume``,
      - objects already exposing ``intensity_zyx`` + ``x_um``/``y_um``/``z_um``.

    Note: the repository ``SurfaceField`` is a *single plane* and has no stack; in
    that case use :func:`extract_plane_from_surfacefield`.  Passing one here raises
    :class:`UnsupportedSurfaceFieldError`.
    """
    if surface_field is None:
        raise MissingOpticalFieldError(
            "No optical field provided to extract_stack_from_surfacefield(). "
            "A real field is required; refusing to fabricate a placeholder stack."
        )

    obj = surface_field
    # A result dict may nest the volume.
    if _is_mapping(obj) and "intensity_stack" not in obj and "volume" in obj:
        obj = obj["volume"]

    # Path 1: already-canonical stack object/dict.
    if _has(obj, "intensity_zyx") and _get(obj, "intensity_zyx") is not None:
        return stack_from_arrays(
            intensity_zyx=_get(obj, "intensity_zyx"),
            x_um=_get(obj, "x_um"),
            y_um=_get(obj, "y_um"),
            z_um=_get(obj, "z_um"),
            field_label=str(_get(obj, "field_label", "array_stack")),
            source_status=str(source_status),
            metadata=dict(_get(obj, "metadata", {}) or {}),
        )

    # Path 2: repository propagate_volume dict.
    if _has(obj, "intensity_stack") and _get(obj, "intensity_stack") is not None:
        stack = validate_intensity_stack(
            np.asarray(_get(obj, "intensity_stack"), dtype=float), name="intensity_stack"
        )
        nz, ny, nx = stack.shape
        z_m = _get(obj, "z")
        if z_m is None:
            raise UnsupportedSurfaceFieldError(
                "Volume dict has 'intensity_stack' but no 'z' axis."
            )
        z_um = np.asarray(z_m, dtype=float) * _UM_PER_M

        crop_grid = _get(obj, "crop_grid")
        coord_metadata: dict[str, Any] = {}
        if crop_grid is not None and (
            _get(crop_grid, "x") is not None or _get(crop_grid, "dx") is not None
        ):
            dx_um = None
            dy_um = None
            x_um = None
            y_um = None

            if _get(crop_grid, "x") is not None:
                x_um = np.asarray(_get(crop_grid, "x"), dtype=float) * _UM_PER_M
                if x_um.size >= 2:
                    dx_um = _uniform_spacing(x_um, "crop_grid['x']")
            if _get(crop_grid, "y") is not None:
                y_um = np.asarray(_get(crop_grid, "y"), dtype=float) * _UM_PER_M
                if y_um.size >= 2:
                    dy_um = _uniform_spacing(y_um, "crop_grid['y']")
            elif x_um is not None:
                y_um = x_um.copy()
                dy_um = dx_um
                coord_metadata["assumed_y_equals_x"] = True

            if dx_um is None and _get(crop_grid, "dx") is not None:
                dx_um = float(_get(crop_grid, "dx")) * _UM_PER_M
            if dy_um is None:
                dy_um = dx_um

            if dx_um is None or dy_um is None:
                raise UnsupportedSurfaceFieldError(
                    "Volume crop_grid has neither usable coordinates nor spacing."
                )
            if x_um is None or x_um.size != nx:
                x_um = _centered_coords_um(nx, dx_um)
                coord_metadata["reconstructed_x_from_spacing"] = True
            if y_um is None or y_um.size != ny:
                y_um = _centered_coords_um(ny, dy_um)
                coord_metadata["reconstructed_y_from_spacing"] = True
        else:
            raise UnsupportedSurfaceFieldError(
                "Volume dict has 'intensity_stack' but no 'crop_grid' with an 'x' axis or 'dx'; "
                "cannot determine transverse sampling."
            )

        metadata = {
            "adapter": "extract_stack_from_surfacefield",
            "origin_type": type(surface_field).__name__,
            "propagation_method": _get(obj, "propagation_method", None),
            "peak_index": _get(obj, "peak_index", None),
            **coord_metadata,
        }
        return stack_from_arrays(
            intensity_zyx=stack,
            x_um=x_um,
            y_um=y_um,
            z_um=z_um,
            field_label="propagate_volume_intensity_stack",
            source_status=str(source_status),
            metadata=metadata,
        )

    raise UnsupportedSurfaceFieldError(
        "Object is not a recognised optical stack. Expected a propagate_volume dict "
        "(intensity_stack + z + crop_grid) or a canonical stack (intensity_zyx + "
        f"x_um/y_um/z_um). Available: {_describe_keys(obj)}. "
        "If this is a single-plane SurfaceField, use extract_plane_from_surfacefield()."
    )
