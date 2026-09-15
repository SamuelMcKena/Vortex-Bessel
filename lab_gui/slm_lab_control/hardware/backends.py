from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from ..phase import save_phase_png, TWOPI


class BackendError(RuntimeError):
    pass


@dataclass
class SlmDeviceState:
    name: str
    serial: str
    connected: bool = False
    wavelength_nm: float | None = None
    phase_mode_verified: bool = False
    last_path: str = ""
    last_transfer: str = ""


class BaseSlmBackend:
    """Abstract-ish SLM backend used by the GUI.

    The backend deliberately supports three transfer concepts:
      1. PNG file transfer: save uint8 phase mask, then show the file.
      2. Direct gray-array transfer: send the uint8 pixel buffer to the SDK.
      3. Direct phase-array transfer: send the floating phase map to the SDK if supported.

    Dummy mode never touches hardware; it saves the files that would have been used.
    """

    name = "base"

    def __init__(self, output_root: Path, sdk_major: int = 4, sdk_minor: int = 2):
        self.output_root = Path(output_root)
        self.sdk_major = sdk_major
        self.sdk_minor = sdk_minor
        self.devices: Dict[str, SlmDeviceState] = {}

    def init_sdk(self) -> str:
        return "SDK not implemented for base backend."

    def connect(self, name: str, serial: str, wavelength_nm: float | None = None) -> str:
        self.devices[name] = SlmDeviceState(
            name=name, serial=serial, connected=True, wavelength_nm=wavelength_nm,
            phase_mode_verified=False,
        )
        suffix = f", requested wavelength={wavelength_nm:g} nm" if wavelength_nm is not None else ""
        return f"{name} connected ({serial}{suffix})."

    def show_png(self, name: str, png_path: Path) -> str:
        state = self._state(name)
        state.last_path = str(png_path)
        state.last_transfer = "png_file"
        return f"{name}: show PNG {png_path}"

    def show_phase_file(self, name: str, png_path: Path) -> str:
        """Show a phase-valued image file when the backend supports that distinction.

        Base/dummy implementations preserve the file only. Hardware backends may
        interpret integer 0..255 as phase 0..2*pi rather than as generic image data.
        """
        return self.show_png(name, png_path)

    def show_gray_array(
        self,
        name: str,
        gray_uint8: np.ndarray,
        save_path: Path,
        allow_png_fallback: bool = True,
    ) -> str:
        # Base implementation is safe fallback: persist exact array as PNG, then use PNG route.
        save_phase_png(save_path, np.ascontiguousarray(gray_uint8, dtype=np.uint8))
        if not allow_png_fallback:
            raise BackendError("Direct gray-array transfer is not implemented by this backend.")
        return self.show_png(name, save_path)

    def show_phase_array(
        self,
        name: str,
        phase_rad: np.ndarray,
        gray_uint8: np.ndarray,
        save_path: Path,
        allow_png_fallback: bool = True,
    ) -> str:
        # Base implementation persists the equivalent wrapped gray PNG and falls back.
        save_phase_png(save_path, np.ascontiguousarray(gray_uint8, dtype=np.uint8))
        if not allow_png_fallback:
            raise BackendError("Direct phase-array transfer is not implemented by this backend.")
        return self.show_png(name, save_path)

    # Backwards-compatible alias used by old v0.1 code paths.
    def show_array(
        self,
        name: str,
        gray_uint8: np.ndarray,
        save_path: Path,
        allow_png_fallback: bool = True,
    ) -> str:
        return self.show_gray_array(name, gray_uint8, save_path, allow_png_fallback=allow_png_fallback)

    def blank(
        self,
        name: str,
        save_path: Path,
        shape=(1080, 1920),
        gray: int = 0,
        use_direct: bool = False,
        allow_png_fallback: bool = True,
    ) -> str:
        arr = np.full(shape, int(gray), dtype=np.uint8)
        if use_direct:
            return self.show_gray_array(name, arr, save_path, allow_png_fallback=allow_png_fallback)
        save_phase_png(save_path, arr)
        return self.show_png(name, save_path)

    def close(self) -> str:
        for state in self.devices.values():
            state.connected = False
        return "Backend closed."

    def _state(self, name: str) -> SlmDeviceState:
        if name not in self.devices or not self.devices[name].connected:
            raise BackendError(f"{name} is not connected.")
        return self.devices[name]


class DummySlmBackend(BaseSlmBackend):
    """Safe backend for development without real SLMs."""

    name = "dummy"

    def init_sdk(self) -> str:
        self.output_root.mkdir(parents=True, exist_ok=True)
        return "Dummy backend ready. No hardware calls will be made."

    def connect(self, name: str, serial: str, wavelength_nm: float | None = None) -> str:
        self.devices[name] = SlmDeviceState(
            name=name, serial=serial or "DUMMY", connected=True, wavelength_nm=wavelength_nm,
            phase_mode_verified=True,
        )
        suffix = f", phase wavelength={wavelength_nm:g} nm" if wavelength_nm is not None else ""
        return f"{name} connected in dummy mode ({serial or 'DUMMY'}{suffix})."

    def show_png(self, name: str, png_path: Path) -> str:
        state = self._state(name)
        state.last_path = str(png_path)
        state.last_transfer = "png_file"
        return f"{name}: dummy PNG transfer registered -> {png_path}"

    def show_gray_array(
        self,
        name: str,
        gray_uint8: np.ndarray,
        save_path: Path,
        allow_png_fallback: bool = True,
    ) -> str:
        self._state(name)
        save_phase_png(save_path, np.ascontiguousarray(gray_uint8, dtype=np.uint8))
        self.devices[name].last_path = str(save_path)
        self.devices[name].last_transfer = "direct_gray_array_dummy"
        return f"{name}: dummy direct gray-array transfer saved -> {save_path}"

    def show_phase_array(
        self,
        name: str,
        phase_rad: np.ndarray,
        gray_uint8: np.ndarray,
        save_path: Path,
        allow_png_fallback: bool = True,
    ) -> str:
        self._state(name)
        phase_path = Path(save_path).with_suffix(".phase_rad.npy")
        phase_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(phase_path, np.ascontiguousarray(np.mod(phase_rad, TWOPI), dtype=np.float32))
        save_phase_png(save_path, np.ascontiguousarray(gray_uint8, dtype=np.uint8))
        self.devices[name].last_path = str(phase_path)
        self.devices[name].last_transfer = "direct_phase_array_dummy"
        return f"{name}: dummy direct phase-array transfer saved -> {phase_path} plus {save_path}"


class HedsSlmBackend(BaseSlmBackend):
    """HOLOEYE HEDS SDK backend.

    Direct-array support is version/API dependent. The public HOLOEYE material says the
    SLM Display SDK can show images and phase arrays directly, but exact Python method
    names differ between SDK generations/examples. This backend therefore uses robust
    introspection and tries likely method names. The log will state which method worked.
    If strict direct mode is chosen and none works, it raises a clear error. In auto mode,
    it safely falls back to the proven PNG route.
    """

    name = "heds"

    # Candidate names intentionally broad; the lab PC can reveal the exact installed API.
    GRAY_ARRAY_METHODS: Tuple[str, ...] = (
        "showImageData",
        "showImageDataFromArray",
        "showImageDataFromNumpyArray",
        "showData",
        "showDataArray",
        "showGrayData",
        "showImage",
    )
    PHASE_ARRAY_METHODS: Tuple[str, ...] = (
        "showPhaseValues",
        "showPhaseData",
        "showPhaseArray",
        "showPhaseDataFromArray",
        "showDataFromPhaseValues",
        "showPhaseValuesData",
    )

    def __init__(self, output_root: Path, sdk_major: int = 4, sdk_minor: int = 2):
        super().__init__(output_root, sdk_major=sdk_major, sdk_minor=sdk_minor)
        self.HEDS = None
        self.HEDSERR_NoError = None
        self._sdk_ready = False
        self._slm_objects: Dict[str, object] = {}

    def init_sdk(self) -> str:
        try:
            import HEDS  # type: ignore
            from hedslib.heds_types import HEDSERR_NoError  # type: ignore
        except Exception as exc:  # pragma: no cover - only hit on lab machine
            raise BackendError(
                "Could not import HOLOEYE HEDS SDK. Use the same Anaconda/Spyder environment "
                "that successfully imports HEDS, or add the SDK folder to PYTHONPATH."
            ) from exc

        self.HEDS = HEDS
        self.HEDSERR_NoError = HEDSERR_NoError
        HEDS.SDK.PrintVersion()
        err = HEDS.SDK.Init(int(self.sdk_major), int(self.sdk_minor))
        if err != HEDSERR_NoError:
            raise BackendError(HEDS.SDK.ErrorString(err))
        self._sdk_ready = True
        return f"HEDS SDK initialised ({self.sdk_major}.{self.sdk_minor})."

    def connect(self, name: str, serial: str, wavelength_nm: float | None = None) -> str:
        if not self._sdk_ready:
            self.init_sdk()
        assert self.HEDS is not None and self.HEDSERR_NoError is not None
        arg = f"-slm serial:{serial}"
        slm = self.HEDS.SLM.Init(arg, True, 0.0)
        if slm.errorCode() != self.HEDSERR_NoError:
            raise BackendError(self.HEDS.SDK.ErrorString(slm.errorCode()))

        # HEDS 4.2 explicitly documents setWavelength() as:
        #   "Set the SLM into phase modulation and apply a monochrome wavelength".
        # Do this before any phase-valued transfer and verify the SDK readback.
        verified = False
        readback = None
        if wavelength_nm is not None:
            if not hasattr(slm, "setWavelength") or not hasattr(slm, "getWavelength"):
                raise BackendError(
                    "Installed HEDS binding does not expose setWavelength/getWavelength; "
                    "refusing to claim calibrated phase mode."
                )
            err = slm.setWavelength(float(wavelength_nm))
            if err != self.HEDSERR_NoError:
                raise BackendError(self.HEDS.SDK.ErrorString(err))
            readback = float(slm.getWavelength())
            if not np.isfinite(readback) or abs(readback - float(wavelength_nm)) > 0.5:
                raise BackendError(
                    f"{name}: HEDS wavelength readback {readback!r} nm does not match "
                    f"requested {float(wavelength_nm):.3f} nm."
                )
            verified = True

        self._slm_objects[name] = slm
        self.devices[name] = SlmDeviceState(
            name=name, serial=serial, connected=True,
            wavelength_nm=readback if readback is not None else wavelength_nm,
            phase_mode_verified=verified,
        )
        method_hint = self.available_direct_methods(name)
        phase_note = (
            f"Phase mode verified at {readback:.3f} nm. " if verified else
            "Phase wavelength was not configured. "
        )
        return f"{name} connected via HEDS ({serial}). {phase_note}{method_hint}"

    def available_direct_methods(self, name: str) -> str:
        slm = self._slm_objects.get(name)
        if slm is None:
            return "Direct API not inspected yet."
        gray = [m for m in self.GRAY_ARRAY_METHODS if hasattr(slm, m)]
        phase = [m for m in self.PHASE_ARRAY_METHODS if hasattr(slm, m)]
        return f"Detected direct methods: gray={gray or 'none'}, phase={phase or 'none'}"

    def _call_direct_method(
        self,
        slm: object,
        method_names: Iterable[str],
        arrays: Iterable[np.ndarray],
    ) -> Tuple[str, object]:
        errors: List[str] = []
        for method_name in method_names:
            if not hasattr(slm, method_name):
                continue
            method = getattr(slm, method_name)
            for arr in arrays:
                try:
                    result = method(arr)
                    return method_name, result
                except TypeError as exc:
                    errors.append(f"{method_name}({arr.dtype}, {arr.shape}) TypeError: {exc}")
                except Exception as exc:
                    errors.append(f"{method_name}({arr.dtype}, {arr.shape}) failed: {exc}")
        detail = "\n".join(errors[-8:]) if errors else "No candidate direct methods found on this SLM object."
        raise BackendError(detail)

    def _check_heds_err(self, value: object) -> None:
        # Some SDK calls return an error code. Others may return a data handle or None.
        if value is None or self.HEDSERR_NoError is None:
            return
        if isinstance(value, int) and value != self.HEDSERR_NoError:
            assert self.HEDS is not None
            raise BackendError(self.HEDS.SDK.ErrorString(value))

    def show_png(self, name: str, png_path: Path) -> str:
        self._state(name)
        assert self.HEDS is not None and self.HEDSERR_NoError is not None
        if name not in self._slm_objects:
            raise BackendError(f"{name} has no HEDS object.")
        slm = self._slm_objects[name]
        err = slm.showImageDataFromFile(str(Path(png_path).resolve()))
        if err != self.HEDSERR_NoError:
            # ErrorString(err) is usually more useful than slm.errorCode() here.
            raise BackendError(self.HEDS.SDK.ErrorString(err))
        self.devices[name].last_path = str(png_path)
        self.devices[name].last_transfer = "png_file"
        return f"{name}: PNG-file transfer via showImageDataFromFile -> {png_path}"

    def show_phase_file(self, name: str, png_path: Path) -> str:
        """Use HEDS' explicit phase-file route without changing legacy PNG behavior.

        HEDS 4.2 documents showPhaseDataFromFile() as converting integer image
        values 0..255 to phase values spanning a 2*pi phase unit.
        """
        state = self._state(name)
        assert self.HEDS is not None and self.HEDSERR_NoError is not None
        if not state.phase_mode_verified:
            raise BackendError(f"{name}: phase wavelength/mode has not been verified.")
        if name not in self._slm_objects:
            raise BackendError(f"{name} has no HEDS object.")
        slm = self._slm_objects[name]
        if not hasattr(slm, "showPhaseDataFromFile"):
            raise BackendError("Installed HEDS binding has no showPhaseDataFromFile().")
        err = slm.showPhaseDataFromFile(str(Path(png_path).resolve()))
        if err != self.HEDSERR_NoError:
            raise BackendError(self.HEDS.SDK.ErrorString(err))
        state.last_path = str(png_path)
        state.last_transfer = "phase_file:showPhaseDataFromFile"
        return f"{name}: PHASE-file transfer via showPhaseDataFromFile -> {png_path}"

    def show_gray_array(
        self,
        name: str,
        gray_uint8: np.ndarray,
        save_path: Path,
        allow_png_fallback: bool = True,
    ) -> str:
        self._state(name)
        save_phase_png(save_path, np.ascontiguousarray(gray_uint8, dtype=np.uint8))
        if name not in self._slm_objects:
            raise BackendError(f"{name} has no HEDS object.")
        slm = self._slm_objects[name]
        arr_u8 = np.ascontiguousarray(gray_uint8, dtype=np.uint8)
        # Some bindings expect uint8, others accept float64 0..255; try both.
        arrays = (arr_u8, np.ascontiguousarray(arr_u8.astype(np.float64)))
        try:
            method_name, result = self._call_direct_method(slm, self.GRAY_ARRAY_METHODS, arrays)
            self._check_heds_err(result)
            self.devices[name].last_path = str(save_path)
            self.devices[name].last_transfer = f"direct_gray_array:{method_name}"
            return f"{name}: direct gray-array transfer via {method_name}; logged PNG -> {save_path}"
        except BackendError as exc:
            if allow_png_fallback:
                msg = self.show_png(name, save_path)
                return f"{name}: direct gray-array unavailable; fell back. Reason: {exc}\n{msg}"
            raise BackendError(
                f"Direct gray-array transfer failed. Use auto/png mode, or check the installed HEDS method name.\n{exc}"
            ) from exc

    def show_phase_array(
        self,
        name: str,
        phase_rad: np.ndarray,
        gray_uint8: np.ndarray,
        save_path: Path,
        allow_png_fallback: bool = True,
    ) -> str:
        """Send the final composed phase explicitly as radians through HEDS 4.2.

        The uploaded HEDS wrapper defines showPhaseData(data, flags=None,
        phase_unit=2*pi). We therefore do not guess method names or phase units
        here. The array is wrapped once, passed in radians, and HEDS is told the
        phase unit explicitly. Legacy image/PNG transfer remains available through
        the separate transfer modes and is not silently substituted in strict mode.
        """
        state = self._state(name)
        save_phase_png(save_path, np.ascontiguousarray(gray_uint8, dtype=np.uint8))
        phase_path = Path(save_path).with_suffix(".phase_rad.npy")
        phase_path.parent.mkdir(parents=True, exist_ok=True)
        phase_wrapped = np.ascontiguousarray(np.mod(phase_rad, TWOPI), dtype=np.float32)
        np.save(phase_path, phase_wrapped)

        if not state.phase_mode_verified:
            raise BackendError(
                f"{name}: refusing direct phase transfer because HEDS phase wavelength/mode "
                "was not verified with setWavelength()/getWavelength()."
            )
        if name not in self._slm_objects:
            raise BackendError(f"{name} has no HEDS object.")
        slm = self._slm_objects[name]
        if not hasattr(slm, "showPhaseData"):
            raise BackendError("Installed HEDS binding has no showPhaseData().")

        # Exact HEDS 4.2 API: phase values are in radians and phase_unit=2*pi.
        # Retry float64 only for binding compatibility; never reinterpret as waves.
        errors = []
        for arr in (phase_wrapped, np.ascontiguousarray(phase_wrapped, dtype=np.float64)):
            try:
                result = slm.showPhaseData(arr, flags=None, phase_unit=TWOPI)
                self._check_heds_err(result)
                state.last_path = str(phase_path)
                state.last_transfer = "direct_phase_array:showPhaseData:radians"
                return (
                    f"{name}: direct PHASE transfer via showPhaseData; "
                    f"wavelength={state.wavelength_nm:.3f} nm, phase_unit=2*pi rad; "
                    f"logged phase -> {phase_path}, PNG -> {save_path}"
                )
            except TypeError as exc:
                errors.append(f"{arr.dtype}: {exc}")
            except Exception as exc:
                errors.append(f"{arr.dtype}: {exc}")

        detail = "; ".join(errors) or "unknown showPhaseData failure"
        if allow_png_fallback:
            # Auto mode may fall back, but it is loud and uses the explicit PHASE-file
            # path rather than generic image data. This preserves phase semantics.
            try:
                fallback = self.show_phase_file(name, save_path)
                return f"{name}: direct showPhaseData failed ({detail}); {fallback}"
            except Exception as exc:
                raise BackendError(
                    f"Direct HEDS phase transfer failed ({detail}); phase-file fallback also failed: {exc}"
                ) from exc
        raise BackendError(f"Direct HEDS showPhaseData failed: {detail}")

    def close(self) -> str:
        messages = []
        for name, slm in list(self._slm_objects.items()):
            try:
                err = slm.window().close()
                messages.append(f"{name}: close err={err}")
            except Exception as exc:
                messages.append(f"{name}: close failed: {exc}")
        self._slm_objects.clear()
        for state in self.devices.values():
            state.connected = False
        return "; ".join(messages) or "HEDS backend closed."


def make_backend(kind: str, output_root: Path, sdk_major: int = 4, sdk_minor: int = 2) -> BaseSlmBackend:
    if kind == "heds":
        return HedsSlmBackend(output_root, sdk_major=sdk_major, sdk_minor=sdk_minor)
    return DummySlmBackend(output_root, sdk_major=sdk_major, sdk_minor=sdk_minor)
