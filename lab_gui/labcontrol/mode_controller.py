"""Mode-aware controller for live, recorded and virtual operation.

The legacy/unified LabController remains untouched.  This subclass adds a hard
provider boundary so Virtual Lab cannot accidentally open HEDS and Recorded Lab
cannot pretend historical data can respond to new hardware commands.
"""

from __future__ import annotations

from typing import Iterable

from .controller import CastReceipt, LabController
from .devices.camera import BeamageCameraProvider, CameraFrame, ReplayCameraProvider
from .devices.stage import ManualStageProvider
from .phase_service import PhaseBundle
from .state import AcquisitionState, ConnectionState, DataKind, utc_now
from .virtual_lab import (
    DataOrigin,
    OperatingMode,
    RecordedReadOnlySlmProvider,
    VirtualBenchEngine,
    VirtualCameraProvider,
    VirtualSlmProvider,
    VirtualStageProvider,
)


class ModeAwareLabController(LabController):
    """LabController with explicit LIVE / RECORDED / VIRTUAL safety modes."""

    def __init__(self, *args, virtual_engine: VirtualBenchEngine | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.operating_mode = OperatingMode.LIVE_LAB
        self.data_origin = (
            DataOrigin.SIMULATED
            if getattr(self.camera_provider, "name", "") == "dummy"
            else DataOrigin.LIVE_MEASURED
        )
        self.virtual_engine = virtual_engine or VirtualBenchEngine()
        self.stage_provider = ManualStageProvider()

    def _clear_cast_target_state(self, *, reason: str) -> None:
        def clear(state):
            for target in (state.slm1, state.slm2):
                target.connection = ConnectionState.DISCONNECTED
                target.last_cast_sha256 = None
                target.last_cast_utc = None
                target.last_hardware_command = reason
                target.phase_mode_verified = False
            state.camera.last_frame_id = None
            state.camera.last_frame_utc = None

        self.store.update(clear, source="operating_mode", reason=reason)

    def _close_current_providers(self) -> None:
        try:
            if self.camera_provider is not None:
                try:
                    self.camera_provider.stop()
                except Exception:
                    pass
                try:
                    self.camera_provider.disconnect()
                except Exception:
                    pass
        finally:
            if self.slm_provider is not None:
                try:
                    self.slm_provider.close()
                except Exception:
                    pass
        self.camera_provider = None
        self.slm_provider = None
        self._provider_signature = None

    def set_operating_mode(
        self,
        mode: OperatingMode | str,
        *,
        replay_provider: ReplayCameraProvider | None = None,
    ) -> str:
        new_mode = mode if isinstance(mode, OperatingMode) else OperatingMode(str(mode))
        if new_mode == self.operating_mode:
            provider_name = getattr(self.camera_provider, "name", None)
            provider_ok = (
                (new_mode is OperatingMode.LIVE_LAB and provider_name == "beamage")
                or (new_mode is OperatingMode.VIRTUAL_LAB and provider_name == "virtual")
                or (new_mode is OperatingMode.RECORDED_LAB and provider_name == "replay")
            )
            if new_mode is OperatingMode.RECORDED_LAB and replay_provider is not None:
                self.set_camera_provider(replay_provider, provider_name="replay")
                self.data_origin = DataOrigin.RECORDED_MEASURED
                return "RECORDED LAB replay source updated."
            if provider_ok:
                return f"Operating mode already {new_mode.value}."

        self._close_current_providers()
        self._clear_cast_target_state(
            reason=f"Cast target invalidated while switching to {new_mode.value}; explicit recast required."
        )
        self.operating_mode = new_mode

        if new_mode is OperatingMode.VIRTUAL_LAB:
            self.data_origin = DataOrigin.SIMULATED
            self.stage_provider = VirtualStageProvider()
            self.stage_provider.connect()
            self.slm_provider = VirtualSlmProvider(
                self.virtual_engine,
                self.store.snapshot,
                phase_service=self.phase_service,
            )
            self.set_camera_provider(
                VirtualCameraProvider(self.virtual_engine, self.store.snapshot),
                provider_name="virtual",
            )
            self.store.update(
                lambda state: (
                    setattr(state.camera, "pixel_size_um", self.virtual_engine.pixel_size_um),
                    setattr(state.camera, "shape_yx", self.virtual_engine.shape_yx),
                    setattr(state.camera, "z_reference", self.virtual_engine.geometry.camera_z_reference),
                    setattr(state.camera, "frame_quality", "SIMULATED_NUMERICAL_INTENSITY"),
                ),
                source="operating_mode",
                reason="Virtual Lab providers installed",
            )
            return (
                "VIRTUAL LAB active — camera, z stage and SLM cast target are simulated. "
                "Physical HEDS is not opened."
            )

        if new_mode is OperatingMode.RECORDED_LAB:
            self.data_origin = DataOrigin.RECORDED_MEASURED
            self.stage_provider = VirtualStageProvider()
            self.stage_provider.connect()
            self.slm_provider = RecordedReadOnlySlmProvider()
            if replay_provider is not None:
                self.set_camera_provider(replay_provider, provider_name="replay")
            else:
                self.store.update(
                    lambda state: (
                        setattr(state.camera, "provider", "replay"),
                        setattr(state.camera, "implementation_status", "REPLAY_AWAITING_FILES"),
                        setattr(state.camera, "connection", ConnectionState.DISCONNECTED),
                        setattr(state.camera, "acquisition", AcquisitionState.STOPPED),
                        setattr(state.system, "data_kind", DataKind.REPLAY),
                    ),
                    source="operating_mode",
                    reason="Recorded Lab selected; choose recorded numerical frames",
                )
            return "RECORDED LAB active — historical measurements are read-only; choose replay frames."

        # LIVE LAB is intentionally disconnected after a mode transition.  A user
        # must explicitly reconnect/cast; a simulated correction is never pushed
        # onto physical hardware by changing this selector.
        self.data_origin = DataOrigin.LIVE_MEASURED
        self.stage_provider = ManualStageProvider()
        self.slm_provider = None
        self.set_camera_provider(BeamageCameraProvider(), provider_name="beamage")
        return (
            "LIVE LAB selected — physical providers are disconnected. "
            "Explicitly connect Beamage/HEDS before use."
        )

    def _ensure_slm_provider(self):
        if self.operating_mode is OperatingMode.VIRTUAL_LAB:
            if not isinstance(self.slm_provider, VirtualSlmProvider):
                self.slm_provider = VirtualSlmProvider(
                    self.virtual_engine,
                    self.store.snapshot,
                    phase_service=self.phase_service,
                )
            return self.slm_provider
        if self.operating_mode is OperatingMode.RECORDED_LAB:
            if not isinstance(self.slm_provider, RecordedReadOnlySlmProvider):
                self.slm_provider = RecordedReadOnlySlmProvider()
            return self.slm_provider
        return super()._ensure_slm_provider()

    def set_replay_provider(self, provider: ReplayCameraProvider) -> None:
        if self.operating_mode is not OperatingMode.RECORDED_LAB:
            self.set_operating_mode(OperatingMode.RECORDED_LAB)
        self.data_origin = DataOrigin.RECORDED_MEASURED
        self.set_camera_provider(provider, provider_name="replay")

    def acquire_frame(self, *, fresh: bool = True, timeout_s: float = 2.0) -> CameraFrame:
        frame = super().acquire_frame(fresh=fresh, timeout_s=timeout_s)
        frame.metadata.setdefault("operating_mode", self.operating_mode.value)
        frame.metadata.setdefault("data_origin", self.data_origin.value)
        frame.metadata.setdefault("provenance_classification", self.data_origin.value)
        return frame

    def move_stage(self, z_mm: float) -> float:
        request = self.stage_provider.request_move(float(z_mm))
        if request.requires_operator_confirmation:
            raise RuntimeError(request.prompt)
        actual = self.stage_provider.confirm_position(float(z_mm))
        self.store.set_camera_z(actual, source="stage_provider")
        return actual

    def cast(self, names: Iterable[str], *, persist: bool = True) -> CastReceipt:
        if persist or self.operating_mode is not OperatingMode.VIRTUAL_LAB:
            return super().cast(names)

        selected_names = tuple(dict.fromkeys(str(name).upper() for name in names))
        if not selected_names or any(name not in {"SLM1", "SLM2"} for name in selected_names):
            raise ValueError("Select SLM1 and/or SLM2 for casting.")

        # Fast virtual optimisation cast: update the same cast hashes and provider
        # state as a normal cast, but do not write hundreds of redundant full-size
        # PNG bundles during modal sweeps.
        with self._lock:
            bundle: PhaseBundle = self.generate()
            provider = self._ensure_slm_provider()
            if any(provider.status(name).connection != ConnectionState.CONNECTED for name in selected_names):
                self.connect_slms()
            folder = self.project_root / "outputs" / "virtual_lab" / "ephemeral_casts"
            folder.mkdir(parents=True, exist_ok=True)
            config = self.app_config()
            messages = tuple(
                provider.cast(
                    name,
                    bundle.results[name],
                    config.transfer_mode,
                    folder / f"{name.lower()}_virtual_not_written.png",
                )
                for name in selected_names
            )
            stamp = utc_now()

            def record(state) -> None:
                for name, message in zip(selected_names, messages):
                    target = getattr(state, name.lower())
                    target.last_cast_sha256 = bundle.hashes[name]
                    target.last_cast_utc = stamp
                    target.last_hardware_command = message
                    target.connection = provider.status(name).connection
                    target.phase_mode_verified = False

            self.store.update(
                record,
                source="virtual_slm_provider",
                reason=f"Virtual cast complete for {', '.join(selected_names)}",
            )
            return CastReceipt(folder, dict(bundle.hashes), messages, config.transfer_mode)
