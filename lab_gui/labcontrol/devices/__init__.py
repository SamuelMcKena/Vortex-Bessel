"""Hardware-provider contracts and safe development implementations."""

from .camera import (
    BeamageCameraProvider,
    CameraFrame,
    CameraProvider,
    DummyCameraProvider,
    ReplayCameraProvider,
)
from .beamage_pipe import BeamageIdentity, BeamagePipeClient, PIPE_PATH
from .slm import (
    BackendSlmProvider,
    DummySlmProvider,
    HedsSlmProvider,
    SlmProvider,
    slm_provider_from_config,
)
from .stage import DummyStageProvider, ManualStageProvider, StageProvider

__all__ = [
    "BackendSlmProvider",
    "DummySlmProvider",
    "HedsSlmProvider",
    "BeamageCameraProvider",
    "CameraFrame",
    "CameraProvider",
    "DummyCameraProvider",
    "DummyStageProvider",
    "ManualStageProvider",
    "ReplayCameraProvider",
    "SlmProvider",
    "slm_provider_from_config",
    "StageProvider",
]
