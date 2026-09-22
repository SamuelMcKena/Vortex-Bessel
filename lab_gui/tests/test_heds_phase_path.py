from pathlib import Path

import numpy as np

from slm_lab_control.hardware.backends import HedsSlmBackend
from slm_lab_control.phase import TWOPI


class FakeSDK:
    @staticmethod
    def ErrorString(err):
        return f"ERR:{err}"


class FakeSlm:
    def __init__(self):
        self._wl = 0.0
        self.phase_calls = []
        self.phase_file_calls = []

    def errorCode(self):
        return 0

    def setWavelength(self, value):
        self._wl = float(value)
        return 0

    def getWavelength(self):
        return self._wl

    def showPhaseData(self, data, flags=None, phase_unit=TWOPI):
        self.phase_calls.append((np.array(data, copy=True), flags, float(phase_unit)))
        return 0

    def showPhaseDataFromFile(self, filename):
        self.phase_file_calls.append(str(filename))
        return 0


class FakeSLMFactory:
    def __init__(self, slm):
        self._slm = slm
        self.calls = []

    def Init(self, arg, open_preview, scale):
        self.calls.append((arg, open_preview, scale))
        return self._slm


class FakeHEDS:
    def __init__(self, slm):
        self.SDK = FakeSDK()
        self.SLM = FakeSLMFactory(slm)


def make_backend(tmp_path):
    slm = FakeSlm()
    backend = HedsSlmBackend(tmp_path)
    backend.HEDS = FakeHEDS(slm)
    backend.HEDSERR_NoError = 0
    backend._sdk_ready = True
    return backend, slm


def test_connect_sets_and_verifies_1030_nm_phase_mode(tmp_path):
    backend, slm = make_backend(tmp_path)
    msg = backend.connect("SLM2", "6010-2381", 1030.0)
    assert slm.getWavelength() == 1030.0
    assert backend.devices["SLM2"].phase_mode_verified is True
    assert backend.devices["SLM2"].wavelength_nm == 1030.0
    assert "Phase mode verified at 1030.000 nm" in msg


def test_direct_phase_uses_exact_heds_show_phase_data_in_radians(tmp_path):
    backend, slm = make_backend(tmp_path)
    backend.connect("SLM2", "6010-2381", 1030.0)
    phase = np.array([[-0.25, 0.0, np.pi], [2*np.pi + 0.5, 7.0, -7.0]], dtype=float)
    gray = np.zeros(phase.shape, dtype=np.uint8)
    save = tmp_path / "cast.png"
    msg = backend.show_phase_array("SLM2", phase, gray, save, allow_png_fallback=False)
    assert len(slm.phase_calls) == 1
    sent, flags, phase_unit = slm.phase_calls[0]
    assert phase_unit == TWOPI
    assert flags is None
    assert np.allclose(sent, np.mod(phase, TWOPI), atol=1e-6)
    assert backend.devices["SLM2"].last_transfer == "direct_phase_array:showPhaseData:radians"
    assert "phase_unit=2*pi rad" in msg


def test_phase_file_uses_phase_file_api_not_image_api(tmp_path):
    backend, slm = make_backend(tmp_path)
    backend.connect("SLM2", "6010-2381", 1030.0)
    p = tmp_path / "mask.png"
    p.write_bytes(b"dummy")
    msg = backend.show_phase_file("SLM2", p)
    assert slm.phase_file_calls == [str(p.resolve())]
    assert backend.devices["SLM2"].last_transfer == "phase_file:showPhaseDataFromFile"
    assert "PHASE-file transfer" in msg
