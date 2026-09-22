import numpy as np
import pytest
from slm_lab_control.hardware.backends import DummySlmBackend
from slm_lab_control.lab_workbench import cast_session_trial
from vbb_study.lab import core
from vbb_study.lab.demo import setup_demo


def test_trial_cast_exact_total_radians_and_restore(tmp_path):
    root=setup_demo(tmp_path/'s');r=core.plan_round(root,'astig_x');tid=r['trials'][1]
    backend=DummySlmBackend(tmp_path/'casts');backend.init_sdk()
    backend.connect('SLM2','DEMO_SLM2',1030.)
    receipt=cast_session_trial(backend,root,tid)
    sent=np.load(backend.devices['SLM2'].last_path)
    expected=np.load(root/'trials'/tid/'total_phase_rad.npy')
    np.testing.assert_allclose(sent,expected)
    assert receipt['phase_sha256']==core.digest(root/'trials'/tid/'total_phase_rad.npy')
    assert set(backend.devices)=={'SLM2'}
    cast_session_trial(backend,root,'baseline',restore_base=True)
    restored=np.load(backend.devices['SLM2'].last_path)
    np.testing.assert_allclose(np.exp(1j*restored),np.exp(1j*np.load(root/'calibration/base_phase_rad.npy')),atol=6e-6)


def test_trial_cast_refuses_wrong_panel_or_missing_backend(tmp_path):
    root=setup_demo(tmp_path/'s');r=core.plan_round(root,'astig_x')
    with pytest.raises(ValueError,match='Connect'):
        cast_session_trial(None,root,r['start'])
    backend=DummySlmBackend(tmp_path/'casts');backend.init_sdk();backend.connect('SLM2','WRONG',1030.)
    with pytest.raises(ValueError,match='differs'):
        cast_session_trial(backend,root,r['start'])
