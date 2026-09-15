"""Failure-path and end-to-end contracts for manual lab correction."""
import json
import shutil
import struct
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from vbb_study.lab import core
from vbb_study.lab.beamage import read_bmg
from vbb_study.lab.demo import setup_demo, synthetic_capture, run_demo


def captured_round(tmp_path):
    root = setup_demo(tmp_path/'session')
    r = core.plan_round(root, 'astig_x')
    for i, tid in enumerate(r['trials']):
        synthetic_capture(root, tid, seed=i)
    return root, r


def test_full_demo_preserves_scientific_status(tmp_path):
    result = run_demo(tmp_path/'demo')
    assert result['acceptance']['verification_passed']
    assert not result['acceptance']['experimental_accepted']
    assert 'SIMULATED' in result['acceptance']['status']
    assert Path(result['report']).is_file()


def test_phase_sum_preserves_vortex_and_carrier():
    y, x = np.indices((128,128)); theta = np.arctan2(y-63.5,x-63.5)
    base = 20*theta+2*np.pi*x/20
    correction = .2*core.zernike(base.shape, [63.5,63.5], 48, 2,2)
    actual = core.compose_phase(base, correction)
    np.testing.assert_allclose(np.exp(1j*actual), np.exp(1j*base)*np.exp(1j*correction), atol=4e-7)
    outside = np.hypot(y-63.5,x-63.5) > 48
    np.testing.assert_allclose(np.exp(1j*actual[outside]),np.exp(1j*base[outside]),atol=4e-7)


def test_zernike_sign_and_rms():
    a = core.zernike((501,501), [250,250], 240, 2,2)
    y,x=np.indices(a.shape); disk=np.hypot(y-250,x-250)<=240
    assert abs(np.mean(a[disk]**2)-1)<.002
    assert a[250,400] > 0 and a[400,250] < 0
    b = core.zernike((501,501), [250,250], 240, 2,-2)
    assert b[350,350]>0 and b[150,350]<0


def test_changed_calibration_and_phase_are_blocked(tmp_path):
    root = setup_demo(tmp_path/'s'); r=core.plan_round(root,'coma_x')
    phase=root/'trials'/r['trials'][1]/'total_phase_rad.npy'
    np.save(phase,np.zeros((64,64)))
    with pytest.raises(ValueError,match='phase modified'):
        core.trial_info(root,r['trials'][1])
    p=root/'calibration/profile.json'; p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='calibration modified'):
        core.session(root)


def test_no_acceptance_without_fresh_verification(tmp_path):
    root,r=captured_round(tmp_path)
    assert core.evaluate(root)['recommended_trial']
    with pytest.raises(ValueError, match='verify-plan'):
        core.accept(root)
    ids=core.plan_verification(root)
    with pytest.raises(ValueError,match='already planned'):
        core.plan_verification(root)
    source=root/'trials'/r['start']/'raw'
    for f in source.iterdir(): shutil.copyfile(f,root/'incoming'/ids[0]/f.name)
    with pytest.raises(ValueError,match='Reused raw'):
        core.capture(root,ids[0],settings_id='SYNTHETIC_FIXED')


def test_metadata_change_does_not_make_reused_pixels_fresh(tmp_path):
    root,r=captured_round(tmp_path); ids=core.plan_verification(root)
    for f in (root/'trials'/r['start']/'raw').iterdir():
        # Text encoding differs, intensity values remain identical.
        np.savetxt(root/'incoming'/ids[0]/(f.stem+'.txt'),np.load(f))
    with pytest.raises(ValueError, match='Reused pixel'):
        core.capture(root,ids[0],settings_id='SYNTHETIC_FIXED')


def test_drift_blocks_candidate(tmp_path):
    root=setup_demo(tmp_path/'s'); r=core.plan_round(root,'astig_x')
    for i,tid in enumerate(r['trials']):
        synthetic_capture(root,tid,seed=i,drift=.25 if tid==r['end'] else 0.)
    assert core.evaluate(root)['recommended_trial'] is None


def test_failed_verification_does_not_update_accepted(tmp_path):
    root,r=captured_round(tmp_path)
    for i,tid in enumerate(core.plan_verification(root)):
        synthetic_capture(root,tid,seed=50+i,worsen=True)
    result=core.accept(root)
    assert not result['verification_passed']
    assert core.read_json(root/'session.json')['accepted_trial'] is None


def test_power_loss_blocks_false_shape_success(tmp_path):
    root=setup_demo(tmp_path/'s'); r=core.plan_round(root,'astig_x')
    for i,tid in enumerate(r['trials']):
        _,t=core.trial_info(root,tid)
        synthetic_capture(root,tid,seed=i,power_scale=.5 if t['role']=='candidate' else 1.)
    assert core.evaluate(root)['recommended_trial'] is None


def test_next_round_accumulates_only_accepted_phase(tmp_path):
    root,r=captured_round(tmp_path)
    for i,tid in enumerate(core.plan_verification(root)): synthetic_capture(root,tid,seed=80+i)
    assert core.accept(root)['verification_passed']
    _,s,p=core.session(root)
    accepted=np.load(root/'trials'/s['accepted_trial']/'correction_phase_rad.npy')
    next_=core.plan_round(root,'coma_x')
    np.testing.assert_array_equal(np.load(root/'trials'/next_['start']/'correction_phase_rad.npy'),accepted)


def test_lut_owner_prevents_double_conversion(tmp_path):
    root=setup_demo(tmp_path/'s');r=core.plan_round(root,'astig_x')
    with pytest.raises(ValueError,match='double conversion'):
        core.export_phase(root,r['start'],lut_csv='not_a_real_lut.csv')


def test_bmg_layout_signed_values_and_truncation(tmp_path):
    p=tmp_path/'test.bmg';header=bytearray(481);struct.pack_into('<4I',header,0,1,12,2048,2048)
    a=np.zeros((2048,2048),dtype='<i4');a[0,0]=-95;a[200,400]=3272
    p.write_bytes(header+a.tobytes())
    image,meta=read_bmg(p);np.testing.assert_array_equal(image,a)
    p.write_bytes(p.read_bytes()[:-4])
    with pytest.raises(ValueError,match='length'):read_bmg(p)


def test_rgb_and_summary_txt_rejected(tmp_path):
    f=tmp_path/'rgb.png';Image.fromarray(np.zeros((10,10,3),dtype=np.uint8)).save(f)
    with pytest.raises(ValueError,match='2D'):core.load_array(f)
    f=tmp_path/'base.txt';f.write_text('Software Version: 1.07.10\nCamera Name: Beamage-4M\n')
    with pytest.raises(ValueError):core.load_array(f)


def test_saturation_incomplete_and_camera_settings_fail(tmp_path):
    root=setup_demo(tmp_path/'s');r=core.plan_round(root,'astig_x');tid=r['start']
    with pytest.raises(ValueError,match='settings-id'):core.capture(root,tid,settings_id='other')
    with pytest.raises(ValueError,match='Missing'):core.capture(root,tid,settings_id='SYNTHETIC_FIXED')
    rng=np.random.default_rng(42)
    for z in range(3):
        for rep in range(1,4):
            a=rng.normal(100,1,(64,64));a[20,20]=4095
            np.save(root/'incoming'/tid/f'z{z:03d}_r{rep:02d}.npy',a)
    with pytest.raises(ValueError,match='saturated'):core.capture(root,tid,settings_id='SYNTHETIC_FIXED')


def test_dark_core_and_radial_profile_are_explicit_metrics():
    y,x=np.indices((101,101));r=np.hypot(y-50,x-50)
    target=np.exp(-.5*((r-20)/3)**2);roi=r<40
    images=[target.copy() for _ in range(3)]
    base=core.plane_scores(images,target,roi)
    filled=[a+.5*np.exp(-r*r/30) for a in images]
    metrics=core.plane_scores(filled,target,roi)
    assert metrics['core_fraction']>base['core_fraction']+.01
    assert metrics['radial_profile_l1']>base['radial_profile_l1']
