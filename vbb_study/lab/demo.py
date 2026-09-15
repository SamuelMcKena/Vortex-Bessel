"""Synthetic camera harness, NOT a propagation simulation or experimental result."""
from pathlib import Path
import numpy as np
from . import core


def setup_demo(root, *, planes=3):
    root = Path(root)
    core.init_session(root)
    p = core.read_json(root/'profile.json')
    p['data_kind'] = 'synthetic_demo'
    p['slm'].update(shape_yx=[64, 64], panel_id='DEMO_SLM2', center_yx_px=[31.5, 31.5],
                    illuminated_radius_px=24, native_coordinates_verified=True,
                    phase_owner='existing_driver', phase_path_evidence=str((root/'demo_evidence.txt').resolve()))
    p['camera'].update(shape_yx=[64, 64], full_scale=4095, exposure_us=100, gain=0,
                       settings_id='SYNTHETIC_FIXED', linear_raw_export_verified=True)
    p['planes'] = [{'id': f'z{i:03d}', 'z_mm': float(i), 'axis_yx_px':[31.5, 31.5]} for i in range(planes)]
    p['z_reference'] = 'Synthetic stage coordinates only'
    p['roi_radius_px'] = 24
    (root/'demo_evidence.txt').write_text('SYNTHETIC camera and driver evidence; no hardware measured.', encoding='utf-8')
    core.save_json(root/'profile.json', p)
    rng = np.random.default_rng(500)
    darks = []
    for i in range(3):
        f = root/f'dark{i}.npy'; np.save(f, 10+rng.normal(0, .05, (64,64))); darks.append(f)
    y, x = np.indices((64,64))
    base = 20*np.arctan2(y-31.5,x-31.5) - 2*np.pi*y/20
    np.save(root/'base.npy', base)
    core.start_session(root, root/'base.npy', darks)
    return root


def synthetic_capture(root, tid, *, seed, drift=0., worsen=False, power_scale=1.):
    root, _, p = core.session(root)
    _, t = core.trial_info(root, tid)
    rng = np.random.default_rng(seed)
    y, x = np.indices(p['camera']['shape_yx'])
    radius, theta = np.hypot(y-31.5,x-31.5), np.arctan2(y-31.5,x-31.5)
    # Deliberately transparent test response: the negative trial reduces an
    # angular intensity modulation. It does not model SLM propagation physics.
    coeff = t['coefficient_rad_rms']
    if worsen:
        coeff = abs(coeff)
    for i, z in enumerate(p['planes']):
        for repeat in range(1, p['repeats']+1):
            signal = 1100*np.exp(-.5*((radius-(12+.3*i))/2)**2)*(1+(.4+coeff+drift)*np.cos(2*theta))
            image = 10+signal*power_scale+rng.normal(0, .15, signal.shape)
            np.save(root/'incoming'/tid/f'{z["id"]}_r{repeat:02d}.npy', image)
    return core.capture(root, tid, settings_id=p['camera']['settings_id'])


def run_demo(root):
    root = setup_demo(root)
    round_ = core.plan_round(root, 'astig_x')
    for i, tid in enumerate(round_['trials']):
        synthetic_capture(root, tid, seed=100+i)
    selection = core.evaluate(root)
    for i, tid in enumerate(core.plan_verification(root)):
        synthetic_capture(root, tid, seed=200+i)
    acceptance = core.accept(root)
    from .report import build_report
    report = build_report(root)
    return {'data_kind': 'SYNTHETIC_DEMO_NOT_LAB_EVIDENCE', 'selected': selection['recommended_trial'],
            'acceptance': acceptance, 'report': str(report)}
