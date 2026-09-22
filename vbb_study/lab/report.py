"""Portable local HTML report with frozen-target provenance and measured frames."""
from pathlib import Path
import html
import csv
import numpy as np
from PIL import Image
from . import core


def build_report(root):
    root, s, p = core.session(root)
    result = core.evaluate(root)
    folder = root/'report'; folder.mkdir(exist_ok=True)
    rows = result['results']
    with (folder/'scores.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['trial_id', 'coefficient_rad_rms', 'mean_loss', 'improvement', 'eligible', 'reasons'])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows)
    esc = html.escape
    pieces = ['<!doctype html><meta charset="utf-8"><title>Beam correction session</title>',
      '<style>body{font:17px system-ui;margin:32px;max-width:1200px;background:#10151e;color:#eef2f8}table{border-collapse:collapse;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #364153}img{width:250px;image-rendering:pixelated}figure{display:inline-block;margin:12px}p{line-height:1.6}a{color:#8dcaff}</style>',
      '<h1>Beam correction session</h1>', f'<p>{esc(result["status"])}</p>',
      f'<p>{esc(result["target"])}</p>',
      f'<p>Recommended trial: <b>{esc(str(result["recommended_trial"]))}</b>. Accepted cumulative trial: <b>{esc(str(s["accepted_trial"]))}</b>.</p>',
      '<p>Lower loss is better. Power checks use background-subtracted counts at fixed settings. Test coefficients are native-map radians RMS, not the old GUI waves coefficients. No trial is accepted by this screening report.</p>',
      '<table><tr><th>Trial</th><th>Added rad RMS</th><th>Shape loss</th><th>Result</th></tr>']
    for row in rows:
        pieces.append(f'<tr><td>{esc(row["trial_id"])}</td><td>{row["coefficient_rad_rms"]:+.3f}</td><td>{row["mean_loss"]:.5f}</td><td>{esc("Eligible for fresh verification" if row["eligible"] else "; ".join(row["reasons"]))}</td></tr>')
    pieces.append('</table><h2>Measured images</h2><p>Fixed axis and crop; shared counts scale within each z plane. These are measured camera images, not simulated corrections.</p>')
    selected = [s['rounds'][-1]['start']]
    if result['recommended_trial']:
        selected.append(result['recommended_trial'])
    selected.append(s['rounds'][-1]['end'])
    arrays = {tid: core.measurements(root, tid) for tid in selected}
    for plane in p['planes']:
        cy, cx = map(lambda v: int(round(v)), plane['axis_yx_px'])
        h = int(p['roi_radius_px'])
        crops = [np.mean(arrays[tid][plane['id']], axis=0)[cy-h:cy+h+1, cx-h:cx+h+1] for tid in selected]
        scale = max(float(a.max()) for a in crops)
        pieces.append(f'<h3>{esc(plane["id"])} — {plane["z_mm"]:g} mm</h3>')
        for tid, crop in zip(selected, crops):
            name = f'{tid}_{plane["id"]}.png'
            Image.fromarray(np.uint8(np.clip(crop/scale, 0, 1)*255)).save(folder/name)
            pieces.append(f'<figure><img src="{name}"><figcaption>{esc(tid)}</figcaption></figure>')
    pieces.append('<p>The rotational target preserves the first baseline radial profile. This workflow addresses asymmetric distortion; it cannot certify ring size, propagation invariance outside the measured planes, vortex sign, or a corrected optical wavefront.</p>')
    (folder/'index.html').write_text('\n'.join(pieces), encoding='utf-8')
    return folder/'index.html'
