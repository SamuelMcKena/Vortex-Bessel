"""Inventory yesterday's known Beamage acquisition naming without guessing gains."""
from pathlib import Path
import csv
import re
import numpy as np
from .beamage import read_bmg
from .core import digest, save_json

# Literal prefixes from Calibration.zip. These amplitudes are UNNORMALISED
# GUI waves on its configured pupil, not coefficients of the new RMS-rad basis.
KNOWN = {'base': ('baseline', 0.), 'Astig_cos_ne': ('astig_cos', -.5),
 'Astig_cos_po': ('astig_cos', .5), 'Astig_sin_ne': ('astig_sin', -.5),
 'Astig_sin_po': ('astig_sin', .5), 'coma_cos_ne': ('coma_cos', -2.),
 'coma_cos_pos2.0': ('coma_cos', 2.), 'coma_sin_ne': ('coma_sin', -2.),
 'coma_sin_po': ('coma_sin', 2.), 'spherical_neg4': ('spherical', -4.),
 'spherical_pos4': ('spherical', 4.)}


def audit_calibration(folder, output):
    folder, output = Path(folder), Path(output)
    files = sorted(folder.glob('*.bmg'))
    if not files:
        raise ValueError('No BMG files found; point to the extracted Calibration directory.')
    if output.exists():
        raise ValueError('Audit output already exists; choose a new directory.')
    output.mkdir(parents=True)
    rows = []
    for f in files:
        match = re.fullmatch(r'(.+)_(\d+)', f.stem)
        prefix = match.group(1) if match else f.stem
        mode, value = KNOWN.get(prefix, ('UNRESOLVED_GAIN_OR_CONDITION', None))
        a, meta = read_bmg(f)
        rows.append({'file': f.name, 'sha256': digest(f), 'prefix': prefix,
                     'mode': mode, 'gui_waves_coefficient': value,
                     'repeat': int(match.group(2)) if match else None,
                     'min_counts': int(a.min()), 'max_counts': int(a.max()),
                     'negative_pixel_fraction': float(np.mean(a < 0)),
                     'fraction_at_nominal_full_scale': float(np.mean(a >= 4095))})
    with (output/'raw_inventory.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    logs = []
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() != '.bmg':
            text = f.read_text(encoding='utf-8', errors='replace')
            if text.startswith('Software Version:'):
                logs.append({'file': f.name, 'sha256': digest(f), 'kind': 'Beam profiler summary log, NOT a 2D intensity matrix'})
    groups = {}
    for r in rows:
        groups.setdefault(r['prefix'], []).append(r['repeat'])
    result = {'raw_bmg_count': len(rows), 'summary_log_count': len(logs),
              'groups': groups, 'summary_logs': logs,
              'unresolved_groups': sorted({r['prefix'] for r in rows if r['gui_waves_coefficient'] is None}),
              'status': 'HISTORICAL_INVENTORY_ONLY',
              'limits': ['No z positions or dark references supplied by filenames.',
                         'correction_1.bmg and correction__1.bmg do not uniquely encode correction gain; no gain assignment made.',
                         'Summary logs for several gains do not establish which raw group belongs to each gain.',
                         'Nominal saturation check on exported pixels cannot recover pre-processing/clipping inside camera software.',
                         'No historical acquisition is inserted as a fresh correction-validation capture.']}
    save_json(output/'inventory.json', result)
    return result
