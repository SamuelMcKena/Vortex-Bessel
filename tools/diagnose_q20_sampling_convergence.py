"""Check transverse sampling convergence of the q=20 full-route/Miao comparison.

The 8 mm / 512 grid used by the iterative benchmarks has dx=15.625 um.  This
passes the z-sweep spectral-support guard, but q=20 radial structure is fine
enough that a separate transverse convergence check is required.  The same
physical route and Miao compatibility calculation are therefore repeated at
512, 768 and 1024 samples over the same 8 mm window, on three planes within the
formed Bessel region.
"""
from __future__ import annotations
from pathlib import Path
import json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import benchmark_q20_miao_vs_digital_twin as base  # noqa: E402
import diagnose_miao_full_route_compatibility as comp  # noqa: E402
from vbb_study.digital_twin.vortex_system_route import SystemErrorConfig  # noqa: E402

OUT=ROOT/'outputs'/'validation'/'q20_sampling_convergence'
Z=np.asarray([85e-3,110e-3,132e-3],float)
WINDOW=8e-3


def run_n(n: int) -> dict:
    base.GRID_N=int(n); base.WINDOW_M=WINDOW
    route=base._route(SystemErrorConfig(),None)
    stack=base._propagate(route,Z)
    rows=[comp._miao_prediction(route,float(z),stack[i]) for i,z in enumerate(Z)]
    return {'N':int(n),'dx_um':WINDOW/n*1e6,'rows':rows,
            'mean_pearson_r':float(np.mean([r['pearson_r'] for r in rows])),
            'mean_rmse':float(np.mean([r['rmse_peak_normalized'] for r in rows]))}


def build(out: Path=OUT):
    out.mkdir(parents=True,exist_ok=True)
    results=[run_n(n) for n in (512,768,1024)]
    s={'study':'q20 transverse sampling convergence on formed Bessel region',
       'window_mm':WINDOW*1e3,'results':results}
    (out/'summary.json').write_text(json.dumps(s,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(s,indent=2)); return s

if __name__=='__main__': build()
