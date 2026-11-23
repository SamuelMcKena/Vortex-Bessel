"""Rebuild the Gaussian-vs-Bessel concept plate after the 2026-11-23 QA failure.

Illustrative analytical model only. This script does not compare absolute powers or
represent a calibrated optical bench. Usage: python repair_fig2_1.py --out-dir output
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
from scipy.special import j0
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def build(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    lam = 1.029e-6
    waist = 3.9211168010713995e-6
    rayleigh = np.pi*waist**2/lam
    radial_extent = 2e-3
    cone_deg = 4.583452048821907
    cone = np.deg2rad(cone_deg)
    kr = 2*np.pi*np.sin(cone)/lam
    zmax = radial_extent/np.tan(cone)
    x = np.linspace(-27e-6,27e-6,450)
    zg = np.linspace(-.2e-3,.2e-3,600)
    zb = np.linspace(0,43e-3,1050)
    w = waist*np.sqrt(1+(zg/rayleigh)**2)
    Igauss = (waist/w)[None,:]**2*np.exp(-2*x[:,None]**2/w[None,:]**2)
    Iz = zb*np.exp(-2*(zb*np.tan(cone)/radial_extent)**2)
    Iz /= Iz.max()
    Ibessel = j0(kr*x[:,None])**2*Iz[None,:]
    plt.rcParams.update({'font.family':'serif','mathtext.fontset':'cm','font.size':8,
                         'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7})
    fig, axs = plt.subplots(1,3,figsize=(160/25.4,64/25.4),
                            gridspec_kw={'width_ratios':[1,1,1.24]},
                            layout='constrained')
    a,b,c = axs
    a.imshow(Igauss,origin='lower',extent=[zg[0]*1e3,zg[-1]*1e3,x[0]*1e6,x[-1]*1e6],
             aspect='auto',vmin=0,vmax=1,cmap='inferno',interpolation='nearest')
    b.imshow(Ibessel,origin='lower',extent=[0,zb[-1]*1e3,x[0]*1e6,x[-1]*1e6],
             aspect='auto',vmin=0,vmax=1,cmap='inferno',interpolation='nearest')
    a.set(xlabel='Axial position (mm)',ylabel='Transverse x (µm)')
    b.set(xlabel='Axial position (mm)',ylabel='Transverse x (µm)')
    b.axvline(zmax*1e3,c='white',ls=':',lw=.8)
    c.plot(zb*1e3,1/(1+(zb/rayleigh)**2),lw=1.3,label='Gaussian focus')
    c.plot(zb*1e3,Iz,lw=1.3,label='Finite Bessel')
    c.axvline(zmax*1e3,c='gray',ls='--',lw=.8)
    c.set(xlabel='Distance from waist (mm)',ylabel='I / respective maximum',xlim=(-.5,43),ylim=(0,1.1))
    c.legend(loc='upper right',frameon=False,fontsize=7)
    for ax,letter in [(a,'a'),(b,'b'),(c,'c')]:
        ax.text(.04,.97,'('+letter+')',va='top',transform=ax.transAxes,
                color='white' if ax != c else 'black',fontsize=8,fontweight='bold')
    fig.savefig(out_dir/'fig2_1_repaired.pdf',bbox_inches='tight',pad_inches=.08)
    fig.savefig(out_dir/'fig2_1_repaired.png',bbox_inches='tight',pad_inches=.08,dpi=600)
    plt.close(fig)
    (out_dir/'fig2_1_repaired.json').write_text(json.dumps({
        'status':'illustrative_not_bench_validated','wavelength_m':lam,
        'gaussian_waist_um':waist*1e6,'rayleigh_range_mm':rayleigh*1e3,
        'gaussian_zoom_mm':[-.2,.2],'bessel_zone_mm':[0,43],
        'geometric_zone_mm':zmax*1e3,'assumed_cone_half_angle_deg':cone_deg,
        'normalisation':'Each intensity map separately normalised to own maximum',
        'caption':'Gaussian focus at enlarged axial scale (a), finite Bessel zone (b), '
                  'and on-axis comparison at extended scale (c); not absolute-energy comparison.'},
        indent=2))
if __name__=='__main__':
    ap=ArgumentParser()
    ap.add_argument('--out-dir',default='report_figure_review/generated')
    args=ap.parse_args()
    build(Path(args.out_dir))
