"""Render the audited report-facing vector-beam figures.

Historical Stage-6 scalar/vector montages are deliberately excluded because
their panels do not share one forward model or one observable.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from scipy.special import jv


def vector_field(order: int, mode: str, n: int = 601):
    x = np.linspace(-55.0, 55.0, n)
    X, Y = np.meshgrid(x, x)
    r, phi = np.hypot(X, Y), np.arctan2(Y, X)
    amp = jv(order, 2*np.pi*r/6.2) * np.exp(-(r/42.0)**2)
    alpha = order*phi + (np.pi/2 if mode == "azimuthal" else 0.0)
    return x, amp*np.cos(alpha), amp*np.sin(alpha)


def analyser(ex, ey, angle_deg):
    a = np.deg2rad(angle_deg)
    return np.abs(ex*np.cos(a) + ey*np.sin(a))**2


def render_architecture(png: Path, pdf: Path):
    """Render the accepted collinear route without implying a split-arm interferometer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    labels = [
        "1029 nm\nGaussian", "POL + HWP\nequal H/V", "SLM1\n$+m\\phi$ on H",
        "HWP\nswap", "SLM2\n$-m\\phi+\\pi/2$ on V", "HWP\nswap back",
        "common 4F\norder filter", "QWP\n$-45^\\circ$", "physical\naxicon",
        "vector Bessel zone\n/ camera on z stage",
    ]
    x = np.arange(len(labels), dtype=float) * 2.15
    fig, ax = plt.subplots(figsize=(18.5, 3.2), constrained_layout=True)
    for i, (xi, label) in enumerate(zip(x, labels)):
        width = 1.55 if i < 9 else 2.25
        colour = "#eaf2f8" if i not in (2, 4, 7) else "#fff2cc"
        ax.add_patch(FancyBboxPatch((xi-width/2, -0.47), width, 0.94,
                    boxstyle="round,pad=0.04", facecolor=colour,
                    edgecolor="#24445f", linewidth=1.25))
        ax.text(xi, 0, label, ha="center", va="center", fontsize=9)
        if i < len(labels)-1:
            ax.annotate("", xy=(x[i+1]-0.83, 0), xytext=(xi+width/2, 0),
                        arrowprops=dict(arrowstyle="->", lw=1.25, color="#24445f"))
    ax.text((x[3]+x[5])/2, -0.83,
            "swap plates are conditional on the SLM polarisation axes",
            ha="center", va="top", fontsize=8.5, color="#555555")
    ax.set_title("Accepted sequential two-SLM vector-beam architecture", fontsize=14, pad=14)
    ax.set_xlim(-1.1, x[-1]+1.4); ax.set_ylim(-1.15, 0.82); ax.axis("off")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=320); fig.savefig(pdf); plt.close(fig)


def render_atlas(png: Path, pdf: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("radial",1),("radial",3),("azimuthal",1),("azimuthal",3)]
    angles = (0,45,90,135)
    fig, axes = plt.subplots(4,5,figsize=(14.6,11.6),constrained_layout=True)
    image = None
    for i,(mode,m) in enumerate(rows):
        x,ex,ey = vector_field(m,mode)
        s0 = np.abs(ex)**2 + np.abs(ey)**2
        assert np.allclose(analyser(ex,ey,17)+analyser(ex,ey,107),s0,rtol=2e-13,atol=2e-15)
        panels = [s0]+[analyser(ex,ey,a) for a in angles]
        scale=max(float(s0.max()),np.finfo(float).eps)
        for j,p in enumerate(panels):
            image=axes[i,j].imshow(np.clip(p/scale,0,1)**0.55,origin="lower",extent=[x[0],x[-1],x[0],x[-1]],cmap="magma",vmin=0,vmax=1)
            if i==0: axes[i,j].set_title("total" if j==0 else f"analyser {angles[j-1]} deg")
            if j==0: axes[i,j].set_ylabel(f"{mode}, m={m}\ny (um)")
            else: axes[i,j].set_yticklabels([])
            if i==3: axes[i,j].set_xlabel("x (um)")
            else: axes[i,j].set_xticklabels([])
    fig.colorbar(image,ax=axes,shrink=.82,label=r"display $(I/I_{S0,max})^{0.55}$")
    fig.suptitle(r"Analytic cylindrical-vector Bessel--Gauss observables"+"\n"+r"$I_\beta=|E_x\cos\beta+E_y\sin\beta|^2$; no device-realism branch",fontsize=15)
    png.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(png,dpi=320); fig.savefig(pdf); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,default=Path("outputs/figures/report_vector_audit")); a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    render_architecture(a.output_dir/"sequential_two_slm_architecture.png",a.output_dir/"sequential_two_slm_architecture.pdf")
    render_atlas(a.output_dir/"vector_analyser_atlas.png",a.output_dir/"vector_analyser_atlas.pdf")
    with (a.output_dir/"manifest.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["figure","evidence_type","sample_plane_claim"]); w.writerows([["sequential_two_slm_architecture","architecture_schematic",False],["vector_analyser_atlas","analytic_jones_observable",False]])

if __name__ == "__main__": main()
