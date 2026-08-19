"""Render a slide-ready rounded-apex avoidance schematic.

The axicon is drawn as a single conical body whose final sharp-apex region is
replaced by a smooth rounded/blunted cap. The cap remains within the envelope of
the corresponding ideal sharp cone: no circle, bulb, insert, or protruding tip is
overlaid. The two panels compare ordinary B0 illumination with a hollow
non-vortex l=0 input that clears the rounded apex while retaining the conical
rays needed to reconstruct a bright-centred Bessel field.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, FancyArrowPatch
from matplotlib import patheffects as pe

BG="#020407"; TEXT="#eef3f6"; MUTED="#b8c0c7"
BLUE="#63c4ff"; CYAN="#45dfcf"; RED="#ff4150"; ORANGE="#ff9e23"; GOLD="#ffe06d"


def _hexrgb(value: str):
    value=value.lstrip("#")
    return tuple(int(value[i:i+2],16)/255 for i in (0,2,4))


def render(output: Path):
    fig,ax=plt.subplots(figsize=(16,9),dpi=180)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off")

    def glow_line(x,y,color,lw=1.7,alpha=1,z=5,ls="-"):
        for w,a in ((9,.035),(5,.07),(2.5,.14)):
            ax.plot(x,y,color=color,lw=lw+w,alpha=a,solid_capstyle="round",zorder=z-1,ls=ls)
        ax.plot(x,y,color=color,lw=lw,alpha=alpha,solid_capstyle="round",zorder=z,ls=ls)

    def txt(x,y,s,color=TEXT,size=13,ha="center",va="center",weight=None):
        t=ax.text(x,y,s,color=color,fontsize=size,ha=ha,va=va,weight=weight,zorder=20)
        t.set_path_effects([pe.withStroke(linewidth=3,foreground=BG,alpha=.8)])
        return t

    def arrow(x0,y0,x1,y1,color,lw=1.5,ms=15):
        p=FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=ms,color=color,lw=lw,zorder=18)
        p.set_path_effects([pe.withStroke(linewidth=6,foreground=color,alpha=.06)])
        ax.add_patch(p)

    def rounded_tip_axicon(cx,cy,h=3.35,w=1.75,round_height=.34):
        """Single-body axicon with a recessed smooth rounded apex.

        The upper/lower tangent points lie on the original ideal-cone flanks.
        The cubic cap replaces the final sharp region and never extends beyond
        the x-position of the ideal sharp apex.
        """
        x_left=cx-w/2
        x_apex=cx+w/2
        y_top=cy+h/2
        y_bottom=cy-h/2

        # Pick symmetric tangent points on the straight cone flanks.
        y_tan=float(round_height)
        flank_fraction=1.0-y_tan/(h/2.0)
        x_tan=x_left+flank_fraction*(x_apex-x_left)

        # One smooth rounded cap replaces the final tip.  Both Bezier control
        # points remain just behind the ideal sharp apex, so the rounded region
        # reads as a blunted cone rather than a protruding bulb/nipple.
        cap_x=x_apex-0.035
        verts=[
            (x_left,y_top),
            (x_tan,cy+y_tan),
            (cap_x,cy+0.22),
            (cap_x,cy-0.22),
            (x_tan,cy-y_tan),
            (x_left,y_bottom),
            (x_left,y_top),
        ]
        codes=[
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.CURVE4,MplPath.CURVE4,MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ]
        patch=PathPatch(MplPath(verts,codes),facecolor="#173b5d",edgecolor=BLUE,lw=1.7,alpha=.95,zorder=9)
        patch.set_path_effects([
            pe.withStroke(linewidth=12,foreground=BLUE,alpha=.035),
            pe.withStroke(linewidth=7,foreground=BLUE,alpha=.07),
            pe.withStroke(linewidth=3.2,foreground=BLUE,alpha=.13)])
        ax.add_patch(patch)
        glow_line([x_left,x_left],[y_bottom,y_top],BLUE,1.55,z=10)
        return cap_x

    def beam_gradient(x0,x1,yc,sigma,color,annular=False):
        nx,ny=700,240
        ys=np.linspace(yc-1.2,yc+1.2,ny)
        _,Y=np.meshgrid(np.linspace(x0,x1,nx),ys)
        intensity=np.exp(-.5*((Y-yc)/sigma)**2)
        if annular:
            intensity*=1-np.exp(-.5*((Y-yc)/.28)**2)
        rgba=np.zeros((ny,nx,4)); rgba[...,:3]=_hexrgb(color); rgba[...,3]=.48*intensity
        ax.imshow(rgba,extent=[x0,x1,ys[0],ys[-1]],origin="lower",aspect="auto",zorder=2)

    txt(8,8.55,"Rounded-tip axicon — ordinary illumination versus apex avoidance",size=24,weight="bold")
    txt(8,8.15,"same blunted axicon • only the incident field is changed",color=MUTED,size=12.5)
    glow_line([8,8],[.65,7.85],"#26303a",.7,z=1)

    yc=4.25; xc=4.0
    txt(xc,7.58,"ordinary B0 illumination",size=18.5,weight="bold")
    txt(xc,7.20,"rounded apex is illuminated",color=RED,size=12)
    tip=rounded_tip_axicon(xc-.35,yc)
    beam_gradient(xc-3.75,xc-1.14,yc,.60,RED,False)
    arrow(xc-3.25,yc,xc-1.25,yc,RED)
    txt(xc-2.35,5.35,"Gaussian / B0 input",size=11.5)
    glow_line([xc-3.8,xc+3.2],[yc,yc],"#c9eaff",.7,alpha=.55,z=3,ls="--")
    glow_line([tip,xc+2.65],[yc,yc+1.05],RED,1.5,z=10)
    glow_line([tip,xc+2.65],[yc,yc-1.05],RED,1.5,z=10)
    glow_line([tip,xc+2.45],[yc,yc],ORANGE,1.7,z=9,ls="--")
    txt(xc+1.15,4.68,"apex contribution",color=ORANGE,size=10.5)
    for i,xpos in enumerate(np.linspace(tip+.28,xc+2.15,7)):
        amplitude=[.55,.72,.88,.64,.92,.62,.75][i]
        glow_line([xpos,xpos],[yc-.11*amplitude,yc+.11*amplitude],RED,2.2,z=8)
    txt(xc+1.05,6.15,"rounded-apex field interferes\nwith the conical contribution",size=12)
    txt(xc,1.12,"longitudinal modulation",color=RED,size=13,weight="bold")
    arrow(xc,1.7,xc,3.5,RED,1.2,11)

    yc=4.25; xc=12.0
    txt(xc,7.58,"annular ℓ = 0 illumination",size=18.5,weight="bold")
    txt(xc,7.20,"dark centre clears the rounded apex",color=CYAN,size=12)
    tip=rounded_tip_axicon(xc-.35,yc)
    beam_gradient(xc-3.75,xc-1.14,yc,.68,CYAN,True)
    arrow(xc-3.2,yc+.58,xc-1.25,yc+.58,CYAN,1.4,13)
    arrow(xc-3.2,yc-.58,xc-1.25,yc-.58,CYAN,1.4,13)
    txt(xc-2.35,5.45,"hollow / annular input\n(no helical phase)",size=11.2)
    glow_line([xc-3.8,xc+3.2],[yc,yc],"#c9eaff",.7,alpha=.45,z=3,ls="--")
    glow_line([tip,xc+2.65],[yc,yc+1.00],CYAN,1.5,z=10)
    glow_line([tip,xc+2.65],[yc,yc-1.00],CYAN,1.5,z=10)
    glow_line([tip+.40,xc+2.35],[yc,yc],GOLD,2.0,z=11)
    for off,a in ((.28,.8),(-.28,.8),(.48,.5),(-.48,.5)):
        glow_line([tip+.48,xc+2.30],[yc+off,yc+off],CYAN,.9,alpha=a,z=7)
    txt(xc+1.05,6.15,"conical field reconstructs a\nbright-centred Bessel beam",size=12)
    txt(xc,1.12,"reduced apex sensitivity",color=CYAN,size=13,weight="bold")
    arrow(xc,1.7,xc,3.5,CYAN,1.2,11)

    txt(8,.35,"avoid the imperfect apex while preserving the conical rays that generate the Bessel field",color=MUTED,size=11.3)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,facecolor=BG,bbox_inches="tight",pad_inches=.03)
    plt.close(fig)


if __name__=="__main__":
    render(Path("figures/presentation/schematic_axicon_tip_avoidance_clean.png"))
