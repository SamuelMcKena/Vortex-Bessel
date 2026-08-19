"""Render a clean presentation schematic for rounded-apex avoidance.

The schematic is intentionally explanatory rather than a quantitative ray trace:
ordinary B0 illumination overlaps a local rounded apex and therefore admits an
additional apex contribution, whereas a hollow l=0 input clears the imperfect
central region while retaining the conical rays needed to reconstruct a bright-
centred Bessel field.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle, FancyArrowPatch

BG="#05080b"; PANEL="#0a0f14"; TEXT="#f1f4f6"; MUTED="#aeb7bf"
CYAN="#43d7c4"; BLUE="#6ec8ff"; RED="#ff4d45"; ORANGE="#ff9d00"; GOLD="#ffd45c"; GRID="#27313a"


def render(output: Path):
    fig, ax = plt.subplots(figsize=(16,9)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off")
    ax.text(8,8.55,"Rounded-tip axicon — avoiding the apex with annular illumination",color=TEXT,fontsize=24,ha="center",weight="bold")
    ax.text(8,8.16,"same axicon geometry • only the incident beam is changed",color=MUTED,fontsize=12.5,ha="center")
    ax.add_patch(Rectangle((0.45,0.7),7.25,6.95,facecolor=PANEL,edgecolor=GRID,lw=1.2))
    ax.add_patch(Rectangle((8.30,0.7),7.25,6.95,facecolor=PANEL,edgecolor=GRID,lw=1.2))

    def arr(x0,y0,x1,y1,c,lw=2,ms=15):
        ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=ms,color=c,lw=lw))
    def axicon(cx,cy):
        pts=[[cx-0.75,cy+1.55],[cx+0.32,cy],[cx-0.75,cy-1.55]]
        ax.add_patch(Polygon(pts,closed=True,facecolor="#17314a",edgecolor=BLUE,lw=2))
        ax.plot([cx-0.75,cx-0.75],[cy-1.55,cy+1.55],color=BLUE,lw=2)
        ax.add_patch(Circle((cx+0.25,cy),0.25,facecolor="#214f6f",edgecolor=CYAN,lw=1.8))
        ax.add_patch(Circle((cx+0.25,cy),0.46,fill=False,edgecolor=CYAN,lw=1.2,ls="--"))
        return cx+0.32

    # ordinary input
    ax.text(4.05,7.2,"ordinary B0 illumination",color=TEXT,fontsize=18,ha="center",weight="bold")
    ax.text(4.05,6.87,"rounded apex is illuminated",color=RED,fontsize=12,ha="center")
    yc=4.25; tip=axicon(3.65,yc)
    ax.add_patch(Rectangle((0.8,yc-0.72),2.1,1.44,facecolor=RED,alpha=0.16,edgecolor="none"))
    ax.plot([0.8,2.9],[yc+0.72,yc+0.72],color=RED,lw=1.5); ax.plot([0.8,2.9],[yc-0.72,yc-0.72],color=RED,lw=1.5)
    arr(1.05,yc,2.75,yc,RED,1.8,13); ax.text(1.85,5.25,"Gaussian / B0 input",color=TEXT,fontsize=11.5,ha="center")
    ax.add_patch(Circle((3.90,yc),0.40,fill=False,edgecolor=RED,lw=2.2)); ax.text(3.95,2.28,"significant power\ncrosses rounded apex",color=RED,fontsize=11,ha="center"); arr(3.95,2.85,3.95,3.62,RED,1.4,10)
    ax.plot([tip,6.95],[yc,yc+1.0],color=RED,lw=1.7); ax.plot([tip,6.95],[yc,yc-1.0],color=RED,lw=1.7)
    ax.plot([tip,6.90],[yc,yc],color=ORANGE,lw=2.0,ls="--"); ax.text(5.45,4.55,"apex contribution",color=ORANGE,fontsize=10,ha="center")
    for x in [4.65,5.05,5.45,5.85,6.25,6.65]: ax.plot([x,x],[yc-0.14,yc+0.14],color=RED,lw=2.2,alpha=0.95)
    ax.text(5.82,6.03,"interference produces\nlongitudinal modulation",color=TEXT,fontsize=12,ha="center")
    ax.text(4.05,1.05,"rounded-apex sensitivity",color=RED,fontsize=13,ha="center",weight="bold")

    # annular l=0 input
    ax.text(11.92,7.2,"annular ℓ = 0 illumination",color=TEXT,fontsize=18,ha="center",weight="bold")
    ax.text(11.92,6.87,"dark centre clears the rounded apex",color=CYAN,fontsize=12,ha="center")
    yc=4.25; tip=axicon(11.25,yc)
    for off in (0.67,-0.67):
        ax.add_patch(Rectangle((8.65,yc+off-0.18),1.95,0.36,facecolor=CYAN,alpha=0.18,edgecolor="none"))
        ax.plot([8.65,10.60],[yc+off+0.18,yc+off+0.18],color=CYAN,lw=1.4); ax.plot([8.65,10.60],[yc+off-0.18,yc+off-0.18],color=CYAN,lw=1.4)
        arr(8.92,yc+off,10.35,yc+off,CYAN,1.6,11)
    ax.text(9.58,5.48,"hollow / annular input\n(no helical phase)",color=TEXT,fontsize=11,ha="center")
    ax.add_patch(Circle((11.50,yc),0.40,fill=False,edgecolor=CYAN,lw=2.2)); ax.text(11.50,2.28,"near-zero power\nthrough defective apex",color=CYAN,fontsize=11,ha="center"); arr(11.50,2.85,11.50,3.62,CYAN,1.4,10)
    ax.plot([tip,14.75],[yc,yc+0.95],color=CYAN,lw=1.7); ax.plot([tip,14.75],[yc,yc-0.95],color=CYAN,lw=1.7)
    ax.plot([12.2,14.9],[yc,yc],color=GOLD,lw=2.6)
    for off in (0.28,-0.28,0.46,-0.46): ax.plot([12.2,14.9],[yc+off,yc+off],color=CYAN,lw=1.0,alpha=0.6)
    ax.text(13.55,6.03,"conical field reconstructs a\nbright-centred Bessel beam",color=TEXT,fontsize=12,ha="center")
    ax.text(11.92,1.05,"reduced apex sensitivity",color=CYAN,fontsize=13,ha="center",weight="bold")
    ax.text(8,0.22,"Suppress illumination of the imperfect apex while preserving the conical rays that form the Bessel field",color=MUTED,fontsize=11.2,ha="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output,dpi=220,facecolor=BG,bbox_inches="tight",pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    render(Path("figures/presentation/schematic_axicon_tip_avoidance_clean.png"))
