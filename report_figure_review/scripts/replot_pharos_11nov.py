"""Read-only replot of actual PHAROS 11 Nov J1/J2 delay CSVs from original archive.

Usage:
  python report_figure_review/scripts/replot_pharos_11nov.py Laser_Lab_Results_2026-11-11.zip --out-dir plots
Does not connect to any laser controller, scope or hardware.
"""
from argparse import ArgumentParser
from pathlib import Path
from zipfile import ZipFile
import csv, io, json, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kstest

def read_jitter(zf, needle):
    matches=[n for n in zf.namelist() if needle in n and n.endswith("/jitter_delays.csv")]
    if len(matches)!=1: raise RuntimeError("Expected exactly one "+needle+" jitter CSV; found "+str(len(matches)))
    data=zf.read(matches[0])
    rows=list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    v=np.array([float(r["delay_us"]) for r in rows if r["status"].strip()=="ok"])
    if len(v)!=30: raise RuntimeError("Expected 30 successful jitter records; got "+str(len(v)))
    return v, {"archive_member":matches[0],"sha256":hashlib.sha256(data).hexdigest(),"n":len(v)}

def main(zip_path, out_dir):
    out_dir.mkdir(parents=True,exist_ok=True)
    with ZipFile(zip_path) as z:
        one, m1=read_jitter(z,"J1_jitter_1kHz")
        two, m2=read_jitter(z,"J2_jitter_1Hz")
    plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":8,"pdf.fonttype":42})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(160/25.4,65/25.4),layout="constrained")
    result={}
    for label,values,meta,color in [
        ("nominal 1 kHz",one,m1,"#176ea3"),("nominal 1 Hz",two,m2,"#d65f0e")]:
        ax.hist(values,bins=np.arange(0,45,5),alpha=.55,edgecolor="white",color=color,label=label)
        x=np.sort(values)
        bx.step(np.r_[0,x,40],np.r_[0,np.arange(1,len(x)+1)/len(x),1],
                where="post",lw=1.2,color=color,label=label)
        kstest_result=kstest(values,"uniform",args=(0,40))
        result[label]=dict(meta,mean_us=float(values.mean()),median_us=float(np.median(values)),
                            sd_us=float(values.std(ddof=1)),min_us=float(values.min()),
                            max_us=float(values.max()),KS_D=float(kstest_result.statistic),
                            KS_p=float(kstest_result.pvalue))
    bx.plot([0,40],[0,1],"k--",lw=.9,label="U(0,40 µs) reference")
    ax.set(xlim=(0,40),xlabel="LX8 fall to detector rise (µs)",ylabel="Capture count")
    bx.set(xlim=(0,40),ylim=(0,1.01),xlabel="LX8 fall to detector rise (µs)",
           ylabel="Empirical cumulative probability")
    ax.legend(frameon=False,fontsize=7)
    bx.legend(frameon=False,fontsize=7,loc="lower right")
    for a,l in [(ax,"(a)"),(bx,"(b)")]:
        a.text(.03,.97,l,transform=a.transAxes,ha="left",va="top",fontsize=8,fontweight="bold")
    fig.savefig(out_dir/"pharos_11nov_raw_jitter.pdf",bbox_inches="tight",pad_inches=.08)
    fig.savefig(out_dir/"pharos_11nov_raw_jitter.png",dpi=600,bbox_inches="tight",pad_inches=.08)
    plt.close(fig)
    (out_dir/"pharos_11nov_raw_jitter.json").write_text(json.dumps({
        "claim_boundary":"actual CSV-derived detector timing; not absolute optical pulse time",
        "timing_definition":"LX8 falling 2.5V crossing to CH2 50pct rise (scope exported traces)",
        "reference_period_us":40,"series":result,
        "warning":"Failure to reject U(0,40us) does not establish cause or one-shot reliability."
        },indent=2))

if __name__=="__main__":
    ap=ArgumentParser()
    ap.add_argument("zip_path",type=Path)
    ap.add_argument("--out-dir",type=Path,default=Path("report_figure_review/generated"))
    a=ap.parse_args()
    main(a.zip_path,a.out_dir)
