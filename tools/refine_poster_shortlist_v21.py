"""Second-pass curation for poster figure shortlist v2.

The all-z q=20 atlas, dense metrics dashboard, and raw 3-D mesh are scientifically
useful but poor poster hero figures.  Replace them with cleaner canonical evidence:
a single-z confirmation, longitudinal falsification maps, and agreement-vs-z.
"""

from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "poster" / "figure_shortlist_v2"


def copy(src: Path, name: str) -> Path:
    if not src.exists():
        raise FileNotFoundError(src)
    dst = OUT / name
    shutil.copy2(src, dst)
    return dst


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in (
        "08_measured_fit_corrected_ideal.png",
        "09_single_mask_metrics_vs_z.png",
        "10_measured_vs_corrected_3d.png",
    ):
        p = OUT / old
        if p.exists():
            p.unlink()

    copy(
        ROOT / "figures/experimental/q20_aberration/single_mask/single_z_double_confirmation_minus10.png",
        "08_q20_single_z_confirmation.png",
    )
    copy(
        ROOT / "figures/experimental/q20_aberration/phase_error_recreation/phase_error_recreation_signed_xz_yz.png",
        "09_q20_phase_error_recreation_xz_yz.png",
    )
    copy(
        ROOT / "figures/experimental/q20_aberration/phase_error_recreation/phase_error_recreation_agreement_vs_z.png",
        "10_q20_phase_error_agreement_vs_z.png",
    )

    ordered = [
        OUT / "01_self_healing_sequence.png",
        OUT / "02_sample_interface_comparison.png",
        OUT / "03_computational_route.png",
        OUT / "04_ideal_beam_family.png",
        OUT / "05_axicon_decentre.png",
        OUT / "06_nonideal_apex.png",
        OUT / "07_retrieved_aberration_phase.png",
        OUT / "08_q20_single_z_confirmation.png",
        OUT / "09_q20_phase_error_recreation_xz_yz.png",
        OUT / "10_q20_phase_error_agreement_vs_z.png",
    ]

    fig = plt.figure(figsize=(18, 20), facecolor="#171b1f")
    gs = fig.add_gridspec(5, 2, hspace=0.20, wspace=0.10)
    for i, p in enumerate(ordered):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_facecolor("#171b1f")
        ax.imshow(plt.imread(p))
        ax.set_title(p.stem.replace("_", " "), color="white", fontsize=12, weight="bold")
        ax.axis("off")
    fig.suptitle("Poster figure shortlist v2.1 — visual quality first", color="white", fontsize=22, weight="bold", y=0.995)
    fig.text(
        0.5,
        0.006,
        "Explicitly binned: generic sensitivity bar chart · jagged V3 surface · stitched source-to-sample plot · all-z q20 atlas",
        ha="center",
        color="#ffb4ae",
        fontsize=12,
        weight="bold",
    )
    fig.savefig(OUT / "00_shortlist_contact_sheet_v21.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
