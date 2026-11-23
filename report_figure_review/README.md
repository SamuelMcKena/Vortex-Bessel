# Research figure repair — Claude handoff

**Status: UNAPPROVED / REPAIR NEEDED.** This is a working branch for research-figure triage, NOT a validated optical bench release. Main production model and hardware controls are unchanged.

## First read
1. [FIGURE_QA_INDEX.md](FIGURE_QA_INDEX.md) — a 34-item figure-by-figure defect log.
2. [CAPTIONS_DRAFT.md](CAPTIONS_DRAFT.md) — one replacement/restricted caption for each figure, to revise after fixing the content.
3. [CLAUDE_TASK.md](CLAUDE_TASK.md) — full execution requirements and scientific check list.

## Data and code pointers already present in repository
- [18-plane q=20 scan derived metrics](../notebooks/experimental/axicon_aberration_correction/bessel_zscan_metrics.csv) and [raw-frame QC table](../notebooks/experimental/axicon_aberration_correction/bessel_zscan_frame_qc.csv).
- [Retrieved phase coefficients — UNCALIBRATED](../notebooks/experimental/axicon_aberration_correction/UNCALIBRATED_DO_NOT_APPLY_aberration_coefficients.csv).
- [Nominal f=300 mm relay model](../vbb_study/digital_twin/nominal_f300_4f.py), [nominal configuration](../configs/hardware/cslm_f300_nominal_4f_profile.json), and [explicit physical relay alternative](../vbb_study/digital_twin/vortex_explicit_4f.py).
- Related work: `feature/hexapod-laser-standalone` and `feature/virtual-lab-blind-correction`.

## Recovered but not yet GitHub-uploaded binary assets

The **local ChatGPT handoff archive** contains the previous ~34 unapproved figures with scripts/manifests, full 11 November raw scope experiment ZIP, original scope images, trace/jitter CSV extracts, a corrected F2.1 example and newly replotted PHAROS timing. Those binary files could not be written directly through the available GitHub text connector from the local working environment. This branch thus does **not** claim to contain the entire artifact archive or a validated 16 November pulse-envelope sequence. The user can give Claude the matching downloadable repair ZIP from this conversation. A one-command Windows git upload script is inside the handoff ZIP if binaries are required in this branch. Do not claim those files were uploaded until verified in GitHub.

## Optical geometry and claims
- SLM1 conditions/corrects incident wavefront. SLM2 applies chosen vortex and carrier; older +10/-10 split is **historical**.
- SLM1 to SLM2 ~300 mm estimated. Nominal SLM2 to L1 300 mm; L1 to Fourier stop 300 mm; Fourier stop to L2 300 mm; L2 to relay image 300 mm.
- L1–L2 is **600 mm**. Pinhole size/centre and axicon SKU/angle are unverified. Do not silently adjust the model to match a 46–50 µm measured annulus.
- Photodiode electrical detector pulses are **not** pulse energy; small 120 µs scope windows do not establish all later optical emissions.

## Practical acceptance gates
Physics/reference derivation and units; experiment/calibration provenance; grid convergence; unambiguous normalisation; full-size print (160 mm width) visual check; no clipped text, legend overlap or misleading annotations; original-data SHA-256; reproducible script+JSON manifest; report-worthy scientific conclusion. *Omit* figures failing any key gate instead of padding chapter count.