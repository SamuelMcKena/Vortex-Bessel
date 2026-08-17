# Nathan MODE 2W-FIX - Sequential Architecture and Master Figure Repair

**Status:** MODE 2W-FIX presentation/source-audit correction. The original MODE 2W pack is superseded
and must not be treated as the publication/presentation pass.

## Sequential Architecture

The accepted architecture is collinear and sequential:

`1029 nm Gaussian -> POL1/HWP equal H/V prep -> SLM1(phi_H=+alpha+carrier) -> optional swap HWP
-> SLM2(phi_V=-alpha+pi/2+carrier) -> optional swap-back -> common 4F -> QWP -> axicon -> camera`.

No canonical PBS split/recombine H/V interferometer arms are used.

## Equivalence Results

- sequential pre-QWP overlap to abstract H/V synthesis: `1.000000000000`
- sequential post-QWP overlap to validated target: `1.000000000000`
- ideal sequential z=60 correlation to V0: `1.000000000000`
- realistic sequential z=60 correlation to V0: `0.993620949303`
- realistic strict class: `visual_hexagonal_field`
- realistic strict hexagon: `True`

Swap HWPs are required only for the same-panel-orientation implementation. The rotated/orthogonal
SLM2 route is valid if the LC-director orientation and mount geometry are confirmed on the bench.

## Source Audit

Every primary beam panel is recorded in `01_source_audit/mode2w_fix_numerical_source_audit.csv`.
The hero ideal-vs-realistic comparison uses N=1536 native numerical data. Interpolation is display-only;
metrics are computed on native arrays. N=384 data is not used for primary hero comparison figures.
The close-up beam panels in Figures 3A and 4A use scalable angular-spectrum (SAS) zoom rendering:
the audit records both the native input dx and the scaled output dx, so zoom fidelity is separated
from classifier/metric provenance.

## Corrected Power Model

The sequential power ledger removes split-arm H/V and PBS-recombination stages. It tracks a single
beam through preparation, SLM1, optional swap, SLM2, common 4F, QWP, axicon, z=60 and useful-region
power.

## Outcome

Outcome **M2WF-A**: Sequential physical implementation is numerically equivalent to the validated abstract H/V synthesis; the realistic sequential route preserves the strict hexagon; the power ledger is corrected; and the redesigned figure pack is readable and scientifically coherent.

No microfabrication/sample-plane success claim is made.

Output root: `outputs\figures\digital_twin\nathan_mode2w_fix_sequential_master`.
