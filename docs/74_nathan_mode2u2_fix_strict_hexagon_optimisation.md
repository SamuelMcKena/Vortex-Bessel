# Nathan MODE 2U2-FIX - Strict Hexagon Optimisation

**Status:** optimisation integrity correction. MODE 2U3 is paused unless the
outcome below explicitly authorises it.

## Root Cause

The old best-compromise candidate `m2u2_opt_003_c5.75_i0.32_q-0.25_r+0.0_p0.10` reached
`corr_full = 0.9890` and legacy correlation
`0.9891`, but it failed the new strict
eligibility gate:

`candidate drifted from immutable realistic-4F reference`.

The old optimiser treated shape as a soft score. Full-field correlation and
useful-region/peak terms could therefore compensate for visible drift from the
realistic 4F hexagon. The correction makes hexagon preservation a hard
constraint before peak or power ranking.

## Correlation Audit

Old compromise metrics:

- full field correlation: `0.9890`
- focus crop correlation: `0.9866`
- V0-support correlation: `0.9863`
- useful-region correlation: `0.9878`
- angular correlation: `0.9963`
- correlation to immutable realistic 4F reference: `0.9942`

Full-field correlation was not allowed to drive the new selection by itself.

## Classifier Change

The new eligibility gate requires:

- legacy visual hexagon pass;
- no triangular/C3 veto;
- no fourfold/X veto;
- sixfold harmonic floor;
- focus, V0-support, angular and radial correlations above threshold;
- `c120 - c60 <= -0.02`;
- dark-core ratio below `0.01`;
- correlation to the immutable realistic 4F reference above `0.997`.

The fourfold metric is present, but calibration showed that h4/h6 alone does not
separate every suspect candidate from the realistic reference. The decisive
guard is the reference-drift veto plus crop/support/angular requirements.

## New Strict Optima

- strict best shape: `REALISTIC_4F_HEXAGON_REFERENCE`
- strict best peak: `REALISTIC_4F_HEXAGON_REFERENCE`
- strict best useful energy: `REALISTIC_4F_HEXAGON_REFERENCE`
- strict best compromise: `strict_c6.75_i0.40_q-0.25_r+0.0_p0.00`

All reported strict optima pass `strict_hexagon_eligible = true`.

## Calibration

Calibration rows written: `6`. V0 and realistic 4F pass;
triangular and h4/fourfold synthetic controls fail; the old compromise fails.

## Outcome

**M2U2F-B.** The optimiser/classifier issue was corrected, but only a strict eligible subset may be used; discard non-hexagonal old optima.

M2U3 authorised: `true`.

Output root: `outputs/figures/digital_twin/nathan_mode2u2_fix_strict_hexagon_optimisation`.
