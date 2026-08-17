# Nathan MODE 2U - Master High-Resolution Audit And Build Plan

**Status:** source-scale high-resolution audit and build-planning layer. It does
not change the validated MODE 2P/2N/2Q/2S physics and it makes no
microfabrication/sample-plane success claim.

## Regeneration

The audit regenerated high-resolution PNG figures under
`outputs/figures/digital_twin/nathan_mode2u_master_highres_audit` for V0, M2P, M2N, M2Q, M2S, historical MODE 1
contrast, energy ledgers, profile response, optimal-hexagon sweep, high-N focus
panels, and the realistic build plan. Rendered beam panels use focus crops and
`lanczos` interpolation so the sub-mm structure is not
lost in the 10 mm source window.

The file `publication_sampling_audit.csv/json` separates numerical Nyquist
adequacy from publication-grade visual adequacy. In this model the axicon radial
fringe period is about 64 um, so grids near N=1536 on the 10 mm source window are
the first practical publication-recommended setting by the stricter samples per
fringe and ring-diameter-pixels criteria.

## Source-Scale Visual Outcome

The source-scale branch remains visually and metrically confirmed. MODE 2S
outcome is `M2S-A`. The SLM active window does not
fit the 10 mm source field vertically (`window_fits_vertically =
False`), but the 2 mm Gaussian beam clips
only `1.459e-05` of its power.

The microfabrication/sample-objective branch remains a separate negative
contrast branch and is not unblocked by MODE 2U.

## Energy And Sensitivity

The dominant explicit power loss in the practical source-scale route is the 4F
first-order selection/rejected order budget. The most dangerous shape
misalignment remains hologram/axicon axis registration; bounded recentering is
the practical correction.

## Optimal Operating Points

- Best shape: `opt_146_c6.75_i0.52_q-0.25_r+0.0_p0.00` with shape score `0.9723`.
- Best power: `opt_146_c6.75_i0.52_q-0.25_r+0.0_p0.00` with power score `0.9238`.
- Recommended compromise: `opt_146_c6.75_i0.52_q-0.25_r+0.0_p0.00` with combined score `0.9529`.

## High-N Confirmation

clean_source_v0_reference N=1024 corr=1.0000 class=visual_hexagonal_field; realistic_dual_slm_4f_baseline N=1024 corr=0.9936 class=visual_hexagonal_field; moderate_combined N=1024 corr=0.9437 class=visual_hexagonal_field; compensated_0p5mm_axicon_decentre_seeded_recentre N=1024 corr=0.9807 class=visual_hexagonal_field; best_shape_opt_146_c6.75_i0.52_q-0.25_r+0.0_p0.00 N=1024 corr=0.9970 class=visual_hexagonal_field; best_power_opt_146_c6.75_i0.52_q-0.25_r+0.0_p0.00 N=1024 corr=0.9970 class=visual_hexagonal_field; best_compromise_opt_146_c6.75_i0.52_q-0.25_r+0.0_p0.00 N=1024 corr=0.9970 class=visual_hexagonal_field; publication_clean_source_v0_reference N=1536 corr=1.0000 class=visual_hexagonal_field; publication_realistic_dual_slm_4f_baseline N=1536 corr=0.9936 class=visual_hexagonal_field; publication_best_compromise_opt_146_c6.75_i0.52_q-0.25_r+0.0_p0.00 N=1536 corr=0.9970 class=visual_hexagonal_field

## Build Recommendation

Recommended route: **dual-SLM + carrier + shared 4F first-order iris + QWP + source-scale physical axicon**.

The six-polarizer/segmented-polarizer route is not needed for the source-scale
implementation. It is replaced by the dual-SLM carrier/4F route, a uniform QWP,
and a source-scale physical axicon. The practical next stage is
`MODE 2T/2V lab implementation package: export SLM masks, waveplate settings, 4F stop sizing, and alignment workflow`.
