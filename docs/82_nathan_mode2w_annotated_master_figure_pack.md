# Nathan MODE 2W - Annotated Master Figure Pack

**Superseded:** this MODE 2W pack is not accepted as the publication/presentation pass. It is
replaced by `docs/83_nathan_mode2w_fix_sequential_architecture.md`, which corrects the physical
architecture to the sequential single-beam dual-SLM route and rebuilds the figure source audit.

**Status:** presentation / synthesis mode only. MODE 2W does not introduce new physics, does not
revive forbidden optima, and makes no microfabrication/sample-plane success claim.

Canonical source-scale operating point: `REALISTIC_4F_HEXAGON_REFERENCE`.
Secondary strict-eligible operating point: `strict_c6.75_i0.40_q-0.25_r+0.0_p0.00`.
Forbidden old optimum: `m2u2_opt_003_c5.75_i0.32_q-0.25_r+0.0_p0.10`.

## What each figure shows

1. `fig1_optical_system_annotated.*` - the full left-to-right source-scale bench with fixed,
   routine-calibration, and unknown/non-blocking items visually separated.
2. `fig2_target_and_masks_annotated.*` - the target sector field, Stokes/intensity views and the
   native SLM-H/SLM-V phase-mask package.
3. `fig3_ideal_vs_actual_outputs.*` - V0, ideal Jones route, realistic/canonical 4F route,
   moderate realism and one compensated recovery example under the repaired strict gate.
4. `fig4_propagation_and_power.*` - source-scale propagation through the Bessel region plus the
   canonical stage-by-stage power ledger.
5. `fig5_tolerance_correction_build_readiness.*` - tolerance/correction dashboard and first-day
   build-readiness status.
6. `figA1_appendix_metrics_and_provenance.*` - supplementary settings, truth table and provenance audit.

## Route interpretation

The ideal route is the M2P/M2N dual-linear-SLM Jones synthesis with an ideal final QWP. The actual
route is the realistic dual-SLM + carrier + common 4F + QWP + axicon chain. The canonical operating
point is the realistic 4F reference itself, `REALISTIC_4F_HEXAGON_REFERENCE`; the secondary strict
candidate remains `strict_c6.75_i0.40_q-0.25_r+0.0_p0.00`.

## Optical settings and masks

The source-scale bench uses 1029 nm, a 2 mm source beam radius, native 1920 x 1080 SLM panels at
8 um pitch, 6.25 lp/mm carrier, f=300 mm common 4F, +1 order at about 1.929 mm, iris diameter about
1.54 mm, QWP code angle -45 deg and a 2 deg n=1.458 axicon. The masks are panel-space wrapped phase
arrays; preview PNGs are not calibrated hardware masks and the per-panel LUT still comes from docs/75.

## Output differences

V0 and the ideal route are visual/reference controls; the realistic/canonical route is the
strict-eligible operating point under the repaired candidate gate. The moderate realism and
compensated recovery rows are shown explicitly so a high full-field correlation cannot hide a failed
candidate gate, C3/triangular drift, dark-core growth or fourfold/X-like failure.

## Propagation and power

The propagation panel shows xy slices, x-z/y-z centre maps, on-axis intensity, ring peak and useful
power across z. The power panel is a normalised model ledger with 1 W and 10 W linear examples only;
it is not a damage-threshold or power-rating claim.

## Tolerances and correction

The dashboard summarises M2S single-parameter and combined-case tolerances, and M2V closed-loop
correction rows. The main practical tolerance is hologram/mask-to-axicon centring; the correction
mechanism is measured-image-driven digital recentring plus bounded low-order/piston/sector updates.

## Build-ready conclusion

Outcome **M2W-A**: the figure pack presents the validated source-scale branch clearly enough for a
report, slide deck or lab handoff. Source-scale build authorised: `True`.
Remaining calibrations are routine: `SLM LUT/stroke, exact iris/focal confirmation, camera scale/z-stage, parity sign, QWP mount sign, beam centring`.

Output root: `outputs\figures\digital_twin\nathan_mode2w_annotated_master`.
