# Stage 8.8 Visual Usability and Physics Review

This review was added after the Stage 8.7 quick-look work because execution success was not enough: the project must also be usable for live parameter exploration and the figures must not visually overclaim the physics.

## Verdict

**Overall status: Amber.**

The core equations are still governed by the locked scalar, lab-realism, vector, materials, capsule and advanced stages. No new core physics model is introduced here. The main remaining risk is visual communication: many historical figures in `outputs/figures/*` are diagnostic, legacy, or exploratory. They should not be interpreted as final publication figures simply because they exist in the output tree.

## What was changed in this pass

1. A shared notebook-control helper was added at `vbb_study/publication/notebook_controls.py`.
2. Each non-quicklook notebook now exposes an explicit **Editable Notebook Controls** block near the top.
3. The quick-look notebook remains the main live parameter-control notebook.
4. A visual review manifest was written to `outputs/csv/review/visual_review_manifest.csv`.
5. A quick-look visual contact sheet was written to `outputs/figures/review/quicklook_visual_contact_sheet.png`.

## Important boundary

The editable controls added to the stage notebooks are an explicit user interface layer. They do not automatically rewrite the locked physics or reroute all downstream cells. This is intentional: the canonical stage notebooks are still reproducible stage studies, while the quick-look notebook is the correct day-to-day interactive beam-to-sample control surface.

For exploratory parameter work:

- use `notebooks/quicklook/00_quick_beam_to_sample_simulator.ipynb` first;
- change `CONFIG` inside the notebook;
- inspect SLM phase, ideal/conical previews, lab-realistic previews, through-sample maps and material proxies;
- only then decide whether a locked stage notebook needs a controlled rerun.

For canonical publication-style results:

- use the stage runner and canonical CSVs;
- do not use legacy/quarantined figures;
- do not hide fail/marginal QA labels;
- do not use quick-preview figures as final evidence.

## Current quick-look visual interpretation

The current quick-look visual outputs are diagnostic but much healthier than the earlier broken first-order-filter case.

The earlier failure was caused by an invalid blaze/filter geometry: the carrier was too close to the conical spectrum, the effective first-order selection collapsed to roughly `6e-4`, and the lab-realistic maps became artefacts. The current known-good quick-look defaults use a shorter blaze period and larger grid so the selected fraction is no longer near zero.

The lab-realistic vortex XY map can appear as a clean annulus. This is not automatically wrong: a vortex Bessel-like beam is expected to have centre suppression and an annular high-intensity region. However, for `ell=0` lab-realistic cases a dark-centred annulus is not a valid bright-core Bessel result and is now labelled by the visual sanity guardrails.

## Figures that are not publication-ready by default

The following output families should be treated carefully:

- old `NB_*` outputs;
- old `stage_f`, `stage_h`, `stage_h2`, `stage10_discrete` outputs;
- old polygonal/hex outline exploratory figures;
- quick-look figures unless manually promoted after review;
- material/capsule figures unless proxy caveats are visible;
- vector figures unless the current-lab versus future-hardware boundary is visible;
- any figure generated from quick-preview settings where sampling or propagation-power QA is marginal/fail.

The figure registry and caption gate remain the authority for final-export use.

## Notebook usability model

The notebooks now follow a two-layer model:

1. **Stage notebooks** — reproducible, governed studies. They expose controls for visibility and local exploration, but canonical regeneration should still use the runner.
2. **Quick-look notebook** — live exploratory beam-to-sample simulator. This is the primary place to change parameters on the fly.

This is preferable to making every stage notebook fully interactive, because some stage notebooks are designed to regenerate fixed canonical CSVs. Making every one of them behave like an open-ended playground would risk breaking reproducibility.

## Recommended next work before final publication export

1. Manually review the quick-look contact sheet and decide which plots are visually useful.
2. Regenerate any final publication figures from canonical CSVs only.
3. Build final export figures from the registry allow-list, not by sweeping `outputs/figures`.
4. Create final polished figures with:
   - readable sizes;
   - colourbar labels;
   - units;
   - QA labels;
   - caveat captions;
   - consistent colour scales where comparisons matter.
5. Keep quick-look outputs diagnostic unless explicitly promoted.

## Non-negotiable caveats

- Material and capsule outputs remain planning proxies unless experimentally calibrated.
- True radial/azimuthal vector generation remains future-hardware-required under the current bench assumptions.
- Focal-plane polygonal or hexagonal patterns are not automatically propagation-stable beams.
- Display interpolation improves readability only; it does not increase numerical accuracy.
- Propagation-power fail/marginal labels must remain visible.
