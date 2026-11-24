# Bench correction — physical axicon measured at 10° (24 November 2026)

**Authoritative bench fact reported by the researcher:** the installed axicon's **physical/base angle is 10°**, directly measured. The prior 'Thorlabs 20° / AX2520' identification in the first-year report is not correct for the installed optic. The **part number remains unknown**, and the measurement method and uncertainty have not been documented; do not fabricate them.

## Consequences for the model
- Report wavelength 1029 nm (1030 nm commonly rounded). Using an *assumed* fused-silica n≈1.4501, Snell refraction gives cone half-angle θ=arcsin(n sin α)−α ≈ **4.58°** for physical α=10°. The prior α=20° gives θ≈9.73°.
- For an ideal scalar ℓ=20 mode, ring peak radius r≈j'20,1/(k sin θ) changes from **≈21.5 µm for 20°** to **≈45.5 µm for 10°**.
- Quality-screened archived camera-ring radii are approximately 46–52 µm. The independently described second camera/periscope relay has f1=f2=300 mm, hence *nominal* |M|=1. Therefore the 10° ring prediction is much closer. This is **one consistency check**, **not experimental validation** of the full axicon system and not proof of a manufacturer SKU.
- Finite-energy geometric Bessel length for assumed 2 mm illuminated radius changes from ~11.6 mm (20°) to ~25 mm (10°). Recorded axial scan positions and the exact camera z reference must be independently reconciled; beam size still assumed.
- The camera effective object-plane pixel sampling is nominally 5.5 µm (for |M|=1), so **the earlier 2.33× inferred camera magnification and all figures built on it must be removed/recomputed**. Do not claim detector sampling fully explains the fourfold rings until experimentally verified.
- Recompute all 20° downstream products: ideal z/radial profiles, beam-zone extent, annular-coupling error predictions, axicon obliquity, aperture/tip effects, camera forward model, inversion/correction interpretation, comparison to data, and captions/abstract/conclusions.
- SLM1 corrects/conditions upstream field; SLM2 adds freely selected charge and carrier. Historical +10/−10 split is an experiment, not the active design. F300 relay retains lens1–pinhole=300 mm, pinhole–lens2=300 mm; **lens separation 600 mm**. Separate camera/periscope f1=f2=300 mm relay is 1×.
- Pulse-timing measurements do not depend on axicon angle and should retain the original evidence caveats.

## Report edits
**Must edit**: Abstract, §2.1 Fig 2.1, §2.2 estimated ℓ=20 radius, §3.1–3.4 simulations/tables, §4.1/4.2 Table 4.1, §§4.3–4.6 all comparisons, Ch.5 telescope/camera sampling and ring reconciliation, Ch.7 Conclusions, Appendix A. Update bibliography [53]; do not cite 'AX2520' as the actual installed component.

**Critical rule:** A 20° simulation is still a correct simulation of an assumed 20° axicon; it is **not** a prediction for the installed 10° optic. Do not merely relabel old output. Preserve the old runs under an explicit historical/comparator tag and regenerate new 10° outputs from source with fresh numerical manifests.

## Code integration status
A new **reference hardware record** has been committed in `configs/hardware/confirmed_physical_axicon_2026-11-24.json`. It has not yet been connected to the accepted canonical hardware manifest or numerical running code, since doing so requires revalidation. These main-code changes belong to the Claude/Codex implementation and QA pass, not a silent replacement of accepted models.
