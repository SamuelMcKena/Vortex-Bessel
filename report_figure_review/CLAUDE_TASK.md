# Claude: replace the defective first-year PhD figure pack

This is an implementation/reconstruction task. Do NOT simply polish an existing unvalidated 34-figure package. It has ten BLOCK figure failures and 21 further figures needing revision; only three explanatory schematics are pass-with-caption. See the register and draft captions in this branch.

**Priority 0 — correctness:** F2.1 has no visible Gaussian; z_R≈0.047 mm while the old plot spans 0–50 mm. F3.1 numerical/theory disagreement and sampled Nyquist guard failure. F4.13/4.14 use synthetic toy inverse examples but sound like validated full-system optical diagnoses. F5.3 uses disconnected quality-selected ring centres to fit trends. F6.1 contains schematic trace even though raw 11 Nov scope arrays have now been found. Do not represent unsupported output as empirical.

**Priority 1 — readable figures:** F3.6 vector-beam montage too small at 160 mm; F4.1 and F4.3 text collisions; F4.12 small labels, wrong aberration units. Check all 34 individually at 160 mm in both PDF and PNG, not only a contact sheet.

**Scientific modelling:**
- Intended bench architecture: SLM1 correction + propagation → SLM2 selectable vortex/carrier → first-order 4f filter → physical axicon → camera/sample. Historical split-charge studies must be identified as such.
- Lens spacing is 600 mm, central Fourier pinhole midway, 300 mm from each lens. Other SLM/relay positions remain nominal if unmeasured.
- Axicon angle is unresolved; the measured charge-20 ring 46–50 µm is NOT a part-number calibration. Do not tune angle or pixel scale to force agreement.
- A 0.75 **mrad** pointing case is not the same as 0.75 degrees. The plotted oblique-axicon ray model is not a propagated field; compare properly against Thaning/Hu papers.
- Separate scalar analytical references, simplified wave-optics models, measured BMG data, retrieved modal inference, and PHAROS electrical detector traces.

**Experimental figure recovery:** The 11 November original archive `Laser_Lab_Results_2026-11-11.zip` is available in the ChatGPT repair package and contains measurements M1/M2c/M5a/M5b/P1, J1+J2 60 timing samples, and 12 original trace images. Use original CSVs, not decorative invented waveforms. Timing definition: CH1 LX8 falling 2.5 V threshold to CH2 50% rise, on 100 ns exports for J1/J2. Compare to 25 kHz period only as an interpretation, not proof. Keep 16 November detector-envelope conclusions preliminary unless the raw ZIP is supplied and inspected. The provided share link `https://chatgpt.com/share/6b0499f0-a9f4-83ed-8e41-d12156980846` should be reviewed by Claude if accessible; it could not be retrieved by the text web fetch in this session.

**Required deliverables:** Rebuild chosen core figures, not a forced 34-figure quota. For each: true print-size PDF (160 mm full width) and 600-dpi PNG; source; separate source-data/provenance manifest with SHA-256 and numbers; captions with panel definitions/normalisations/uncertainty/limitations; tests of validated propagation and sampling; clean full-size figure book/contact sheet; an explicit review checklist. Separate fully supported from illustrative figures in a strict acceptance register. No synthetic experiment-data substitutions. Do not overwrite main or claim real hardware validation.

**Code and data in GitHub:** This branch contains the governing audit/captions and existing main repository model/data. The binary old figure pack and raw 11 Nov archive are supplied as a downloadable ZIP in the ChatGPT conversation, not presently committed to GitHub. Incorporate them using the included upload script before running a full reproducibility build, or use the attachment directly. No need to chase unavailable GUI screenshots or 16 Nov inputs if nonessential; omit optional figures.
