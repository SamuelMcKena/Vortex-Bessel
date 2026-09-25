# Report Figure Physics Audit

The report-facing route is the accepted collinear sequential chain:

`Gaussian -> POL1/HWP -> SLM1 -> conditional swap HWP -> SLM2 -> optional swap-back -> common 4F -> QWP -> physical axicon -> camera`.

The older PBS split/recombine architecture and the Stage-6 scalar/vector
fidelity ladder are superseded and must not be used as current evidence.
`figures/report_figure_audit.csv` records the disposition of the affected
figures. `tools/render_report_vector_figures.py` generates the two retained
report figures: the current architecture and analytic Jones-field analyser
observables. Neither makes a sample-plane or material-processing claim.

## Corrected scalar source-scale comparison

Run `python tools/render_matched_scalar_figures.py --output-dir outputs/figures/matched_scalar_audit`.
This regenerates B0, V1 and V3 from the same Phase-2A Gaussian input,
SLM/4F/pupil route and physical-axicon *model* at z=60 mm. The ideal and
nominal-device branches differ only in modelled SLM terms. It refuses the
10 mm/N=512 source grid (3.29 samples per radial period); N=1024 has 6.59,
and valid scalable-angular-spectrum propagation has 20.84 samples per
period at the output. The rendered ROI is in physical millimetres. The CSV
records both grid spacings, first-order fractions, retained power and
shape overlap. A small difference is expected: the selected device model
treats fill factor as throughput, **not** pixel-border diffraction.
No measured LUT, true pixel aperture, calibrated axicon angle or sample
interface is in this comparison. Consequently the figure is a verified
source-scale simulation and is **not** evidence that the physical bench
will reproduce a specific morphology.

The nominal Phase-2A manifest uses a 2-degree *base angle* for the source
parity case, whereas the physical optic is described as a 20-degree axicon.
The manufacturer's angle convention and optic identity must be checked
before transferring these predictions to the actual lab. Calling the
throughput-only branch a validated real-lab output is unsupported.

## Polygonal and hexagonal panels

The attached Publication Study distinguishes the finite-direction
plane-wave lattice from the localized continuous-k-ring polygonal model.
Do not interpret a six-wave lattice, sixfold brightness modulation, a
single-plane contour or a phase-mask preview as a propagation-stable hollow
hexagon. The Stage 15--17 panels remain candidate diagnostics until a
matched, adequately sampled sequential two-SLM/HWP/QWP propagation and
held-out axial assessment are regenerated. The optical architecture
schematic and analytic Jones atlas are safe to use as *theory*; neither
establishes a particular hexagonal optical result or material channel.

The legacy Mode-2Q screen can label a low-resolution sixfold image as
`visual_hexagonal_field` when the inferred ring radius spans only about
4.3 pixels at N=384. That label is retained for audit provenance, but the
new `report_sampling_ok` and `report_eligible_hexagon` flags forbid its
use as report evidence unless the ring radius reaches at least 12 pixels.
This gate is a necessary sampling check, not proof of propagation stability;
the high-resolution sequential-vector validation remains outstanding.
