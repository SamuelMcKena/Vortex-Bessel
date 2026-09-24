# Report Figure Physics Audit

The report-facing route is the accepted collinear sequential chain:

`Gaussian -> POL1/HWP -> SLM1 -> conditional swap HWP -> SLM2 -> optional swap-back -> common 4F -> QWP -> physical axicon -> camera`.

The older PBS split/recombine architecture and the Stage-6 scalar/vector
fidelity ladder are superseded and must not be used as current evidence.
`figures/report_figure_audit.csv` records the disposition of the affected
figures. `tools/render_report_vector_figures.py` generates the two retained
report figures: the current architecture and analytic Jones-field analyser
observables. Neither makes a sample-plane or material-processing claim.
