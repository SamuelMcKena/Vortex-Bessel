# q20 recovery recipe

Quick recovery is the first serious application of the general framework. Its
defaults are configurable, with current useful starting planes of 31, 38 and
45 mm and at least three repeats.

## Sequence

1. Confirm physical axicon `IN`, z reference, planes, repeats and fixed camera settings.
2. Put q=0 on SLM1 and capture identical near/middle/far planes.
3. Put q=20 on SLM1 and capture the same planes.
4. Fit both beam walks and compare their slope vectors.
5. Run configured low-order signed sweeps; current suggested order is
   `astig_x`, `coma_y`, `coma_x`, then `astig_xy` when justified.
6. Acquire a fresh verification capture at the recommended command.
7. Accept only when verification criteria pass; otherwise restore the baseline.
8. Update calibration evidence and re-evaluate readiness.

Similar q=0 and q20 slopes are classified primarily as common post-axicon or
measurement geometry. A material difference flags a possible vortex/SLM/asymmetric
contribution. Neither classification uniquely proves cause.

## Recovery escalation

- **Level 1 — quick:** three planes, q=0/q20 comparison, relevant low-order modes,
  fresh verification.
- **Level 2 — short:** 3–5 richer planes and a calibrated residual-map update when
  quick recovery fails.
- **Level 3 — full commissioning:** dense propagation/retrieval after major relay,
  camera, SLM or optical changes, or failure outside the known correction space.

## Headless replay demonstration

From the repository root:

```bash
python lab_gui/run_labcontrol_cli.py run recover-q20 \
  --camera replay \
  --output outputs/q20_recovery_demo
```

With no external replay root, this generates deterministic numeric matrices and
then sends them through `ReplayCameraProvider`. Every result is labelled
`SYNTHETIC`, the SLM cast status is `NOT_CAST_SYNTHETIC_DEMO`, and readiness is
scoped to the demo only.

To use a compatible archived dataset:

```bash
python lab_gui/run_labcontrol_cli.py run recover-q20 \
  --camera replay \
  --replay-root D:\measurements\recovery_fixture \
  --output D:\measurements\analysis
```

Recorded-data readiness is analysis scope only; it does not assert that today's
bench matches the archived hardware state.
