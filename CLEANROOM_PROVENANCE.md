# Cleanroom provenance

## Purpose

This tree is intended to become the clean long-lived PhD structured-beam repository rather than another historical phase branch. The old repository remains the provenance/history store; this tree contains the current source and a deliberately reduced evidence set.

## Source branches

- **Validated numerical base:** `phase2k-mathematical-physics-output-audit` at `6712dd0724702f4e2e8fa56f5062b6e564328e00`.
- **Experimental axicon correction:** `agent/axicon-aberration-correction-study` at `4cc41b97a1e72c79e09ce0da9bd3210da6bf4718`.

The aberration branch diverged before Phase 2K, so it was not used as the repository base. Instead, only its experimental axicon-correction subtree was transplanted onto the Phase 2K base. This preserves the Phase 2K mathematical/physics audit while incorporating the latest measured q=20 work.

## Axicon-correction figure cleanup

The original figure snapshot duplicated several generations of the same analysis in `figures/modal_q20`, `figures/root` and `figures/slm_closed_loop_alignment/modal_q20`.

The clean tree replaces that snapshot with `figures/current_q20`, retaining:

- current realigned q=20 Cartesian, radial, longitudinal and 3-D outputs;
- modal spectrum and annular-aberration phase;
- comprehensive all-z validation;
- phase-error-recreation/falsification outputs;
- single-mask inverse-forward tests;
- measured beam-axis diagnostics;
- iterative closed-loop gain-selection outputs;
- SLM2 preview outputs, still explicitly marked as nominal/not hardware-ready.

Pre-realignment duplicates and the earliest root-level `postcorrectionoutput1`/aberration-phase figures are not part of the current figure set.

## Claim boundaries

The Phase 2K audit remains authoritative for numerical claim maturity. Pre-Phase2K generated outputs are not automatically promoted simply because they are visually newer or convenient. A figure is current scientific evidence only when its producer is consistent with the corrected physics and its required gates have passed.

For the q=20 experimental correction work:

- the measured z-stack is experimental evidence;
- the modal reconstruction and phase-only correction are model inference;
- the retrieved phase currently does not reproduce the complete laboratory angular/fan error structure under the stricter inverse-error checks;
- an SLM command is not hardware-ready until the SLM LUT/phase stroke, beam footprint, coordinate parity/rotation and camera-to-SLM transform are calibrated;
- model-predicted improvement never substitutes for a fresh post-correction z-stack.

## Cleanup rule

Historical code is not deleted from the original repository. The clean tree simply does not carry forward redundant phase summaries, temporary runner files, bulk generated output trees or superseded figure generations unless they are required by the current code or evidence chain.
