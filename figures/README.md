# Canonical figure library

This directory contains the **current preferred visual outputs** for Vortex-Bessel. It is curated rather than being a dump of every historical render.

## Rule

If multiple historical versions exist, prefer the figure listed here and use `manifest.csv` to trace its generator and source. The old repository remains the provenance archive.

## Layout

- `publication/` — selected reusable outputs from maintained Publication_Study workflows.
- `presentation/` — high-resolution presentation-ready views where available.
- `experimental/q20_aberration/reconstruction/` — current realigned modal reconstruction.
- `experimental/q20_aberration/validation/` — all-z / Cartesian / 3-D diagnostics.
- `experimental/q20_aberration/phase_error_recreation/` — phase-error recreation/falsification.
- `experimental/q20_aberration/single_mask/` — stricter single-input-mask forward tests.
- `experimental/q20_aberration/closed_loop/` — iterative controller diagnostics.
- `experimental/q20_aberration/slm_preview/` — uncalibrated SLM previews; not hardware-validated correction.

## q=20 claim boundary

Measured z-stack images are experimental evidence. Modal phase recovery and proposed correction maps are model inference. Phase-error recreation and single-mask forward propagation are falsification tests. SLM previews are not hardware-ready until phase LUT/stroke, illuminated footprint, parity/rotation and camera-to-SLM mapping are calibrated and a fresh post-correction z-stack verifies the result.

Pre-realignment q=20 renders and early `postcorrectionoutput1`-style figures are deliberately not promoted here.

`outputs/` remains the regeneration/transient area; `figures/` is the small committed library for navigation, comparison and presentation reuse.
