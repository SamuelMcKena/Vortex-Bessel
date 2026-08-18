# q=20 Bessel z-scan: phase-reconstruction physics lock

## Why this change exists

The previous presentation material mixed three different objects:

1. **measured intensity** from the 18-plane BeamGage BMG z-scan;
2. **per-plane modal fits / inferred annular phase**;
3. **a 2-D correction-map visualisation** made by interpolating measured z-order directly onto a normalized radius.

That third object was too easy to read as a physically reconstructed SLM phase map. It was not. It also produced visually wild phase structure because wrapped phases from separate z planes were being treated as if their row-to-row phase were directly measured.

The authoritative interpretation is now the one in `q20_phase_physics.py` and `rebuild_q20_presentation_from_bmg.py`.

## The physical model

For an axicon-generated Bessel beam, the stationary-phase relation maps a measured focal-line plane to an annulus of the input field. In the notation of Miao *et al.* (Optics Express **30**, 11360–11371, 2022),

\[
\rho_z = z\tan\alpha,
\qquad
k_\perp = k\tan\alpha
\]

in the paraxial limit. The code uses the exact k-vector relation

\[
k_z = \sqrt{k^2-k_\perp^2},
\qquad
\tan\alpha = \frac{k_\perp}{k_z}
\]

when converting z spacing to annulus spacing.

The key consequence is:

> **z is measurement diversity for one transverse input wavefront. It is not a second, independent longitudinal correction phase.**

Each measured z plane constrains the residual phase on a different input annulus. Those annular estimates can be assembled into a transverse residual

\[
\psi_{\mathrm{res}}(\rho,\theta),
\]

and an SLM/deformable-mirror correction is its transverse conjugate

\[
\phi_{\mathrm{corr}}(\rho,\theta)=-\psi_{\mathrm{res}}(\rho,\theta).
\]

Longitudinal behaviour then follows from propagating the corrected transverse complex field.

## What is and is not recoverable from the present intensity stack

The current q=20 modal model factors the programmed vortex out of the residual fit. Therefore the correction must **not** contain the target `q*theta` phase.

Independent intensity fits at each z plane also have an arbitrary global phase (piston). Consequently the present passive intensity stack does not determine the axisymmetric annulus-to-annulus radial piston. The new code makes that ambiguity explicit and removes row piston only as a gauge choice for visualising the **non-axisymmetric residual**. It does not call that gauge a measured radial phase.

This is consistent with the Bessel-specific retrieval literature: the measured focal line samples different input annuli and the full input-aperture aberration is assembled from those annular retrievals. A particular z alone constrains only the corresponding annulus.

## What changed in code

### `q20_phase_physics.py`

- converts z spacing to annulus spacing using conical-ray geometry;
- distinguishes **relative** annulus radius from an absolutely calibrated radius;
- aligns the unobservable annulus piston explicitly;
- interpolates phase through complex unit phasors rather than interpolating wrapped phase values;
- assembles only the **residual** transverse phase, never the programmed q=20 vortex;
- marks radial piston as unrecovered;
- hard-blocks any claim that the result is hardware-ready.

### `rebuild_q20_presentation_from_bmg.py`

The presentation replacements are generated from the complete raw acquisition and are provenance-checked:

- exactly 18 measured z planes;
- exactly 4 BMG repeats per plane (72 files total);
- registered repeat average at each plane;
- measured-only x-z and y-z figures;
- residual-phase figure with q=20 removed and z treated as annular/radial diversity;
- a single-transverse-phase forward-model figure.

The final forward-model panel is deliberately constructed as:

1. lab measured intensity;
2. ideal model **plus the retrieved transverse residual**;
3. the same model **after its transverse conjugate correction**.

There is no independently invented longitudinal correction. All z evolution comes from propagation.

## Presentation language

Use the following labels:

- **Measured q=20 BMG stack**
- **Measured x-z / y-z evolution**
- **Retrieved non-axisymmetric residual phase (model-inferred)**
- **Single-transverse-phase forward-model diagnostic**

Do **not** label the current inferred phase as:

- direct phase measurement;
- measured SLM correction;
- experimentally validated before/after correction;
- full radial phase;
- hardware-ready correction.

## Remaining calibration before applying a mask to SLM2

A hardware correction still requires, at minimum:

- absolute mapping between camera z and the sampled input annulus;
- relay magnification from the relevant input/axicon plane to SLM2;
- camera-to-SLM rotation and parity;
- beam footprint / centre on SLM2;
- SLM phase LUT at 1030 nm;
- low-gain application followed by a new 18-plane measured z-scan and acceptance-gate comparison.

Until those are complete, generated correction files remain diagnostics/model predictions.

## Reference

B. Miao, L. Feder, J. E. Shrock, and H. M. Milchberg, “Phase front retrieval and correction of Bessel beams,” *Optics Express* **30**(7), 11360–11371 (2022). DOI: 10.1364/OE.454796.
