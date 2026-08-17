# Nathan MODE 2V - Full Jones Build Derivation (docs/79)

All matrices act in the beam-local linear H/V basis; angles are anticlockwise from +x (H) in the
receiver view (standing downstream looking back into the beam); beta marks the FAST axis;
`R(b) = [[cos b, -sin b], [sin b, cos b]]`, retarder `J(d, b) = R(-b) diag(e^{-id/2}, e^{+id/2}) R(b)`.
Every mirror/SLM reflection flips transverse parity; per-arm odd totals are absorbed as a software
x-flip of that arm's mask (STAGE 6 test), so the chain below is written in one consistent frame.

## Stage-by-stage chain

1. **Laser**: `E0 = A(r) [cos(psi), sin(psi)]^T` - linear, orientation psi measured on day one.
2. **Input polariser (POL1)**: projects onto H: `E1 = A(r) cos(psi) [1, 0]^T` (defines the reference axis).
3. **HWP #1 at b1**: `J_HWP(b1) = R(-b1) diag(-i, +i)... = ` rotation of linear polarisation by `2 b1`;
   set `2 b1` so PBS #1 splits 50/50: `E2 = (A/sqrt(2)) [1, 1]^T` (up to global phase).
4. **PBS #1**: H transmits into the H arm, V reflects into the V arm:
   `E_H,in = (A/sqrt(2)) |H>`, `E_V,in = (A/sqrt(2)) |V>` (reflection pi bookkept as arm piston).
5. **H arm / SLM-H**: panel director along H (phase-only): `E_H,out = (A/sqrt(2)) e^{i(+alpha + carrier)} |H>`.
6. **V arm**: HWP #2 at 45 deg rotates V -> director; SLM-V applies `e^{i(-alpha + pi/2 + carrier)}`;
   HWP #3 at 45 deg rotates back to V: `E_V,out = (A/sqrt(2)) e^{i(-alpha + pi/2 + carrier)} |V>`.
   (If SLM-V is mounted rotated 90 deg, HWP #2/#3 are omitted and the algebra is identical.)
7. **PBS #2**: coherent recombination (paths matched within the ~260 fs coherence length):
   `E3 = (A/sqrt(2)) [e^{i(+alpha)}, e^{i(-alpha + pi/2 + delta)}]^T e^{i carrier}` with arm piston `delta`.
8. **Common 4F + iris**: selects the +1 order of BOTH channels and removes the carrier; sector-tail
   clipping costs ~5% power but preserves the phase structure (validated 0.9936 correlation).
9. **QWP at code -45 deg**: `J_QWP(-45) = (1/sqrt(2)) [[1, -i], [-i, 1]]`. Then
   `Ex = (A/sqrt(2))(e^{i alpha} + e^{i(-alpha + delta)}) = A sqrt(2) e^{i delta/2} cos(alpha - delta/2)`
   `Ey = (A/sqrt(2))(-i e^{i alpha} + i e^{i(-alpha + delta)}) = A sqrt(2) e^{i delta/2} sin(alpha - delta/2)`
   i.e. exactly the segmented target `A [cos alpha', sin alpha']` with `alpha' = alpha - delta/2`:
   the arm piston delta only rotates every local polarisation uniformly, and M2S proved the intensity
   observable is invariant to it - so delta is free, not a build tolerance.
10. **Axicon**: radial/azimuthal p/s Fresnel + conical phase produces the hexagonal Bessel zone.

## Conventions tracked

- global phase: irrelevant; relative H/V phase: pi/2 target offset carried by the SLM-V mask.
- coordinate flips: per-arm reflection parity -> software mask x-flip (STAGE 6).
- QWP physical statement: code -45 deg means: the QWP FAST axis lies 45 deg from the lab-horizontal H axis, rotated CLOCKWISE when you stand downstream and look back into the beam (equivalently 45 deg anticlockwise when looking along the propagation direction from behind the source)
- mount-side caveat: any odd number of upstream mirror reflections after the recombiner flips the apparent sense; the sign is therefore fixed on the bench by ONE polarimeter check: with only the H channel open and a uniform mask, the QWP at the correct -45 deg turns H into LEFT-hand circular in receiver view (fast axis clockwise-45 deg); if right-hand circular is observed, rotate the QWP to +45 deg
- fast/slow: beta marks the FAST axis in this codebase (exp(-i*delta/2) on the beta axis).
