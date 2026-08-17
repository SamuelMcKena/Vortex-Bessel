# Nathan MODE 2V - First-Day Lab Procedure (docs/80)

Source-scale bench only; no material processing; no microfabrication claim.

## STAGE 0 - safety / low-power setup
Alignment power only; verify 1029 nm-rated eyewear for every person in the room; interlocks checked;
do NOT run at processing power at any point in this procedure.

## STAGE 1 - Gaussian beam preparation
Verify the 1029 nm output; measure the 1/e field radius (target 2 mm, telescope if needed); centre and
collimate; record the beam profile and polarisation orientation/purity (B_INPUT_BEAM).

## STAGE 2 - SLM panel orientation test
One polariser + displayed grating per panel: find the LC director (maximum first-order diffraction,
minimum amplitude modulation). DECIDES whether V-arm HWP #2/#3 are used or SLM-V is mounted rotated
90 deg. Also run the asymmetric-pattern ('F') test per panel to record flips/parity.

## STAGE 3 - per-SLM phase calibration
Run docs/75 (interferometric or binary-grating) per panel at 1029 nm; store the LUT with a calibration
ID; acceptance: usable stroke >= 2 pi (or validated wrapped mapping), residual RMS <= 0.05 rad.

## STAGE 4 - H/V split
Insert POL1 + HWP #1 + PBS #1; rotate HWP #1 until the two arm powers are equal (M2S tolerates 0.8-1.2).

## STAGE 5 - SLM-H alone
Display the 20-px blaze only; verify a single dominant +1 order; measure first-order efficiency
(model reference ~0.95 x fill-factor effects).

## STAGE 6 - SLM-V alone
Repeat STAGE 5 in the V arm; confirm parity/orientation (apply the software x-flip if the arm has an
odd reflection count).

## STAGE 7 - 4F alignment
Locate the zero order; locate the +1 order; verify the displacement is close to 1.929 mm (this also confirms f = 300 mm); place the ~1.54 mm iris centred on +1; sweep the radius and record the efficiency
plateau (docs/76 steps 8-11); record the measured geometry into the hardware binding.

## STAGE 8 - recombination
Align PBS #2; overlap arm centres and magnification; match path lengths well inside the ~260 fs
coherence length (white-light/fringe-visibility check); confirm stable fringes between arms.

## STAGE 9 - QWP sign calibration
Polarimeter check (docs/78 Q13): open the H channel only with a uniform mask; at the correct code
-45 deg setting the output is LEFT-circular in receiver view; if right-circular, use +45 deg.

## STAGE 10 - pre-axicon validation
With both masks displayed, project onto H/V, D/A and R/L and compare with the predicted segmented
vector field (Stokes responsibility, docs/78); verify the pi/2 sector offset structure.

## STAGE 11 - insert axicon
Centre the hologram/vector singularity on the cone axis; blind placement tolerance <= 0.2 mm (M2S);
then measure and digitally recentre the masks (the single alignment that actually matters).

## STAGE 12 - camera z scan
Scan ~10-200 mm including the exact 60 mm reference plane; record xy planes and assemble x-z/y-z maps;
plane-placement tolerance is +/-20 mm (M2S), so mm-class stage steps are fine.

## STAGE 13 - correction
Measure centre/symmetry errors on the camera; apply the bounded closed-loop corrections (mask centre,
V piston, sector rotation/duty, sector pistons, low-order Zernikes) per the MODE 2V loop; the repaired
strict hexagon gate is the acceptance criterion - never full-field correlation alone.
