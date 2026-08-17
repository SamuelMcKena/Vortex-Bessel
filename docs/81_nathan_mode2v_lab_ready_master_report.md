# Nathan MODE 2V - Lab-Ready Master Report (docs/81)

Canonical operating point: `REALISTIC_4F_HEXAGON_REFERENCE` (best strict-eligible shape/peak/useful-energy within the bounded M2U2-FIX search; no global optimality claim). Secondary: `strict_c6.75_i0.40_q-0.25_r+0.0_p0.00`. Forbidden: `m2u2_opt_003_c5.75_i0.32_q-0.25_r+0.0_p0.10` and every repaired-gate failure. No microfabrication/sample-plane success is claimed.

## The 30 build questions

**1. What exact bench do we build?**

Route B: PHAROS 1029 nm -> POL1 + HWP#1 -> PBS#1 -> SLM-H arm and SLM-V arm (with conditional V-arm HWPs) -> PBS#2 recombination -> ONE common 4F (f=300 mm) with a single +1-order iris -> QWP (code -45 deg) -> axicon (2 deg, n=1.458) -> free-space Bessel zone -> camera on a z stage.

**2. Do we need the old segmented six-piece optic?**

No.

**3. What replaces it?**

programmable dual-SLM phase control of two orthogonal polarisation channels + one uniform QWP; sector pattern/rotation/duty/pistons/centring/aberration precompensation are all digital

**4. What polarisation leaves the laser?**

Linear (PHAROS class); orientation/purity unrecorded in the repo - measured in STAGE 1 and cleaned by POL1.

**5. What does HWP #1 do?**

Sets the H/V power split at PBS #1 (target 50/50; 0.8-1.2 tolerated).

**6. How are H/V channels split?**

PBS #1: H transmits, V reflects.

**7. What polarisation reaches each SLM?**

Linear along each panel's LC director: H directly on SLM-H; V rotated onto the SLM-V director by HWP #2 (or panel mounted rotated 90 deg).

**8. Are extra V-arm HWPs needed?**

Conditional on the STAGE 2 panel orientation test: either HWP #2/#3 at 45 deg around SLM-V, or none if the panel is mounted rotated 90 deg.

**9. What mask goes on SLM-H?**

phi_H = wrap(+alpha + carrier), native 1920x1080, centre pixel (960, 540).

**10. What mask goes on SLM-V?**

phi_V = wrap(-alpha + pi/2 + carrier), same panel geometry and carrier sign.

**11. What carrier is used?**

6.25 lp/mm = 20 SLM pixels per period, along +x on both panels.

**12. What 4F lenses are recommended?**

Two f = 300 mm lenses in a 4f chain (nominal from the bench description; confirmed via docs/76). ONE common 4F after recombination is preferred over per-arm relays (no differential H/V filter errors; matches every validated simulation).

**13. Where is the +1 order physically?**

1.929 mm from the zero order at the Fourier plane (x = lambda f nu).

**14. What iris diameter is recommended?**

1.54 mm (0.40 x carrier separation); M2S passes over the whole 0.24-0.80 range.

**15. How are the channels recombined?**

PBS #2 before the common 4F; path lengths matched inside the ~260 fs coherence length; the relative arm piston is free (uniform polarisation rotation; observable-invariant).

**16. What does the final QWP do?**

Maps the dual-linear channels onto the segmented radial/azimuthal vector field (docs/79 derivation).

**17. What is the physical QWP angle convention?**

code -45 deg means: the QWP FAST axis lies 45 deg from the lab-horizontal H axis, rotated CLOCKWISE when you stand downstream and look back into the beam (equivalently 45 deg anticlockwise when looking along the propagation direction from behind the source)

**18. Where is the axicon?**

Directly after the QWP at the 4F output plane, centred on the hologram axis (<= 0.2 mm blind, then digitally recentred).

**19. Where is the camera?**

On a z translation stage scanning ~10-200 mm behind the axicon, including the exact 60 mm reference plane.

**20. How much power survives each stage?**

Model fractions (vendor factors excluded, not evidenced): 0.50 per arm; x0.865 fill factor; x0.9495 iris; total at z=60 mm = 0.7657 of input (see the power budget CSV).

**21. How much useful-region power remains?**

0.2799 of laser input inside the fixed useful hexagon region (0.3655 of the z=60 mm plane power).

**22. What is the peak-intensity proxy?**

34.25 (simulation units): mean intensity in the 3x3 neighbourhood centred on the maximum pixel.

**23. What is the main practical tolerance?**

Hologram-centre-to-axicon-axis registration: <= 0.2 mm blind, fully correctable by measure-and-recentre; everything else (8-bit phase, fill factor, piston, QWP +/-2 deg, iris range, registration at tens of um, z +/-20 mm) is forgiving (M2S).

**24. What does the camera correct?**

Beam/mask centring, C3/C6 symmetry, dark core, lobe balance, z structure - the strict-gate observables.

**25. What does the Shack-Hartmann correct?**

Common-path low-order wavefront (defocus, astigmatism, coma) feeding the bounded Zernike precompensation.

**26. What does polarimetry validate?**

Pre-axicon vector field (sector structure), H/V balance, relative phase and the QWP mount sign.

**27. How does the closed-loop correction work?**

Display -> measure -> coarse digital recentre scan -> bounded Nelder-Mead over mask centre, V piston, sector rotation/duty, six sector pistons, defocus/astig and V-mask shift, using measured images only; the repaired strict hexagon gate is the acceptance criterion. The search never receives the injected truth. Demonstrations: recovered to strict-eligible: D_unknown_low_order_aberration; substantially recovered (hexagonal, >= 0.98, at the physical ceiling of the injected damage): A_unknown_axicon_mask_offset_0p5mm, E_combined_unknown_errors; improved: A_unknown_axicon_mask_offset_0p5mm, B_moderate_m2s_combined, E_combined_unknown_errors.

**28. What exact masks are exported?**

09_mask_package: panel-space wrapped-phase .npy (radians + normalised) for SLM-H/SLM-V, preview-only uint8 PNGs, high-res previews and metadata (LUT NOT applied; hardware_ready=false).

**29. What still requires calibration?**

Per-panel phase stroke/LUT (docs/75), actual 4F focal/iris geometry (docs/76), camera scale + z stage (docs/77), panel orientation/parity tests, QWP mount sign, beam centring.

**30. Is the system ready for a source-scale lab trial?**

Outcome **M2V-A**: The source-scale dual-SLM + 4F + QWP + axicon bench is fully specified at the architecture level; native masks are exported; physical 4F dimensions are defined; polarisation routing is explicit; remaining unknowns are routine calibration tasks; closed-loop correction is demonstrated. Ready for source-scale laboratory trial.

