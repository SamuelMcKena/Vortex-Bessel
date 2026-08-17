# Nathan MODE 2U3 - Final Hardware Closure and Calibration Bridge (docs/78)

**Status:** hardware/calibration closure for the source-scale branch only. Canonical operating
point: `REALISTIC_4F_HEXAGON_REFERENCE`; secondary strict compromise:
`strict_c6.75_i0.40_q-0.25_r+0.0_p0.00`; forbidden: every old optimum that fails the repaired
M2U2-FIX strict gate, permanently including `m2u2_opt_003_c5.75_i0.32_q-0.25_r+0.0_p0.10`. The
0.997 correlation-to-realistic-reference floor used by that gate is a
**calibrated project-specific eligibility threshold**, not a universal physical definition of a
hexagon. Within the bounded, physically interpretable search that was run, the realistic dual-SLM +
carrier + 4F hexagon is the best strict-eligible candidate for shape, peak intensity and
useful-region energy; no claim of global mathematical optimality outside that tested space is made.
No microfabrication/sample-plane claim is made anywhere in this document.

## The 22 closure questions

**1. What exact SLMs do we have?**

Externally supplied lab identity: two HOLOEYE PLUTO-2.1 NIR-149 phase-only LCOS panels, 1920 x 1080 at 8 um (15.36 x 8.64 mm active). The exact NIR-149 variant is NOT yet repository-verified (the in-repo product link is the NIRO-024 family page).

**2. What is repository/manual verified?**

Panel geometry (1920x1080, 8 um pitch, 15.36x8.64 mm), 8-bit addressing and 93% fill factor: present in the project hardware config (SLMConfig) and in in-repo PLUTO-2.1 family documentation (reference kernel header; THEORY_AND_ANALYSIS.md product link).

**3. What is externally supplied lab information?**

The exact model string 'PLUTO-2.1 NIR-149' and the panel count (two); provenance-labelled `externally_supplied_lab_identity` until the physical labels/manuals are read.

**4. What phase response is known?**

None at the operating wavelength. Drive bit depth 8 is repo-documented; wavelength-specific phase stroke and LUT are `unresolved_requires_calibration` (docs/75). An exact 2*pi stroke is not assumed.

**5. What requires calibration?**

SLM phase stroke/LUT per panel (docs/75); actual 4F focal length, iris centre and radius (docs/76); camera scale and z-stage (docs/77); per-arm reflection parity sign and QWP mount sign (one orientation test + one polarimeter check); hologram-to-axicon centring (0.2 mm tolerance, M2S).

**6. Which wavelength belongs to the source-scale branch?**

Simulation parity: 1030 nm (Nathan source rounding). Physical bench and hardware rebinding: 1029 nm (actual PHAROS / original Digital Twin). The rebind sensitivity check shows the difference is immaterial (~0.1% in fringe period and first-order displacement; strict gate preserved at both).

**7. Which axicon index belongs to the source-scale branch?**

n = 1.458 (validated Nathan fused-silica source value). n = 1.5 is the inherited microfabrication/quicklook placeholder scope and is not forced onto this branch.

**8. What focal length should the 4F use?**

f = 300 mm (nominal from the bench description, F300 profile), confirmed on the bench via the docs/76 displacement measurement. The 100 mm CSLM value is a removed placeholder.

**9. What carrier should be displayed?**

6.25 lp/mm = 20 SLM pixels per period on the 8 um panel (the value used by every validated M2N/M2Q/M2S/M2U run).

**10. Where should the +1 order appear physically?**

x = lambda*f*carrier = 1.929 mm from the zero order at 1029 nm with f = 300 mm.

**11. What physical iris diameter is required?**

1.54 mm (radius 0.772 mm = 0.40 x carrier separation) centred on the +1 order; clearance to the zero order 1.16 mm. M2S: the whole 0.24-0.80 fraction range passes.

**12. What camera calibration is needed?**

Everything (make, pitch, sensor size, magnification, relay-vs-direct, z-stage): docs/77 four-part routine calibration. Observation-side only; does not block M2V.

**13. What does QWP = -45 degrees physically mean?**

code -45 deg means: the QWP FAST axis lies 45 deg from the lab-horizontal H axis, rotated CLOCKWISE when you stand downstream and look back into the beam (equivalently 45 deg anticlockwise when looking along the propagation direction from behind the source) any odd number of upstream mirror reflections after the recombiner flips the apparent sense; the sign is therefore fixed on the bench by ONE polarimeter check: with only the H channel open and a uniform mask, the QWP at the correct -45 deg turns H into LEFT-hand circular in receiver view (fast axis clockwise-45 deg); if right-hand circular is observed, rotate the QWP to +45 deg

**14. Are extra HWPs required?**

Yes: HWP #1 at the input for H/V power balance. In the V arm, EITHER HWP #2/#3 at 45 deg before and after SLM-V, OR mount SLM-V rotated 90 deg (no extra plates). One final QWP closes the chain; no other waveplates are needed.

**15. What polarisation must reach each SLM?**

Linear polarisation aligned to each panel's LC director (phase-only requirement). SLM-H receives H; SLM-V receives V rotated onto the director by HWP #2 or by panel rotation. The exact NIR-149 director orientation is unverified in the repo: one polariser test resolves it.

**16. How are H/V channels split?**

Polarising beamsplitter (PBS #1): H transmits, V reflects, after the input polariser + HWP #1.

**17. How are they recombined?**

PBS #2 recombines the arms collinearly; path lengths matched well within the ~260 fs pulse coherence length; the relative arm piston is free (uniform polarisation rotation, observable-invariant per M2S).

**18. Which M2U2 conflicts are resolved?**

All seven received exactly one closure status:

- `wavelength` -> **resolved_different_scopes**: 1029 nm is the actual PHAROS/original-Digital-Twin laser value and governs the physical bench and hardware rebinding; 1030 nm is the Nathan source-model rounding retained for V0 parity; the rebind sensitivity check shows the difference is immaterial (docs/78)
- `axicon_refractive_index` -> **resolved_different_scopes**: n = 1.458 is the validated Nathan source-scale fused-silica value and stays authoritative for this branch; n = 1.5 belongs to the inherited microfabrication target/quicklook scope and is not forced onto the source-scale axicon
- `4f_focal_length` -> **resolved_placeholder_removed**: the CSLM 100 mm value is a warning-only placeholder and is removed from consideration; the F300 nominal 300 mm bench description is adopted as the recommendation, with the docs/76 blaze-grating displacement measurement confirming the actual focal length on the bench
- `carrier_frequency` -> **resolved_different_scopes**: the source-scale display carrier is 6.25 lp/mm (20 px blaze on the 8 um panel) as used by every validated M2N/M2Q/M2S/M2U run; CSLM command-domain records and vector-arm defaults are different route semantics and stay in their own scopes
- `beam_radius` -> **resolved_different_scopes**: the 2 mm 1/e source beam is the Nathan/Twin source-scale value; the 24 um CSLM value is a diagnostic grid source in a different scope
- `camera_calibration` -> **unresolved_requires_measurement**: no camera make/pitch/magnification exists in the repository; docs/77 converts this into a routine four-part calibration (sensor pitch, magnification target, carrier-displacement cross-check, z-stage translation); observation-side only, so not architecture-critical
- `slm_exact_model_and_phase_stroke` -> **unresolved_requires_measurement**: panel geometry (1920x1080, 8 um, 15.36x8.64 mm, 8-bit, 93% fill) is confirmed by project config plus in-repo PLUTO-2.1 family documentation; the exact NIR-149 model remains externally supplied lab identity until the physical label/manual is read, and the wavelength-specific phase stroke/LUT is a routine docs/75 calibration; neither changes the optical architecture

**19. Which unknowns are routine calibration items?**

SLM phase stroke/LUT, exact iris centre / focal-length confirmation, camera scale, beam centring, per-arm parity sign, QWP mount sign. None changes the optical architecture.

**20. Does hardware rebinding preserve the canonical strict hexagon?**

Yes. At the resolved 1029 nm binding the canonical point remains strict-eligible (corr-to-realistic 1.0000, deltaC -0.0288, dark core 0.0015, first-order efficiency 0.9495).

**21. Does the strict compromise remain eligible?**

Yes: strict-eligible at 1029 nm (corr-to-realistic 0.9991, deltaC -0.0293); it stays the secondary candidate, and the canonical realistic-4F reference remains preferred because rebinding did not change the ranking.

**22. Is M2V authorised?**

Outcome **M2U3-A**: All architecture-critical M2U2-B blockers are resolved or converted into explicit routine laboratory calibration steps, and hardware rebinding preserves the strict hexagon for both the canonical realistic-4F reference and the strict compromise. M2V is authorised.

## Output tree

`outputs/figures/digital_twin/nathan_mode2u3_hardware_closure/` -> `00_slm/`,
`01_phase_calibration/`, `02_4f/`, `03_camera/`, `04_jones_axes/`, `05_conflicts/`, `06_rebind/`,
`07_final_status/`. Calibration bridges: docs/75 (SLM phase), docs/76 (4F), docs/77 (camera).
