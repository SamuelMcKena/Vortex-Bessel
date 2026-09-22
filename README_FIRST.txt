VORTEX-BESSEL LAB GUI v2.1 — VIRTUAL LAB + LIVE LAB

Keep this folder separate from your old working GUI.

Recommended (Spyder / Anaconda)
1. Open SPYDER_RUN_LAB_CONTROL_V2_1.py in Spyder.
2. Press F5.

Or double-click START_LAB_CONTROL_V2_1.bat.

First-time checks:
- RUN_DIAGNOSTICS.bat
- INSTALL_OR_REPAIR_DEPENDENCIES.bat if NumPy, Pillow, PySide6 or pywin32 is missing.
- Start PC-Beamage, connect the Beamage 4M, and enable Pipeline.
- TEST_BEAMAGE_PIPE.bat checks the official named-pipe connection.

VIRTUAL LAB (personal device, no hardware)

Run it:
  cd C:\path\to\Vortex-Bessel
  python .\lab_gui\run_lab_control.py

The full, illustrated instructions are INSIDE the app: sidebar -> "How to use this".

Quick start
1. Sidebar: select VIRTUAL LAB, open the Virtual Lab page.
2. Tab "1 Beam": vortex route + q=20 -> Set route. Optionally Beam radius -> Apply beam size.
3. Under the camera: "Cast masks + go live". Camera z and -/+ move the camera.
4. Tab "2 Errors": e.g. Wavefront -> Coma X 0.8 waves -> Apply faults.
5. Tab "3 Correct": follow Steps 1-5.
   Step 1 choose SLM + modes (Coma / Low order / Everything),
   Step 2 camera planes (at least two, e.g. 31, 45),
   Step 3 Find correction (original masks are restored afterwards),
   Step 4 review the table of proposed SLM changes and the score J -> Apply or Discard,
   Step 5 check the live camera, press Next iteration, repeat.
   The history table records every round; Export... saves it as JSON.
6. Tab "Captures": save repeated frames at the current z as evidence.
7. Tab "Settings": axicon, resolution, hidden scenarios, optimiser tuning, activity log.

Score J (lower = better): 3x beam walk + 1x ring-radius variation + 1.5x ring
eccentricity + 1.5x azimuthal unevenness + 0.5x dark-core light + 4x edge
clipping + 4x saturation, averaged over the measured planes.

The virtual camera is a Beamage-4M: 2048 x 2048 pixels at 5.5 um over the full
11.26 mm sensor. The Bessel intensity fringes (~6.5 um) are resolved first and
then area-integrated over each 5.5 um pixel, so the 4-fold hyperbolic moire
around the core is real sensor aliasing, as on the physical camera. The first
bright ring is small on this camera: ~5 pixels across for q=5, ~17 for q=20.
Press Core to zoom to it at camera-pixel scale. Zoomed out, screen pixels show
the average of the camera pixels beneath them, so the monitor adds no moire.

"View: camera pixels" (above the Virtual Lab camera, and beside Full frame on
Home) flips to "View: native model": the same plane, noise-free, at the fine
2.75 um model sampling, same zoom and overlays -- the clean pattern the offline
simulations show. Click again for what the Beamage records. Captures, metrics
and correction always use the camera frame.

Camera readouts: beam radius D4sigma/2 in x and y with circularity (as Beamage
reports), ring radius, ring roundness (1 - eccentricity), unevenness. Model
millimetres; the real Beamage field of view is not yet calibrated.

The axicon: the default is the bench Thorlabs "20 degree" optic, modelled by
its MEASURED k_perp = 4.83e5 1/m from the real BeamGage q=20 z-scan (the same
value real_bmg_digital_twin_correction.py uses). In the exact refractive
convention that is a ~9.7 degree base angle (4.5 degree cone). "20 degrees" is
NOT the base angle - entering 20 as a base angle doubles the cone. Steep cones
are simulated as in that reference code: SLM relay on the working grid over a
11.26 mm sensor window, Fourier handoff to a finer axicon grid (1024 -> 4096), and one
frozen angular spectrum so each camera plane is a single inverse FFT.

Auto-expose (beside Snapshot) sets the virtual exposure so the peak at the
current plane sits near 75% of full scale, then takes a frame - the bench answer
to a clipped or invisible beam. At fixed sensitivity and 1 ms, a q=0 focus clips
(peak ~9400 counts at z=15 mm) while q=20 reads 2840 there and only 47 at
z=45 mm. The frame chip says CLIPPED when saturated; saturation also costs score
on the Correct tab, so expose first.

The 4F relay: SLM -300- L1(f=300) -300- aperture -300- L2(f=300) -300- axicon,
as reported on 2026-09-17. Every spacing equals a focal length, so it is a
symmetric 4F: the axicon sees the SLM at 1:1, rotated 180 degrees (a beam pushed
+x on the SLM walks -x on the camera), with the aperture in the shared focal
plane. "4F stop diameter" in Settings models that aperture: a stop of diameter D
passes SLM detail to D/(2*lambda*f). Default is ideal (no filtering). For q=20
on a 2 mm beam, 10 mm passes 98% of the light, 3 mm 81%, 1 mm only 14%; a plain
Gaussian passes any of them untouched. A tight iris does not move the ring, it
degrades the vortex that makes it.

Tiny beams: a wider input beam does NOT shrink the rings (that is lens
intuition). The axicon fixes ring size; the input beam sets Bessel length
(L ~ w/tan(theta)). 1 mm in is bright and gone by z=32 mm; 3 mm in is dimmer and
still going, same ring size, same total power. Because the relay is 1:1, only a
demagnifying objective after the axicon shrinks the rings: set "Objective 1:N"
in Settings. An ideal telescope scales the pattern exactly (rings xM, z xM^2,
cone angle /M with M=1/N), so the simulated frame IS the sample-plane pattern in
new units and the camera metrics gain a SAMPLE PLANE line. Ring size follows the
objective; length follows the input beam radius. NA = N x 0.079 for this axicon,
so beyond about 1:12 in air there is no propagating cone and the request is
refused. Small-beam designer (tab 1) turns a size/length target into the needed
objective and input beam radius; the old 3 um / 150 um target comes out as 1:3.3
with a 0.13 mm input beam. Objective pupil truncation and aberrations are not
modelled.

Speed on the default LIVE MODEL grid: ~1.5 s per camera move, ~4.5 s per optimiser
probe (the Correct tab estimates run time before you start). 2048/4096 modes
are much slower; 4096 needs several GB of RAM.

If it closes unexpectedly: every session writes lab_gui\logs\lab_gui_crash.log
and the app warns you at the next start. Keep that file.

The Beamage instructions below apply only to explicitly selected LIVE LAB mode.

Camera viewing:
- Beamage frames stay at full acquired pixel resolution.
- Fit beam / Auto-fit beam changes only the view; it never crops stored data.
- Full frame restores the whole sensor view.
- Mouse wheel or +/- zooms; drag pans; double-click fits the beam.
- Inferno, Gentec-like, Turbo, Viridis and grayscale are display-only maps.
- Exposure is read back in this GUI. The supplied official Pipeline commands do
  not include exposure writing, so use the PC-Beamage controls button to bring
  the manufacturer window forward and change it there.

The PC-Beamage BMP remains LIVE PREVIEW ONLY until validated against a numeric
camera export. Formal quantitative capture remains disabled for this route.
