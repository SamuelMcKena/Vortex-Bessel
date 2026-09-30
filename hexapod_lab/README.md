# Hexapod + Laser Lab

Standalone controller/digital twin for the Newport HXP Stewart platform and the
PHAROS process-beam control path discussed as **LX13**. It remains deliberately
separate from `lab_gui/` while the motion/beam/attenuator workflow is validated.

The operator UI is split into four tabs:

- **CONTROL** — ordinary manual stage, Pockels-cell and attenuator control,
  sample edge calibration, and a script run bar.
- **SCRIPT BUILDER** — experiment sequence construction, preflight and execution.
- **COMMISSIONING** — recovered hardware evidence and guarded self-tests.
- **SETUP + DIAGNOSTICS** — HXP network configuration, LX13 GPIO mapping,
  attenuator hardware binding, CAD loading and diagnostics.

A global header is visible on every tab with stage/beam/attenuator/sample state
plus **CLOSE BEAM** and **STOP MOTION + CLOSE BEAM**.

## Keyboard

| Key | Action |
| --- | --- |
| `Esc` | Stop motion + close the beam (software abort) |
| `Ctrl+B` | Close the process beam |
| `Ctrl+R` | Run the loaded script |
| `Ctrl+.` | Stop the running script |
| `Ctrl+E` | Log the current pose as a sample edge point |
| `Ctrl+1…4` | Switch tab |
| `Enter` / `Delete` / `Ctrl+D` | Edit / remove / duplicate the selected block |
| `Alt+↑` / `Alt+↓` | Move the selected block up or down |

Jog buttons auto-repeat while held, like a physical pendant.

## MOCK LAB and REAL LAB

The header now has one explicit operating-mode selector rather than asking the
operator to mentally coordinate three independent providers.

### MOCK LAB

MOCK LAB guarantees that ordinary Control/Script actions are routed only to the
virtual stage, virtual Pockels provider and virtual attenuator. The digital twin,
2D map, recipe engine and native target-velocity Line behavior remain active.

The mock can be switched between the source-derived historical profiles:

- **LabVIEW v3 candidate** — GPIO3.DO / mask 1, states 0/1 with physical polarity
  deliberately unresolved;
- **TCL legacy writing map** — GPIO4.DO / mask 1, writing=1, non-writing=0.

The Commissioning tab also includes fault-injection presets for stage disconnected,
Pockels unavailable and attenuator unavailable so GUI failure handling can be
tested without hardware.

### REAL LAB

REAL LAB selects the real HXP and real Pockels paths and marks the normal
attenuator as unbound until a calibrated real device is available. Real Pockels
opening remains blocked until the present wiring is commissioned and explicitly
verified.

The persistent header makes the distinction visible:

- `MOCK • SAFE`
- `REAL • DISARMED`
- `REAL • MAPPING VERIFIED`
- `REAL • POCKELS ARMED`

Provider selectors are read-only indicators during ordinary operation; the global
MOCK/REAL mode owns routing so a script cannot quietly mix real and simulated
hardware by accident.

See [UX_FUNCTIONALITY_AUDIT.md](UX_FUNCTIONALITY_AUDIT.md) for the self-audit and
the design changes made after reviewing the first GUI pass.

## Manual control

The Control tab provides:

- Virtual or real Newport HXP selection;
- live XYZUVW readout;
- absolute target moves;
- XYZUVW jogging with separate linear/angular step sizes;
- home and software abort;
- six calculated actuator/strut lengths;
- exact STEP-based articulated digital twin;
- fixed lab laser that terminates at the moving sample surface;
- sample-local laser-written path, split by beam state;
- 2D path map on a metric grid;
- selectable **laser path on sample** or **HXP XY carriage path**;
- adjustable map span, scroll-to-zoom and fit-to-data;
- sample edge calibration and sample-bound motion;
- process-beam Pockels control;
- normalized attenuator control;
- a script run bar: recipe name, live preflight state, progress and
  **RUN SCRIPT** / **STOP SCRIPT** without leaving the control screen.

### Process beam / Pockels cell

The operator controls are intentionally explicit:

- **LASER ON — OPEN POCKELS CELL**
- **LASER OFF — CLOSE POCKELS CELL**

This means enabling/disabling the process beam. It does **not** power-cycle the
PHAROS laser source.

Two providers exist:

- `VirtualLaserGate` — dummy Pockels/LX13 state;
- `HXPDigitalLaserGate` — verified HXP digital output mapped to the PHAROS LX13
  path.

Real beam opening additionally requires **ARM MANUAL REAL-BEAM CONTROL**. Closing
the beam never requires the arm state.

The real provider intentionally contains no guessed GPIO name, pin, active level
or electrical mapping. It cannot arm until the operator explicitly marks the
wiring verified and provides a non-zero GPIO mask with distinct OPEN/CLOSED
values.

### Attenuator

The GUI now includes attenuator control from 0–100 % requested transmission:

- slider + numeric setpoint;
- 0/25/50/75/100 % presets;
- explicit **SET ATTENUATOR** action;
- header/readout state;
- scriptable attenuator setpoints.

The user-facing quantity is deliberately normalized transmission percent. A
device-specific provider can later convert this through the measured calibration
to waveplate/motor angle, analogue voltage or another native quantity.

The actual physical attenuator model/controller/protocol has not yet been
identified, so the real attenuator provider is intentionally **unconfigured and
refuses commands**. No protocol or calibration is fabricated.

## Sample edge calibration

The beam is fixed and the sample rides on the hexapod, so "where is the sample?"
is answered in hexapod pose coordinates. The CONTROL tab therefore has a
capture workflow rather than a guessed sample size:

1. Jog until the beam sits on a physical sample edge or corner.
2. Press **CAPTURE THIS CORNER** (or `Ctrl+E`). The full XYZUVW pose and the
   beam footprint in sample coordinates are logged.
3. Repeat for the remaining corners.

Two captured points are read as opposite corners of an axis-aligned rectangle;
three or more give the convex outline of what was actually measured. A
least-squares plane through the captured Z values reports the sample surface
tilt and flatness.

Ticking **BIND MOTION TO THE CALIBRATED SAMPLE AREA** makes that outline a hard
software bound. It applies to jogs, absolute moves, quick writing lines, raster
sweeps and every script step, and it is path-sampled, so a move whose endpoints
are both on the sample but whose straight path is not is refused as well. An
**edge margin** holds the beam a chosen distance inside the measured edge, and
an optional Z band follows the fitted surface.

The bound is deliberately conservative about arming:

- it cannot be armed with fewer than two distinct corners;
- it cannot be armed while the current pose already violates it, so the stage
  can never be trapped outside its own permitted region;
- editing or removing points that invalidate the bound releases it and says so;
- a loaded calibration file restores the geometry but never re-arms the bound.

The calibration is stored with the hardware config and can also be saved and
loaded on its own. It is drawn on the 2D map (outline, margin and numbered
capture points) and on the sample surface in the 3D twin.

## Quick manual writing line

The Control tab now mirrors the useful part of the recovered LabVIEW front panel:

- **MOVE LINE**
- **WRITE LINE — POCKELS OPEN DURING MOVE**
- **RETURN + ROW — BEAM CLOSED**

Defaults reproduce the familiar historical geometry: a `-7 mm` writing line and
`+0.02 mm` row pitch. The move is executed with the HXP-native
`HexapodMoveIncrementalControlWithTargetVelocity(..., Work, Line, ...)` command
rather than host-side timing. In a writing move the Pockels command is issued
before the blocking HXP Line command and a close command is issued when the move
finishes, fails or is aborted.

## Commissioning

The dedicated Commissioning tab is the bridge between a convincing mock and the
physical lab.

It includes:

- the two hardware candidates recovered from the old TCL/LabVIEW files;
- one-click loading of a candidate into Setup **without arming it**;
- read-only self-test of HXP firmware, XYZUVW pose, `GPIO1.DO`, `GPIO3.DO`,
  `GPIO4.DO` and `GPIO2.DAC1`;
- explicit physical-wiring checklist before a real Pockels mapping can be marked
  verified;
- guarded raw analogue read/write for the historical `GPIO2.DAC1` path, limited
  to the 1–5 values actually observed in the old scripts and blocked while the
  Pockels cell is open;
- mock fault scenarios.

The raw analogue control is intentionally labelled **uncalibrated**. It is for
commissioning the historical path, not for claiming that a value such as `3`
means 60% transmission or a known pulse energy.

## Script Builder

Script construction no longer occupies the general-control screen.

Supported blocks are:

- absolute HXP pose;
- relative HXP move;
- native HXP **Line move at target velocity**;
- **MOVE WHILE WRITE** — one line with the beam open for the move only;
- **LASER ON / POCKELS OPEN**;
- **LASER OFF / POCKELS CLOSED**;
- attenuator transmission setpoint;
- wait.

### MOVE WHILE WRITE

The recovered LabVIEW `LINE Move_While Write` behaviour is a single block rather
than three. It opens the Pockels cell, issues the native line move, and closes
the cell when the move finishes, fails or is stopped. Because the open/close
pair cannot be separated, reordering a sequence can no longer leave the beam
open across a repositioning move. Preflight still reports it as both a motion
and a beam step, so real-hardware arming rules are unchanged.

The raster generator can emit these blocks instead of separate
OPEN / LINE / CLOSED modules, which halves the module count of a large sweep.

### Editing the sequence

- Drag modules from the library into any position; the sequence scrolls while
  you drag past its top or bottom edge, and the insertion point is drawn as a
  blue line.
- Dropping between two blocks inserts there rather than appending.
- Double-click a library module to insert it without dragging.
- `Alt+↑` / `Alt+↓` or the ↑ / ↓ buttons reorder without dragging at all.
- Double-click a block (or press Enter) to edit its values.

Recipes save/load as JSON. Recipe version 3 is written; version 2 and legacy
v0.1 `laser_gate` steps still load.

A **Raster / Parameter Sweep** generator creates legacy-style line arrays without
manually adding tens or hundreds of blocks. It supports write length, row pitch,
series spacing, velocity range/step, return velocity and optional attenuation
series. Every generated writing line has an explicit Pockels OPEN before the Line
move and CLOSED before the return move.

Preflight checks include:

- malformed motion steps;
- Pockels cell left OPEN at the end;
- duplicate Pockels state commands;
- attenuator range;
- attenuation changes while the Pockels cell is open;
- selected real HXP not connected;
- selected real LX13 provider not armed;
- real attenuator requested without a configured driver;
- mixed real/virtual execution providers.
- every sampled motion segment against Cartesian, strut-length and strut-angle
  limits;
- line velocity against the configured ceiling.

Scripts start by requesting the Pockels cell CLOSED. Completion, failure and stop
also request closure. Any script that touches real hardware requires
**ARM REAL SCRIPT EXECUTION**.

Manual motion, beam-open, attenuation and provider switching are disabled while a
script runs; beam-close and stop remain available.

## Newport HXP

The real provider uses direct HXP TCP/API connections on the configured address
(default port `5001`). Three sockets are used:

- **control** — blocking motion/home/initialization commands;
- **poll** — actual/setpoint/target/status queries;
- **I/O** — abort and digital I/O.

Implemented API calls include:

- `GroupPositionCurrentGet`
- `GroupPositionSetpointGet`
- `GroupPositionTargetGet`
- `GroupStatusGet`
- `GroupStatusStringGet`
- `PositionerUserTravelLimitsGet`
- `HexapodMoveAbsolute`
- `HexapodMoveIncremental`
- `GroupMoveAbort`
- `GroupInitialize`
- `GroupHomeSearch`
- `GPIODigitalGet`
- `GPIODigitalSet`
- gathering configuration/run/stop foundations.

The real-HXP path is software implemented but still requires physical controller
validation before production laser processing.

### Workspace and actuator envelope

Every manual move, jog, writing line, raster and recipe uses the same coupled
workspace validator. It checks XYZUVW bounds, all six CAD-derived strut lengths,
strut direction change and sampled intermediate poses. The supplied controller
manual gives a generic example with home at `Z=0`, 28 mm total Z travel and the
best multi-axis range near `Z=14`; the legacy TCL files establish 7 mm lines,
0.02 mm row spacing and 0.2–2.1 mm/s writing speeds. These are evidence-backed
commissioning defaults, not a model-specific certification.

Real motion and homing remain blocked until the complete envelope is explicitly
verified in Setup + Diagnostics. Once connected, **Read strut limits from HXP**
reads `HEXAPOD.1` through `.6`, obtains their user travel limits and aligns those
controller coordinates with the CAD strut lengths at the measured pose. The
operator must still verify the Cartesian and joint limits before marking the
complete envelope verified.

## STEP digital twin

`assets/cad_profile.json` was extracted from the supplied
`Stewart Platform.STEP` and describes the 41-solid Stewart-platform assembly.

The CAD uses Y as its vertical axis while the HXP convention uses Z, so the
registration remains explicit:

```text
HXP X  -> CAD X
HXP Y  -> CAD Z
HXP Z  -> CAD Y
```

With CadQuery installed, **Load STEP…** tessellates and articulates the actual
assembly:

- fixed base;
- moving six-DOF upper carriage;
- actuator bodies pivoting about lower joints;
- telescoping rods following moving upper joints;
- upper/lower joint pieces moving with their correct parent.

The laser graphic is not drawn through the mechanism. It runs from above and
terminates exactly on the moving sample top surface. Written-path points are
stored in sample-local coordinates so the trace stays attached to the sample.

## Reading the path traces

Both the 3D twin and the 2D map store the path as alternating beam-ON and
beam-OFF runs rather than as one polyline per state. A repositioning move can
therefore never be drawn as though it were written material, and two separate
written lines are never joined by a straight line that was never written.

On the 2D map:

- beam-OFF travel is thin dashed grey;
- beam-ON written material is a thick amber stroke with a soft glow;
- each written run gets a round start marker, a square end marker and an index;
- the run count and total written length are shown in the corner;
- the calibrated sample outline, its margin and the numbered capture points are
  drawn underneath;
- hovering gives a crosshair with the position in mm and the distance from the
  live beam spot;
- the scroll wheel zooms, double-click or **Fit** frames all recorded data, and
  both axes always share the same millimetres-per-pixel scale;
- **Travel**, **Written** and **Sample** toggle each layer in both views.

In the 3D twin the beam itself also changes colour, not only opacity, between
armed and idle, and **Iso / Top / Front / Side** snap the camera.

## Installation

Basic controller:

```powershell
cd hexapod_lab
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_hexapod_lab.py
```

Exact STEP rendering:

```powershell
python -m pip install -r requirements-cad.txt
python run_hexapod_lab.py
```

The Windows launcher `START_HEXAPOD_LAB.bat` is also included.
On a new Windows PC, run `INSTALL_AND_START.bat` once instead; it creates a
private virtual environment, installs the GUI plus exact-CAD dependencies, and
starts the application. Subsequent launches can use `START_HEXAPOD_LAB.bat`.

## First real-hardware validation

Start with the process beam safely disabled.

1. Validate virtual XYZUVW motion and exact CAD articulation.
2. Connect the HXP and compare live pose/status with the Newport GUI.
3. Make small single-axis motions and verify coordinate/sign convention.
4. Verify the software abort on a low-risk motion.
5. Confirm the actual PHAROS LX13 pinout, logic level and electrical
   compatibility.
6. Enter the verified GPIO mapping and test CLOSED first with the process beam
   safely intercepted.
7. Only then test OPEN/CLOSED Pockels commands.
8. Identify the physical attenuator/controller and add its device-specific
   provider/calibration before enabling real attenuation commands.
9. Keep real script execution unarmed until all applicable checks pass.

## Update-loop cost

The controller polls, redraws and re-renders at 20 Hz, so per-tick cost is a
correctness concern rather than a nicety: once a tick exceeds 50 ms the event
loop falls behind, the window stops responding and Windows reports it as hung.

Two properties keep it bounded:

- **The twin renders only when something changed.** A static scene costs
  nothing, so an idle controller uses a fraction of a core rather than
  re-rendering the CAD assembly twenty times a second.
- **Trace painting is bounded by pixels, not by stored points.** Each run
  caches a screen-space path decimated to one point per pixel, finished runs
  are rebuilt only when the view moves, runs outside the view are skipped, and
  written length and bounds accumulate as points arrive.

Measured on a 60,000-point trace, which is the stored cap:

| stage | before | after |
| --- | --- | --- |
| 2D map repaint | 726 ms | 6 ms |
| 3D trace rebuild | 34.5 ms | 0.8 ms |
| idle CPU | ~33 % of a core | ~1.5 % |

Both views cap stored points at 60,000 and drop the oldest runs first, so an
all-day session cannot grow without bound.

## If it crashes

`run_hexapod_lab.py` installs crash logging before the GUI starts. Native faults
(VTK/OpenGL), unhandled Python exceptions, background-thread failures and Qt
warnings are all appended to:

```text
hexapod-lab-source/hexapod_lab_crash.log
```

This matters because the Windows launcher runs under `pythonw`, which has no
console: without the log a crash leaves nothing behind. PySide6 also terminates
the process when an exception escapes a slot or a virtual such as `paintEvent`,
so the 20 Hz update loop and both custom paint events are guarded.

The update-loop guard is not a silent catch. It logs the traceback, shows the
error in the status bar, and after five consecutive failures aborts motion and
requests beam closure, because a controller that cannot read its own state must
not hold the beam open on the strength of a stale display.

When reporting a problem, attach that log and say what was on screen at the time.

## Safety boundary

The GUI is not a safety PLC, hardware interlock or emergency-stop system. The
physical PHAROS interlock/shutter chain and hardware E-stop remain authoritative.

The global **STOP MOTION + CLOSE BEAM** button is a software abort plus a Pockels
closure request; it is intentionally not labelled as an emergency stop.

## Tests / CI

Core tests cover:

- extracted Stewart-platform geometry;
- HXP/CAD axis registration;
- Bryant rotation math;
- articulated leg anchors;
- virtual stage motion and busy state;
- fail-closed Pockels behavior;
- virtual attenuator behavior;
- Pockels/attenuator recipe safety;
- legacy recipe compatibility;
- sample edge geometry, margins, surface fit and path-sampled bounds;
- write-line preflight, serialization and sweep generation;
- path-map trace segmentation by beam state;
- sequence-list drag autoscroll and drop placement.

Run locally with:

```powershell
python -m pip install pytest numpy
pytest -q
```

A dedicated GitHub Actions workflow also compile-checks the package and runs the
core tests for changes under `hexapod_lab/`.

## Remaining hardware-bound work

- validate the HXP TCP layer on the actual controller;
- bind/verify PHAROS LX13 electrical mapping;
- add physical Pockels readback if the installed interface exposes it;
- add the real attenuator provider and calibration once hardware is identified;
- add controller-native high-rate synchronized trajectories/events;
- measure the lab-to-HXP/sample transform for a physically calibrated beam axis;
- later integrate the mature providers and UI into the unified lab controller.

## 2026-09-30 controller-backed REAL LAB update

The standalone controller now includes the configuration recovered from the actual HXP backup and live controller screenshots. REAL LAB no longer treats the Stewart STEP file as the kinematic/safety authority.

- HXP endpoint: `192.168.0.254:5001`, group `HEXAPOD`, `Work` frame.
- Controller frames: Work Z=209 mm, Base Z=25 mm, Tool Z=25 mm.
- Controller geometry reference: RRPS, 240 mm base-joint diameter, 150 mm carriage-joint diameter, 167.5410634 mm actuator length at home preset.
- Six LTA-HX actuator configuration records are stored in `assets/hxp_controller_profile_2026-09-30.json`.
- REAL translation-only Line / Move-While-Write commands call `HexapodMoveIncrementalControlLimitGet` immediately before motion and refuse incomplete trajectories or requested velocities above the controller-returned carriage limit.
- General real XYZUVW moves require Cartesian user limits read directly from the connected HXP. No CAD-derived strut length is used as a real-motion safety decision.
- `GPIO2.DAC1` is now the real attenuator provider. Current lab calibration is `4.00 = 40 % transmission`; the GUI maps DAC 0–10 linearly to 0–100 % and checks analogue readback. Real attenuation changes are blocked while the Pockels cell is OPEN.
- Pockels/LX13 control remains commissioning-locked: GPIO3.DO and GPIO4.DO remain historical candidates until the present physical route and polarity are verified.

The STEP file remains fully useful for the articulated 3D visualisation and path display; it is simply separated from the real controller's motion authority.
