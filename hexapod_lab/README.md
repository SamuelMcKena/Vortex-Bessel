# Hexapod + Laser Lab

Standalone controller/digital twin for the Newport HXP Stewart platform and the
PHAROS process-beam control path discussed as **LX13**. It remains deliberately
separate from `lab_gui/` while the motion/beam/attenuator workflow is validated.

The operator UI is now split into three tabs:

- **CONTROL** — ordinary manual stage, Pockels-cell and attenuator control.
- **SCRIPT BUILDER** — experiment sequence construction, preflight and execution.
- **SETUP + DIAGNOSTICS** — HXP network configuration, LX13 GPIO mapping,
  attenuator hardware binding, CAD loading and diagnostics.

A global header is visible on every tab with stage/beam/attenuator state plus
**CLOSE BEAM** and **STOP MOTION + CLOSE BEAM**.

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
- sample-local laser-written path;
- 2D movement map on a metric grid;
- selectable **laser path on sample** or **HXP XY carriage path**;
- adjustable map span;
- process-beam Pockels control;
- normalized attenuator control.

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
- **LASER ON / POCKELS OPEN**;
- **LASER OFF / POCKELS CLOSED**;
- attenuator transmission setpoint;
- wait.

Rows can be drag-reordered and recipes save/load as JSON. Recipe version 2 is
written; legacy v0.1 `laser_gate` steps load as Pockels-cell steps.

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
- legacy recipe compatibility.

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
