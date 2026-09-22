# Hexapod + Laser Lab

Standalone first-pass controller/sandbox for the Newport HXP Stewart platform and the PHAROS external laser-gate path discussed as **LX13**. It is deliberately separate from `lab_gui/` so the stage/laser workflow can be developed and hardware-tested before it is merged into the unified lab controller.

## What is implemented

### Virtual hexapod

- six HXP-style coordinates: `X Y Z U V W`
- HXP Bryant/Tait-Bryan `ZYX` rotation convention
- absolute and incremental motion
- configurable dummy translation/rotation speed
- smooth simulated motion rather than teleporting between poses
- current / target state
- per-leg Stewart-platform lengths
- abort and home-to-zero

### Real Newport HXP foundation

A small direct TCP client is included instead of depending on a proprietary Python package. The supplied Newport HXP documentation specifies TCP/IP control on port `5001` and the commands used here:

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
- `GatheringConfigurationSet`
- `GatheringRun` / `GatheringStop`

The HXP uses blocking sockets, so the controller opens independent **control**, **poll**, and **I/O/abort** connections. That means the GUI can continue to poll the measured Cartesian position while a move command is executing, and `ABORT` / laser-gate commands are not queued behind the blocking motion socket.

The real-HXP path is software implemented but **not yet validated against the lab controller**. Treat the first hardware session as an interface-validation session, not as production processing.

## 3D Stewart-platform digital twin

The geometry profile in `assets/cad_profile.json` was extracted from the supplied `Stewart Platform.STEP`:

- SolidWorks STEP AP214
- 41 solids
- 500 mm square lower plate
- 500 mm square upper carriage
- six repeating actuator assemblies
- home spherical-joint separation approximately `384.315535 mm`

The supplied CAD happens to use **CAD Y as vertical**, while the HXP documentation defines **HXP Z as vertical**. The software keeps that registration explicit:

```text
HXP X  -> CAD X
HXP Y  -> CAD Z
HXP Z  -> CAD Y
```

The lightweight viewer therefore works without the STEP file and is already driven from the actual dimensions/joint centres extracted from it.

### Exact CAD articulation

Install `requirements-cad.txt`, then click **Load Stewart Platform.STEP…**.

The loader:

1. imports the STEP assembly with CadQuery;
2. tessellates its solids into a local cache;
3. recognizes the six actuator groups from the supplied 41-solid assembly profile;
4. keeps the base fixed;
5. applies the carriage's true six-DOF transform to the upper plate;
6. rotates each actuator body around its lower spherical joint;
7. moves/rotates each piston/rod from the moving upper joint, producing a telescoping visual motion;
8. moves the upper joint/mount with the carriage;
9. keeps the lower mounts attached to the base.

So this is not a canned CAD animation. The same `Pose6D` that drives the virtual stage or comes back from the real HXP drives the CAD rig.

The exact supplied STEP SHA-256 is recorded in the profile and `assets/README.md` so another CAD export cannot silently be treated as the calibrated geometry.

## Fixed laser / sample visualization

A fixed laboratory beam axis is drawn down the centre of the platform. The sample proxy moves with the carriage. On every update the software computes the beam intersection with the moving carriage/sample plane.

Two traces are kept:

- **travel trace** — everywhere the beam/sample intersection travelled;
- **processing trace** — only positions visited while the laser gate was ON.

This is intended to make script mistakes visually obvious before a physical run, e.g. a return move accidentally occurring with the gate enabled.

## PHAROS LX13 laser gate

There are two laser providers:

- `VirtualLaserGate` — safe dummy state for sequence testing;
- `HXPDigitalLaserGate` — maps a verified HXP digital output to the PHAROS LX13 external-control path.

The real provider intentionally contains **no guessed LX13 pin number, TTL level, active-high/active-low assumption, GPIO name, or mask**. Those values must be entered from the lab's actual PHAROS/HXP wiring documentation and the operator must explicitly confirm that the electrical interface has been checked.

When the real gate is armed it is immediately commanded OFF. Stop/abort/close operations also request gate OFF.

The GUI is **not** a laser safety system. The physical interlock, shutter/emission chain and emergency stop remain authoritative.

## Recipe builder

The bottom panel is a first accessible sequence builder. Rows can be drag-reordered and currently support:

- absolute move
- incremental move
- laser gate ON
- laser gate OFF
- wait

Recipes are plain JSON and can be saved/loaded.

Preflight currently validates step payloads and rejects a recipe that finishes with the laser gate ON. Real recipe execution additionally requires an explicit **ARM REAL RECIPE EXECUTION** checkbox.

This is intentionally the first stage of the visual block system. Later blocks can add line/arc trajectories, sweeps, loops, acquisition, SLM changes and controller-native synchronized trajectories without changing the provider architecture.

## Install

For the normal dummy/real controller with the lightweight CAD-derived rig:

```powershell
cd hexapod_lab
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_hexapod_lab.py
```

For exact STEP rendering:

```powershell
python -m pip install -r requirements-cad.txt
python run_hexapod_lab.py
```

If CadQuery is easier through the lab's Anaconda installation, install it in that environment and run `run_hexapod_lab.py` from the same Spyder/Anaconda interpreter.

There is also `START_HEXAPOD_LAB.bat` for the Windows lab PC.

## First lab validation checklist

Do **not** start with the laser enabled. A sensible validation order is:

1. Run Virtual mode and verify X/Y/Z/U/V/W directions visually.
2. Load the exact STEP and check all six leg articulations over small virtual poses.
3. Connect the HXP with the laser gate still set to Virtual.
4. Read current pose/status only; compare against the Newport web GUI.
5. Make very small single-axis moves with the physical workspace clear.
6. Verify GUI actual/setpoint/target values and CAD direction against physical motion.
7. Verify `ABORT` from the independent HXP I/O socket.
8. Only after the PHAROS LX13 pinout/logic/electrical levels are documented, fill in GPIO/mask/ON/OFF values and validate the external gate with the processing beam safely intercepted.
9. Keep recipe real-execution disarmed until the above passes.

## Tests

Core tests do not need Qt/VTK:

```powershell
python -m pip install pytest numpy
pytest -q
```

Current tests cover:

- extracted Stewart-platform home geometry;
- HXP-Z to CAD-Y registration;
- Bryant rotation orthonormality;
- body/rod anchoring under articulated poses;
- virtual-stage motion;
- fail-closed virtual laser behavior;
- recipe gate-off preflight.

## Current scope / next steps

Good next increments after first hardware validation:

- ingest the HXP controller's own hexapod geometry/configuration file and compare its kinematics against the STEP-derived rig;
- live actuator/strut readout from `HEXAPOD.1 ... HEXAPOD.6` in addition to Cartesian `HEXAPOD.X ... HEXAPOD.W`;
- high-rate HXP gathering for measured trajectories;
- controller-native line/arc/rotation trajectories;
- synchronized pulse/gathering support using HXP event/trajectory functions;
- measured lab-to-HXP transform for a physically calibrated laser axis instead of the current centreline default;
- editable sample dimensions/fixture offset;
- richer block recipe editor with loops/sweeps;
- then migrate the mature providers/viewer/recipe engine into the existing unified lab GUI.
