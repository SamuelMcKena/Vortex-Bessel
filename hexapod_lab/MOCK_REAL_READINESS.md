# Mock / Real readiness matrix

This file is the practical status sheet for the standalone Hexapod + Laser Lab GUI.

| Capability | MOCK LAB | REAL LAB |
| --- | --- | --- |
| XYZUVW absolute / relative control | Implemented | Implemented; HXP validation required |
| Native HXP Line + target velocity | Simulated with real timing semantics | Implemented from recovered LabVIEW/TCL API |
| Exact STEP digital twin | Auto-load when STEP + CAD extras are present | Driven by measured HXP pose |
| 2D movement / process map | Implemented | Driven by measured HXP pose + commanded Pockels state |
| Manual WRITE LINE | Implemented | Implemented; real Pockels must be commissioned/armed |
| Pockels OPEN/CLOSED | Logical virtual state | HXP GPIO provider + output-register readback |
| Pockels hardware profile | LabVIEW-v3 and TCL mock profiles | Candidate preload only; physical verification required |
| Attenuator % | Fully simulated | Deliberately unavailable until calibrated |
| Legacy raw analogue 1–5 | Simulated/readable | Guarded commissioning read/write on chosen HXP analogue GPIO |
| Raster / velocity sweep generation | Implemented | Same recipe engine; real-provider preflight applies |
| Fault injection | Stage / Pockels / attenuator faults | N/A |
| Read-only commissioning self-test | Simulated report | Firmware, pose, GPIO1/3/4, GPIO2.DAC1 |
| Recipe preflight | Implemented | Adds connection/arm/provider checks |
| Global close-beam action | Implemented | Implemented; fail-closed request |
| Hardware emergency stop | Not simulated as a safety device | Physical E-stop remains authoritative |

## Operating principle

MOCK LAB is not a decorative demo. It exercises the same high-level control paths,
trajectory types, recipe engine, Pockels semantics, path tracking and digital-twin
updates used by REAL LAB.

REAL LAB does not silently infer current wiring from historical software. The
LabVIEW/TCL evidence is used to pre-fill candidates and diagnostics, while the
present physical Pockels mapping must be explicitly commissioned before it can arm.

## Recovered historical configuration used by the GUI

- HXP: `192.168.0.254:5001`
- timeout: `10 s`
- group: `HEXAPOD`
- processing coordinate system: `Work`
- writing trajectory: `HexapodMoveIncrementalControlWithTargetVelocity`,
  `Line`
- later LabVIEW Pockels candidate: `GPIO3.DO`, mask 1, states 0/1,
  polarity unresolved
- TCL writing mapping: `GPIO4.DO`, mask 1, write=1, non-write=0
- LabVIEW writing-state marker: `GPIO1.DO`, mask 4, states 0/4
- analogue evidence: `GPIO2.DAC1`
- later LabVIEW serial resource: `COM7`

## Remaining real-lab commissioning items

The code is intentionally waiting on physical confirmation for:

- current HXP-output-to-PHAROS-LX13 wiring;
- current Pockels OPEN/CLOSED polarity;
- electrical compatibility of that interface;
- identity/protocol of the COM7 device;
- calibrated relation between raw power/attenuation command and measured optical
  transmission / pulse energy;
- measured lab-to-sample beam-axis registration.

Those are commissioning measurements, not missing GUI architecture.


## Current lab observations — 2026-09-30

The latest lab photographs/screenshots resolve several previously open items:

- HXP web interface is live at `192.168.0.254`.
- `Test-NetConnection 192.168.0.254 -Port 5001` succeeded from the lab PC
  (`192.168.0.100`).
- HXP controller chassis is labelled **Model HXP**.
- Current Work-frame pose in the Newport web UI was approximately:
  `X=0, Y=0, Z=-13.999998 mm, U=-1e-6 deg, V=0, W=0`.
- Coordinate-system snapshot:
  - Tool in Carriage = `[0, 0, 25, 0, 0, 0]`
  - Base in World = `[0, 0, 25, 0, 0, 0]`
  - Work in World = `[0, 0, 209, 0, 0, 0]`
- Digital I/O snapshot showed `GPIO1.DO` raw value **4** (display bit 3 high),
  matching the old LabVIEW mask-4 gate/writing-state marker.
- `GPIO3.DO` and `GPIO4.DO` were both low in that snapshot.
- `GPIO2.DAC1` was **10.0000**, which proves that the old TCL values 1–5 were
  only a historical operating subset and must not be treated as the complete
  present-day DAC range or as calibrated optical power.
- The PHAROS **LX13** connector is physically populated.
- The HXP **GPIO3** connector is physically populated by a D-sub cable, which is
  consistent with the later LabVIEW-v3 GPIO3.DO Pockels candidate.
- The photographs do **not** yet provide an unbroken cable trace proving that the
  populated HXP GPIO3 cable is the exact cable arriving at PHAROS LX13.
- A physical mushroom emergency-stop/interlock device is present in the setup.
- The mechanical hexapod stage model is still not identified from a model label;
  the controller label `Model HXP` is not the mechanical-stage model.

These observations are stored in
`assets/current_lab_observations_2026-09-30.json` and should be treated as the
current lab snapshot rather than legacy-source inference.
