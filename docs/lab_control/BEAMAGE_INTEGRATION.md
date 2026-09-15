# Gentec PC-Beamage named-pipe integration

## Source and protocol

The implementation in `lab_gui/labcontrol/devices/beamage_pipe.py` was derived
from the supplied official `NamedPipeClient-V1.00.12` C++/MFC example. The files
used were `NamedPipeClientDlg.cpp`, `NamedPipeClientDlg.h` and the Visual Studio
project, whose Unicode character set establishes UTF-16LE `TCHAR` framing.

The client opens the same duplex pipe as the example:

```text
\\.\pipe\pipe_beamage
```

It implements the demonstrated commands only: measurement name, acquisition
start/stop and state, all measurements, all positions, image dimensions, and
single/continuous BMP save. The command buffers use the explicit `TCHAR` counts
from the example. Where the C++ sample writes past the terminating NUL, Python
pads deterministically with NULs rather than reproducing an undefined over-read.

## What is operational

On Windows, `BeamageCameraProvider` uses `pywin32` to wait for, open and exchange
messages with the PC-Beamage pipe. Reads have bounded timeouts; disconnect and
error paths close the handle; malformed, empty and negative responses are
reported to the GUI. The device name/serial, image dimensions, vendor beam
measurements, position measurements and exposure readback are retained in frame
metadata.

The normal **Connect** action only opens the pipe. It deliberately does not fire
optional identity/dimension commands during the handshake because installed
PC-Beamage builds may differ. The explicit pipe diagnostic performs those probes.
Live BMP preview is capped at 3 fps, metadata is polled separately, preview-only
frames are downsampled for display, and only the visible camera page renders.
These limits avoid large duplicate 2048×2048 allocations and pipe command bursts.

The launcher appends uncaught Python/native-fault information to
`lab_gui_crash.log` in the package folder. If the GUI still closes, preserve that
file before reopening it.

PC-Beamage must be running with the camera connected and **Pipeline enabled**.
Exposure and gain remain controlled in PC-Beamage because the supplied example
contains no commands for setting them.

## Quantitative-safety boundary

The vendor example requests a BMP and displays it. It does not prove that the BMP
is an untouched quantitative sensor matrix. Consequently:

- the BMP is available in the Home and Measure views as `LIVE_PREVIEW_ONLY`;
- vendor-reported measurements are shown separately;
- the frame metadata says `quantitative_valid=false` and
  `HARDWARE_UNVERIFIED`;
- formal capture rejects that frame route;
- no screenshot or screen scraping is used.

The status can only be promoted after physical validation against a documented
numeric export, including dimensions, orientation, bit depth/full scale,
freshness, exposure, saturation and repeated start/stop behavior.

## Lab-PC check

From the portable folder, run `RUN_DIAGNOSTICS.bat`, then
`TEST_BEAMAGE_PIPE.bat`. Its default check only opens and closes the Pipeline
handle. Optional commands can be exercised deliberately with
`python tools\\test_beamage_pipe.py --probe` and one BMP with `--preview`.
None of these checks by itself validates the BMP as quantitative evidence.

Common failures:

| Message | Action |
|---|---|
| named pipe is Windows-only | Run on the Windows lab PC |
| `pywin32` required | Run `INSTALL_OR_REPAIR_DEPENDENCIES.bat` in the Spyder Anaconda environment |
| could not open pipe | Start PC-Beamage, connect the camera, and enable Pipeline |
| response timeout | Stop/restart acquisition and PC-Beamage, then rerun the diagnostic |
| returned image path unavailable | Check PC-Beamage save permissions and that its returned path is accessible to the same Windows user |
