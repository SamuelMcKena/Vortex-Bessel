# CAD asset

The controller ships with `cad_profile.json`, which was extracted from the supplied `Stewart Platform.STEP` and is enough for the lightweight articulated Stewart-rig view.

For the **exact CAD surface view**, put the original STEP file here (or keep it anywhere on the lab PC) and use **Load Stewart Platform.STEP…** in the GUI. The viewer tessellates the 41 solids into a local cache and then animates the actual actuator body/rod/joint meshes.

Expected supplied-file SHA-256:

`0b8b57b1705bff1b72043518cbebdccb8a604ead2beaba061f1f0432c5b56c85`

The STEP itself is intentionally not required for ordinary dummy-mode testing; this keeps the GUI usable even on a PC without CadQuery.
