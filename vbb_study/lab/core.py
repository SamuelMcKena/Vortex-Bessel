"""Auditable sensorless correction, with no camera, stage or laser actuation.

Trial phases are ADDITIVE native-panel radians. The tested coefficient is the
command to add, not an inferred aberration coefficient to negate. Intensity
improvement is not a measurement of vortex charge or of the wavefront.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

MODES = {"astig_x": (2, 2), "astig_xy": (2, -2), "coma_x": (3, 1),
         "coma_y": (3, -1), "trefoil_x": (3, 3), "trefoil_y": (3, -3),
         "spherical": (4, 0), "defocus": (2, 0)}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_array(path, *, skiprows=0, delimiter=None):
    """Read quantitative 2D arrays; never turn colour screenshots into data."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".bmg":
        from .beamage import read_bmg
        a, _ = read_bmg(path)
    elif suffix == ".npy":
        a = np.load(path, allow_pickle=False)
    elif suffix in {".txt", ".csv"}:
        a = np.loadtxt(path, skiprows=skiprows,
                       delimiter="," if suffix == ".csv" and delimiter is None else delimiter)
    elif suffix in {".bmp", ".png", ".tif", ".tiff"}:
        with Image.open(path) as im:
            if im.mode == "P":
                raise ValueError("Palette images are not quantitative camera matrices; export numeric TXT.")
            a = np.asarray(im)
    else:
        raise ValueError(f"Unsupported {suffix}: use numeric TXT/CSV/NPY or unscaled grayscale BMP/TIFF/PNG. Beamage-4M v1 BMG is also supported.")
    if a.ndim != 2 or min(a.shape) < 4 or not np.isrealobj(a):
        raise ValueError("Expected a real 2D intensity/phase matrix, not RGB, a table of coordinates, or a screenshot.")
    a = np.asarray(a, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("Array contains NaN or infinity.")
    return a


def zernike(shape, center_yx, radius_px, n, m):
    """Real disk Zernike, unit continuous-disk RMS; +m cosine, -m sine.

    x increases right, y down in native display coordinates. Zero outside the
    measured illuminated disk; no hidden transpose, flip, resize or relay map.
    """
    if n < 0 or abs(m) > n or (n - abs(m)) % 2:
        raise ValueError("Invalid Zernike (n,m).")
    y, x = np.indices(shape, dtype=float)
    x = (x - center_yx[1]) / radius_px
    y = (y - center_yx[0]) / radius_px
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    radial = np.zeros_like(r)
    k = abs(m)
    for s in range((n - k)//2 + 1):
        c = (-1)**s * math.factorial(n-s) / (math.factorial(s)*math.factorial((n+k)//2-s)*math.factorial((n-k)//2-s))
        radial += c * r**(n-2*s)
    angular = np.cos(k*theta) if m >= 0 else np.sin(k*theta)
    return np.where(r <= 1, radial * angular * np.sqrt((n+1)*(1 if m == 0 else 2)), 0)


def compose_phase(base, correction):
    """Compose a COMPLETE base command and additive correction, wrap once."""
    if base.shape != correction.shape or not np.isfinite(base).all() or not np.isfinite(correction).all():
        raise ValueError("Base and correction must have identical shapes and finite radians.")
    wrapped = np.mod(base + correction, 2*np.pi).astype(np.float32)
    # float32 can round a value just below 2*pi up to 2*pi. Canonicalise
    # that endpoint so the SDK's own periodic wrap preserves the same raster.
    wrapped[wrapped >= np.float32(2*np.pi)] = 0.0
    return wrapped


def sampling_diagnostic(phase, support):
    # Circular differences reveal steep resolved steps; cannot prove no aliasing
    # when the original unwrapped base phase has already lost sampling information.
    gradients = []
    for axis in (0, 1):
        sl0, sl1 = [slice(None)]*2, [slice(None)]*2
        sl0[axis], sl1[axis] = slice(None, -1), slice(1, None)
        mask = support[tuple(sl0)] & support[tuple(sl1)]
        gradients.extend(np.abs(np.angle(np.exp(1j*np.diff(phase, axis=axis))))[mask].tolist())
    g = np.asarray(gradients)
    return {"p99_resolved_step_rad": float(np.quantile(g, .99)),
            "fraction_resolved_steps_above_pi_over_2": float(np.mean(g > np.pi/2)),
            "scope": "Resolved circular differences only; not a proof of adequate sampling or a diagnosis of aliasing."}


def template():
    return {
        "schema": 1, "data_kind": "experiment", "wavelength_nm": 1030.0, "beam_label": "q20",
        "slm": {"shape_yx": [1080, 1920], "panel_id": "6010-2381",
                "center_yx_px": None, "illuminated_radius_px": None,
                "native_coordinates_verified": False,
                "phase_response_optically_verified": False,
                "phase_owner": "unknown",
                "phase_path_evidence": None,
                "notes": "phase_owner: existing_driver (radians through verified HEDS/GUI), offline_lut (raw grayscale path), or unknown (preview only)."},
        "camera": {"shape_yx": [2048, 2048], "full_scale": 4095, "exposure_us": None,
                   "gain": None, "settings_id": "FILL_FIXED_CAMERA_SETTINGS",
                   "linear_raw_export_verified": False,
                   "txt_skiprows": 0, "txt_delimiter": None},
        "planes": [{"id": "z000", "z_mm": None, "axis_yx_px": None},
                   {"id": "z001", "z_mm": None, "axis_yx_px": None},
                   {"id": "z002", "z_mm": None, "axis_yx_px": None}],
        "z_reference": "FILL_STAGE_REFERENCE_AND_POSITIVE_DIRECTION",
        "roi_radius_px": None, "repeats": 3,
        "limits": {"minimum_fractional_improvement": .02,
                   "max_plane_loss_increase": .02, "max_core_fraction_increase": .01,
                   "max_radial_profile_l1": .10, "min_power_ratio": .90,
                   "max_power_ratio": 1.10, "max_control_loss_drift": .03,
                   "max_edge_power_fraction": .01,
                   "max_saturation_fraction": 0.0},
        "notes": "Nominal SLM size/wavelength are starting values, not calibration. Measure footprint, axes and z positions; keep exposure/gain/attenuation fixed. Axes come from reference measurements, not recentering each trial."
    }


def init_session(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    save_json(path/"profile.json", template())
    (path/"incoming").mkdir()
    return path/"profile.json"


def validate_profile(p):
    def positive(x, label):
        if x is None or not np.isfinite(x) or x <= 0:
            raise ValueError(f"Set a finite positive {label} in profile.json.")
    def shape(a, label):
        if not isinstance(a, list) or len(a) != 2 or any(not isinstance(v, int) or v < 4 for v in a):
            raise ValueError(f"Set {label} to [rows, columns].")
    if p.get("data_kind", "experiment") not in {"experiment", "synthetic_demo"}:
        raise ValueError("Unknown data_kind.")
    slm, camera = p["slm"], p["camera"]
    shape(slm["shape_yx"], "SLM shape"); shape(camera["shape_yx"], "camera shape")
    for val, label in [(p["wavelength_nm"], "wavelength"), (slm["illuminated_radius_px"], "SLM radius"),
                       (camera["full_scale"], "camera full scale"), (camera["exposure_us"], "exposure"),
                       (p["roi_radius_px"], "ROI radius")]:
        positive(val, label)
    if camera["gain"] is None or not np.isfinite(camera["gain"]) or camera["gain"] < 0:
        raise ValueError("Record the fixed camera gain (zero is allowed).")
    if not camera["linear_raw_export_verified"]:
        raise ValueError("Verify numeric linear camera export; false-colour/auto-scaled images cannot be scored.")
    if not isinstance(p["repeats"], int) or p["repeats"] < 3:
        raise ValueError("At least three independent repeat captures are required.")
    def disk(center, radius, dims, label):
        if center is None or len(center) != 2 or not np.isfinite(center).all():
            raise ValueError(f"Set measured {label} [y,x].")
        if any(c-radius < 1 or c+radius > size-2 for c, size in zip(center, dims)):
            raise ValueError(f"{label} disk must fit inside the image/panel with a one-pixel margin.")
    disk(slm["center_yx_px"], slm["illuminated_radius_px"], slm["shape_yx"], "SLM footprint")
    planes = p["planes"]
    if len(planes) < 1 or [v["id"] for v in planes] != [f"z{i:03d}" for i in range(len(planes))]:
        raise ValueError("Use sequential plane ids z000, z001, ...")
    zs = []
    for plane in planes:
        if plane["z_mm"] is None or not np.isfinite(plane["z_mm"]):
            raise ValueError("Fill the actual stage coordinate for every plane; indices are not distances.")
        zs.append(plane["z_mm"])
        disk(plane["axis_yx_px"], p["roi_radius_px"], camera["shape_yx"], "camera optical axis")
    if len(set(zs)) != len(zs):
        raise ValueError("Duplicate z coordinates.")
    for key in ("z_reference",):
        if not p[key] or "FILL" in p[key]:
            raise ValueError(f"Record {key}.")
    if "FILL" in camera["settings_id"] or not camera["settings_id"]:
        raise ValueError("Record camera settings_id, including attenuation and export settings.")
    limits = p["limits"]
    for key, value in limits.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid limit {key}.")
    if not 0 < limits["min_power_ratio"] <= 1 <= limits["max_power_ratio"]:
        raise ValueError("Power-ratio limits must bracket one.")
    if slm["phase_owner"] not in {"unknown", "existing_driver", "offline_lut"}:
        raise ValueError("Unknown phase_owner.")


def start_session(path, base, dark):
    path = Path(path)
    if (path/"session.json").exists():
        raise ValueError("Session already started; use a new session to change calibration/settings.")
    p = read_json(path/"profile.json")
    validate_profile(p)
    # Base must explicitly be radians; never infer radians from an 8-bit bitmap.
    if Path(base).suffix.lower() != ".npy":
        raise ValueError("Base must be the current COMPLETE SLM2 phase as a radians .npy, including carrier and existing corrections.")
    b = load_array(base)
    if list(b.shape) != p["slm"]["shape_yx"]:
        raise ValueError("Base phase shape differs from native SLM shape.")
    ds = [load_array(f, skiprows=p["camera"]["txt_skiprows"], delimiter=p["camera"]["txt_delimiter"]) for f in dark]
    if len(ds) < 3 or any(list(a.shape) != p["camera"]["shape_yx"] for a in ds):
        raise ValueError("Supply at least three dark frames with the configured camera shape.")
    if any(np.min(a) < -p["camera"]["full_scale"] or np.max(a) >= p["camera"]["full_scale"] for a in ds):
        raise ValueError("Dark frames are out of range or saturated; check full_scale/export.")
    if len({digest(f) for f in dark}) != len(dark):
        raise ValueError("Duplicate dark files; capture independent frames.")
    if p["slm"]["phase_owner"] != "unknown":
        if not p["slm"]["native_coordinates_verified"]:
            raise ValueError("Verify native display coordinates before claiming a calibrated phase path.")
        evidence = p["slm"]["phase_path_evidence"]
        if not evidence or not Path(evidence).is_file() or "FILL" in p["slm"]["panel_id"]:
            raise ValueError("Supply phase_path_evidence file and actual panel ID for the verified wavelength/phase path.")
    calibration = path/"calibration"
    calibration.mkdir(exist_ok=True)
    np.save(calibration/"base_phase_rad.npy", b.astype(np.float32))
    np.save(calibration/"dark_mean.npy", np.mean(ds, axis=0))
    files = ["base_phase_rad.npy", "dark_mean.npy"]
    for i, f in enumerate(dark):
        name = f"dark_{i+1:02d}{Path(f).suffix}"
        shutil.copyfile(f, calibration/name); files.append(name)
    if p["slm"]["phase_owner"] != "unknown":
        name = "phase_evidence" + Path(evidence).suffix
        shutil.copyfile(evidence, calibration/name); files.append(name)
        p["slm"]["phase_path_evidence"] = "calibration/"+name
    save_json(calibration/"profile.json", p); files.append("profile.json")
    save_json(path/"session.json", {"created_utc": now(), "schema": 1,
        "calibration_sha256": {"calibration/"+f: digest(calibration/f) for f in files},
        "accepted_trial": None, "rounds": []})
    (path/"trials").mkdir(exist_ok=True)
    return p


def session(path):
    path = Path(path)
    s = read_json(path/"session.json")
    for name, sha in s["calibration_sha256"].items():
        if digest(path/name) != sha:
            raise ValueError(f"Frozen calibration modified: {name}. Start a new session.")
    profile = read_json(path/"calibration/profile.json")
    validate_profile(profile)
    return path, s, profile


def trial_info(path, trial_id):
    if Path(trial_id).name != trial_id or trial_id in {".", ".."}:
        raise ValueError("Invalid trial id.")
    folder = Path(path)/"trials"/trial_id
    t = read_json(folder/"trial.json")
    for name, sha in t["phase_sha256"].items():
        if digest(folder/name) != sha:
            raise ValueError(f"Trial phase modified: {trial_id}/{name}")
    return folder, t


def create_trial(path, profile, trial_id, correction, metadata):
    folder = path/"trials"/trial_id
    folder.mkdir()
    base = np.load(path/"calibration/base_phase_rad.npy", allow_pickle=False)
    np.save(folder/"correction_phase_rad.npy", correction.astype(np.float32))
    np.save(folder/"total_phase_rad.npy", compose_phase(base, correction))
    slm = profile["slm"]
    support = zernike(base.shape, slm["center_yx_px"], slm["illuminated_radius_px"], 0, 0) > 0
    metadata.update({"id": trial_id, "created_utc": now(),
        "phase_units": "radians", "coordinate_space": "native SLM2 pixels, y down, x right",
        "phase_owner": slm["phase_owner"],
        "status": "PREVIEW_ONLY" if slm["phase_owner"] == "unknown" else "TRIAL_REQUIRES_MEASUREMENT",
        "sampling": sampling_diagnostic(base+correction, support),
        "phase_sha256": {f: digest(folder/f) for f in ("correction_phase_rad.npy", "total_phase_rad.npy")}})
    save_json(folder/"trial.json", metadata)
    incoming = path/"incoming"/trial_id
    incoming.mkdir(parents=True)
    lines = ["Save NEW raw frames with EXACT stems below; use .txt/.csv/.npy/.bmp/.tif/.png.",
             "Restore the named total phase before each acquisition. Keep camera settings fixed."]
    for z in profile["planes"]:
        lines.append(f"{z['id']} = stage {z['z_mm']} mm")
        lines.extend(f"{z['id']}_r{r:02d}.bmg" for r in range(1, profile["repeats"]+1))
    (incoming/"CAPTURE_NAMES.md").write_text("\n".join(lines), encoding="utf-8")
    return metadata


def plan_round(path, mode, values=(-.30, -.15, .15, .30), seed=42):
    path, s, p = session(path)
    if mode not in MODES:
        raise ValueError(f"Choose a mode from {list(MODES)}")
    values = [float(v) for v in values]
    if len(values) < 2 or not np.isfinite(values).all() or len(set(values)) != len(values) or 0 in values or min(values) >= 0 or max(values) <= 0 or max(map(abs, values)) > .5:
        raise ValueError("Use distinct signed nonzero coefficients, including both signs, at most 0.5 rad RMS per step.")
    if s["rounds"] and not (path/"trials"/s["rounds"][-1]["end"]/"measurement.json").exists():
        raise ValueError("Finish the previous round's end control before planning another.")
    index = len(s["rounds"])+1
    prefix = f"round{index:02d}"
    parent = s["accepted_trial"]
    accepted = np.zeros(p["slm"]["shape_yx"])
    if parent:
        folder, _ = trial_info(path, parent)
        accepted = np.load(folder/"correction_phase_rad.npy", allow_pickle=False)
    slm = p["slm"]
    mode_map = zernike(accepted.shape, slm["center_yx_px"], slm["illuminated_radius_px"], *MODES[mode])
    order = np.random.default_rng(seed+index).permutation(values).tolist()
    commands = [(prefix+"_start", 0., "control_start")]
    commands += [(f"{prefix}_trial{i+1:02d}", v, "candidate") for i, v in enumerate(order)]
    commands += [(prefix+"_end", 0., "control_end")]
    for tid, value, role in commands:
        create_trial(path, p, tid, accepted+value*mode_map,
                     {"round": index, "mode": mode, "n_m": MODES[mode],
                      "coefficient_rad_rms": value, "parent_accepted_trial": parent, "role": role})
    round_ = {"index": index, "mode": mode, "start": commands[0][0], "end": commands[-1][0],
              "trials": [x[0] for x in commands], "parent_accepted_trial": parent}
    s["rounds"].append(round_); save_json(path/"session.json", s)
    return round_


def capture(path, trial_id, folder=None, settings_id=None):
    path, s, p = session(path)
    out, t = trial_info(path, trial_id)
    if (out/"measurement.json").exists():
        raise ValueError("This capture is sealed. Use a fresh verification trial; never overwrite raw data.")
    if settings_id != p["camera"]["settings_id"]:
        raise ValueError("Pass --settings-id matching the frozen camera/attenuation settings after checking the camera.")
    source = Path(folder) if folder else path/"incoming"/trial_id
    if not source.is_dir():
        raise ValueError(f"Capture folder does not exist: {source}")
    skip = p["camera"]["txt_skiprows"]; delimiter = p["camera"]["txt_delimiter"]
    dark = np.load(path/"calibration/dark_mean.npy", allow_pickle=False)
    previous_hashes = set()
    previous_pixels = set()
    for f in (path/"trials").glob("*/measurement.json"):
        for record in read_json(f)["frames"]:
            previous_hashes.add(record["raw_sha256"])
            previous_pixels.add(record.get("raw_pixels_sha256"))
    pending, hashes, pixels = [], set(), set()
    expected = [f"{z['id']}_r{r:02d}" for z in p["planes"] for r in range(1, p["repeats"]+1)]
    supported = {".txt", ".csv", ".npy", ".bmp", ".png", ".tif", ".tiff", ".bmg"}
    actual = [f for f in source.iterdir() if f.suffix.lower() in supported]
    if sorted(f.stem for f in actual) != sorted(expected):
        raise ValueError("Missing, duplicate-format, or extra frames. Match CAPTURE_NAMES.md exactly; each plane/repeat needs one file.")
    for z in p["planes"]:
        for r in range(1, p["repeats"]+1):
            stem = f"{z['id']}_r{r:02d}"
            f = next(f for f in actual if f.stem == stem)
            sha = digest(f)
            if sha in previous_hashes or sha in hashes:
                raise ValueError(f"Reused raw bytes in {f.name}; require independent captures, including controls.")
            hashes.add(sha)
            raw = load_array(f, skiprows=skip, delimiter=delimiter)
            if list(raw.shape) != p["camera"]["shape_yx"] or np.min(raw) < -p["camera"]["full_scale"] or np.max(raw) > p["camera"]["full_scale"]:
                raise ValueError(f"{f.name}: shape/range mismatch. Check matrix headers, full scale and camera export.")
            pixel_sha = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
            if pixel_sha in previous_pixels or pixel_sha in pixels:
                raise ValueError(f"Reused pixel data in {f.name}, even though file bytes/metadata differ.")
            pixels.add(pixel_sha)
            saturated = float(np.mean(raw >= p["camera"]["full_scale"]))
            if saturated > p["limits"]["max_saturation_fraction"]:
                raise ValueError(f"{f.name}: saturated pixels; lower exposure and start a new session with new darks.")
            a = np.maximum(raw-dark, 0.)
            y, x = np.indices(a.shape)
            rad = np.hypot(y-z["axis_yx_px"][0], x-z["axis_yx_px"][1])
            roi = rad <= p["roi_radius_px"]
            power = float(a[roi].sum())
            if power <= 0:
                raise ValueError(f"{f.name}: no positive background-subtracted signal.")
            edge = np.zeros(a.shape, bool); edge[:2] = edge[-2:] = True; edge[:, :2] = edge[:, -2:] = True
            edge_fraction = float(a[edge].sum()/max(a.sum(), 1e-30))
            if edge_fraction > p["limits"]["max_edge_power_fraction"]:
                raise ValueError(f"{f.name}: excessive sensor-edge signal; inspect clipping and dark subtraction.")
            pending.append((f, a, {"plane": z["id"], "z_mm": z["z_mm"], "repeat": r,
                                  "raw_sha256": sha, "raw_pixels_sha256": pixel_sha, "source_name": f.name,
                                  "saturation_fraction": saturated, "edge_power_fraction": edge_fraction,
                                  "roi_power_counts": power}))
    rawdir = out/"raw"; rawdir.mkdir(exist_ok=True)
    arrays = out/"arrays"; arrays.mkdir(exist_ok=True)
    records = []
    for f, a, record in pending:
        shutil.copyfile(f, rawdir/f.name)
        np.save(arrays/(f.stem+".npy"), a)
        record.update({"raw": "raw/"+f.name, "array": "arrays/"+f.stem+".npy",
                       "array_sha256": digest(arrays/(f.stem+".npy"))})
        records.append(record)
    measurement = {"trial_id": trial_id, "captured_utc": now(), "settings_id": settings_id,
                   "frames": records, "phase_sha256": t["phase_sha256"],
                   "provenance": "User-attested manual display and new camera captures; no hardware telemetry."}
    save_json(out/"measurement.json", measurement)
    return measurement


def measurements(path, tid):
    folder, t = trial_info(path, tid)
    m = read_json(folder/"measurement.json")
    if m["phase_sha256"] != t["phase_sha256"]:
        raise ValueError("Measurement phase provenance mismatch.")
    arrays = {}
    for f in m["frames"]:
        if digest(folder/f["raw"]) != f["raw_sha256"] or digest(folder/f["array"]) != f["array_sha256"]:
            raise ValueError("Sealed measurement data changed.")
        arrays.setdefault(f["plane"], []).append(np.load(folder/f["array"], allow_pickle=False))
    return arrays


def radial_target(mean, axis, radius):
    y, x = np.indices(mean.shape)
    r = np.hypot(y-axis[0], x-axis[1])
    bins = r.astype(int)
    roi = r <= radius
    sums = np.bincount(bins[roi], weights=mean[roi])
    counts = np.bincount(bins[roi])
    radial = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    target = np.zeros_like(mean)
    target[roi] = radial[bins[roi]]
    return target, roi


def plane_scores(images, target, roi):
    t = target[roi]/target[roi].sum()
    norm = np.sum(t*t)
    losses, powers, core_fractions, radial_changes = [], [], [], []
    # Infer only a fixed target geometry from the radial reference. This is not
    # an independently measured optical axis or a vortex-charge measurement.
    y, x = np.indices(target.shape)
    cy, cx = np.mean(y[roi]), np.mean(x[roi])
    radial_bins = np.hypot(y-cy, x-cx).astype(int)
    bins = radial_bins[roi]
    counts = np.bincount(bins)
    radial_target_mass = np.bincount(bins, weights=t)
    peak_radius = int(np.argmax(radial_target_mass/np.maximum(counts, 1)))
    core_mask = bins < max(1., .5*peak_radius)
    for a in images:
        signal = a[roi]; power = signal.sum()
        losses.append(float(np.sqrt(np.sum((signal/power-t)**2)/norm)))
        powers.append(float(power))
        normalised = signal/power
        core_fractions.append(float(normalised[core_mask].sum()))
        radial_mass = np.bincount(bins, weights=normalised, minlength=len(counts))
        radial_changes.append(float(np.sum(np.abs(radial_mass-radial_target_mass))))
    return {"loss": float(np.mean(losses)),
            "loss_sem": float(np.std(losses, ddof=1)/np.sqrt(len(losses))),
            "power": float(np.mean(powers)),
            "core_fraction": float(np.mean(core_fractions)),
            "radial_profile_l1": float(np.mean(radial_changes))}


def evaluate(path, round_index=None):
    path, s, p = session(path)
    if not s["rounds"]:
        raise ValueError("Plan a round first.")
    current = s["rounds"][-1] if round_index is None else s["rounds"][round_index-1]
    missing = [tid for tid in current["trials"] if not (path/"trials"/tid/"measurement.json").exists()]
    if missing:
        raise ValueError("Capture missing trials before evaluation: "+", ".join(missing))
    # Acquisition order matters: bracket the sweep with its actual controls.
    dates = [read_json(path/"trials"/tid/"measurement.json")["captured_utc"] for tid in current["trials"]]
    if dates != sorted(dates):
        raise ValueError("Capture/ingest in prescribed order so the controls bracket the trials.")
    baseline_id = s["rounds"][0]["start"]
    baseline = measurements(path, baseline_id)
    targets = {z["id"]: radial_target(np.mean(baseline[z["id"]], axis=0), z["axis_yx_px"], p["roi_radius_px"]) for z in p["planes"]}
    results = []
    all_scores = {}
    for tid in current["trials"]:
        data = measurements(path, tid)
        all_scores[tid] = {z: plane_scores(data[z], *targets[z]) for z in targets}
    start, end = all_scores[current["start"]], all_scores[current["end"]]
    limits = p["limits"]
    control_reasons = []
    for z in targets:
        ratio = end[z]["power"]/start[z]["power"]
        if abs(end[z]["loss"]-start[z]["loss"]) > limits["max_control_loss_drift"]:
            control_reasons.append(f"{z}: control shape drift")
        if not limits["min_power_ratio"] <= ratio <= limits["max_power_ratio"]:
            control_reasons.append(f"{z}: control power drift")
    for tid in current["trials"]:
        _, t = trial_info(path, tid)
        scores = all_scores[tid]
        reasons = list(control_reasons)
        for z, score in scores.items():
            reference = (start[z]["loss"]+end[z]["loss"])/2
            power = (start[z]["power"]+end[z]["power"])/2
            if score["core_fraction"] > (start[z]["core_fraction"]+end[z]["core_fraction"])/2+limits["max_core_fraction_increase"]:
                reasons.append(f"{z}: central-core filling")
            if score["radial_profile_l1"] > limits["max_radial_profile_l1"]:
                reasons.append(f"{z}: radial-profile change")
            if score["loss"] > reference+limits["max_plane_loss_increase"]:
                reasons.append(f"{z}: worse shape")
            if not limits["min_power_ratio"] <= score["power"]/power <= limits["max_power_ratio"]:
                reasons.append(f"{z}: power change")
        loss = float(np.mean([v["loss"] for v in scores.values()]))
        ref = float(np.mean([(start[z]["loss"]+end[z]["loss"])/2 for z in targets]))
        # Operational repeat-noise margin, not a formal confidence interval.
        sem = np.sqrt(sum(v["loss_sem"]**2 for v in scores.values()))/len(scores)
        refsem = np.sqrt(sum(start[z]["loss_sem"]**2+end[z]["loss_sem"]**2 for z in targets))/(2*len(scores))
        threshold = max(limits["minimum_fractional_improvement"]*ref, 2*np.hypot(sem, refsem))
        improvement = ref-loss
        if improvement <= threshold:
            reasons.append("Improvement does not exceed configured repeat-noise/relative margin")
        results.append({"trial_id": tid, "role": t["role"], "coefficient_rad_rms": t["coefficient_rad_rms"],
                        "mean_loss": loss, "improvement": improvement, "required_improvement": float(threshold),
                        "eligible": t["role"] == "candidate" and not reasons,
                        "reasons": reasons, "planes": scores})
    eligible = sorted((r for r in results if r["eligible"]), key=lambda r: r["mean_loss"])
    result = {"round": current["index"], "created_utc": now(),
              "target": "Frozen rotational average of first baseline, in fixed measured-axis ROIs. This is a symmetry reference, NOT an ideal beam or wavefront reconstruction.",
              "reference_trial": baseline_id, "results": results,
              "recommended_trial": eligible[0]["trial_id"] if eligible else None,
              "status": "FRESH_VERIFICATION_REQUIRED" if eligible else "NO_RELIABLE_IMPROVEMENT",
              "experimental_accepted": False}
    save_json(path/f"round{current['index']:02d}_evaluation.json", result)
    return result


def plan_verification(path):
    path, s, p = session(path)
    result = evaluate(path)
    best = result["recommended_trial"]
    if not best:
        raise ValueError("No candidate passed. Restore the start control; inspect data or test another mode.")
    round_ = s["rounds"][-1]
    if round_.get("verification"):
        raise ValueError("Verification already planned; use its existing capture folders.")
    prefix = f"round{round_['index']:02d}_verify"
    ids = []
    for suffix, source in [("before", round_["start"]), ("candidate", best), ("after", round_["start"])]:
        tid = prefix+"_"+suffix
        src, old = trial_info(path, source)
        create_trial(path, p, tid, np.load(src/"correction_phase_rad.npy", allow_pickle=False),
                     {"role": "verification_"+suffix, "source_trial": source,
                      "coefficient_rad_rms": old["coefficient_rad_rms"], "round": round_["index"]})
        ids.append(tid)
    round_["verification"] = ids
    round_["verification_candidate"] = best
    save_json(path/"session.json", s)
    return ids


def accept(path):
    path, s, p = session(path)
    current = s["rounds"][-1]
    if current.get("accepted"):
        raise ValueError("Round already accepted.")
    if p["slm"]["phase_owner"] == "unknown":
        raise ValueError("Preview-only phase path cannot be accepted as a calibrated trial.")
    if len(p["planes"]) < 3:
        raise ValueError("Acceptance requires at least three distinct measured z planes; one-plane sessions are screening only.")
    ids = current.get("verification")
    if not ids:
        raise ValueError("Run verify-plan and capture a fresh before/candidate/after triplet first.")
    # Reuse the same comparison/gates with a temporary in-memory round, not an
    # overwritten sweep or prediction masquerading as a measurement.
    original = s["rounds"][-1]
    verification_round = dict(original, start=ids[0], end=ids[2], trials=ids)
    return _evaluate_verification(path, s, p, verification_round, original)


def _evaluate_verification(path, s, p, round_, original):
    # Evaluate independently using the same fixed target and criteria. Roles
    # need not be mutated on disk to reuse this pure score comparison.
    baseline = measurements(path, s["rounds"][0]["start"])
    targets = {z["id"]: radial_target(np.mean(baseline[z["id"]], axis=0), z["axis_yx_px"], p["roi_radius_px"]) for z in p["planes"]}
    dates = [read_json(path/"trials"/tid/"measurement.json")["captured_utc"] for tid in round_["trials"]]
    if dates != sorted(dates):
        raise ValueError("Fresh verification controls must bracket the candidate in capture order.")
    scores = []
    for tid in round_["trials"]:
        data = measurements(path, tid)
        scores.append({z: plane_scores(data[z], *targets[z]) for z in targets})
    before, candidate, after = scores
    reasons = []
    lim = p["limits"]
    ref = np.mean([(before[z]["loss"]+after[z]["loss"])/2 for z in targets])
    loss = np.mean([candidate[z]["loss"] for z in targets])
    sem = np.sqrt(sum(candidate[z]["loss_sem"]**2 + (before[z]["loss_sem"]**2+after[z]["loss_sem"]**2)/4 for z in targets))/len(targets)
    for z in targets:
        if abs(before[z]["loss"]-after[z]["loss"]) > lim["max_control_loss_drift"]:
            reasons.append(f"{z}: verification control drift")
        control_power_ratio = after[z]["power"]/before[z]["power"]
        ratio = candidate[z]["power"]/((before[z]["power"]+after[z]["power"])/2)
        if not lim["min_power_ratio"] <= ratio <= lim["max_power_ratio"] or not lim["min_power_ratio"] <= control_power_ratio <= lim["max_power_ratio"]:
            reasons.append(f"{z}: verification power change")
        if candidate[z]["core_fraction"] > (before[z]["core_fraction"]+after[z]["core_fraction"])/2+lim["max_core_fraction_increase"]:
            reasons.append(f"{z}: verification central-core filling")
        if candidate[z]["radial_profile_l1"] > lim["max_radial_profile_l1"]:
            reasons.append(f"{z}: verification radial-profile change")
        if candidate[z]["loss"] > (before[z]["loss"]+after[z]["loss"])/2+lim["max_plane_loss_increase"]:
            reasons.append(f"{z}: verification shape worsened")
    margin = max(lim["minimum_fractional_improvement"]*ref, 2*sem)
    if ref-loss <= margin:
        reasons.append("Independent improvement does not exceed repeat-noise/relative margin")
    if original["parent_accepted_trial"] != s["accepted_trial"]:
        reasons.append("Accepted parent changed; candidate is stale")
    accepted = not reasons
    result = {"created_utc": now(), "experimental_accepted": accepted and p.get("data_kind", "experiment") == "experiment",
              "verification_passed": accepted,
              "status": ("SIMULATED_VERIFICATION_PASSED" if p.get("data_kind") == "synthetic_demo" else "MEASURED_SYMMETRY_IMPROVEMENT") if accepted else "VERIFICATION_REJECTED",
              "reasons": reasons, "mean_reference_loss": float(ref), "mean_candidate_loss": float(loss),
              "required_improvement": float(margin), "verification_trials": round_["trials"],
              "phase_response_optically_verified": p["slm"].get("phase_response_optically_verified", False),
              "scope": "Intensity symmetry and power within configured ROIs/z planes; not ideal-beam, topological-charge, or wavefront validation."}
    save_json(path/f"round{original['index']:02d}_acceptance.json", result)
    if accepted:
        s["accepted_trial"] = original["verification_candidate"]
        original["accepted"] = True
        save_json(path/"session.json", s)
    return result


def export_phase(path, trial_id, lut_csv=None, raw_path_verified=False):
    """Export a driver-owned radians command OR an offline-LUT raw raster."""
    path, s, p = session(path)
    folder, t = trial_info(path, trial_id)
    owner = p["slm"]["phase_owner"]
    if owner == "unknown":
        raise ValueError("Unverified phase command path: arrays are preview-only. Complete the actual phase-path record in a new session.")
    if owner == "existing_driver":
        if lut_csv is not None or raw_path_verified:
            raise ValueError("Driver owns phase conversion. An offline LUT would risk double conversion.")
        return {"file": str(folder/"total_phase_rad.npy"),
                "instruction": "REPLACE the total phase in the verified radians input path. Do not add another carrier/correction, resize, flip or convert this array to gray."}
    if not lut_csv or not raw_path_verified:
        raise ValueError("offline_lut requires --lut CSV and --raw-path-verified: the GUI must display raw grayscale without another LUT or phase conversion.")
    from vbb_study.calibration.slm_phase import SLMPhaseCalibration, calibrated_phase_to_grey
    table = np.genfromtxt(lut_csv, delimiter=",", names=True)
    if set(table.dtype.names or ()) != {"grey", "phase_rad"}:
        raise ValueError("LUT CSV needs exactly grey,phase_rad headers and measured monotonic unwrapped samples.")
    cal = SLMPhaseCalibration(p["slm"]["panel_id"], p["wavelength_nm"]*1e-9, table["grey"], table["phase_rad"])
    mask = calibrated_phase_to_grey(np.load(folder/"total_phase_rad.npy", allow_pickle=False), cal)
    target = folder/"raw_lut_command.png"
    if target.exists():
        raise ValueError("Raw export already exists; refusing to replace its LUT provenance.")
    Image.fromarray(mask.grey_u8).save(target)
    shutil.copyfile(lut_csv, folder/"export_lut.csv")
    save_json(folder/"raw_export.json", {"lut_sha256": digest(lut_csv), "command_sha256": digest(target),
              "phase_sha256": t["phase_sha256"], "raw_path_verified_by_operator": True,
              "metadata": dict(mask.metadata), "status": "TRIAL_COMMAND_NOT_EXPERIMENTAL_ACCEPTANCE"})
    return {"file": str(target), "instruction": "Display at native resolution through the verified raw grayscale path; no second LUT."}
