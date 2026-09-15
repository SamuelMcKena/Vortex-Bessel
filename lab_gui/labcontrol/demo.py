"""Complete synthetic-through-replay closed-loop recovery demonstration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from slm_lab_control.config import AppConfig

from .beam_walk import PlaneObservation, compare_beam_walk, fit_beam_walk
from .calibration import CalibrationKey, CalibrationRegistry, CalibrationStatus
from .controller import LabController
from .devices.camera import DummyCameraProvider, ReplayCameraProvider
from .devices.stage import DummyStageProvider
from .metrics import BeamMetrics, MetricEngine
from .recipes import RecipeEngine
from .sensorless import OptimisationRun, SensorlessOptimiser
from .state import ExperimentState, ExperimentStore, PhysicalAxiconState, utc_now


Z_PLANES_MM = (31.0, 38.0, 45.0)
SWEEP_DELTAS = (-0.30, -0.15, 0.15, 0.30)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _safe_run_root(output: Path) -> Path:
    if not output.exists() or not any(output.iterdir()):
        output.mkdir(parents=True, exist_ok=True)
        return output
    root = output / f"run_{utc_now().replace(':', '').replace('+', '_')}"
    root.mkdir(parents=True)
    return root


def _base_state(shape_yx: tuple[int, int] = (192, 192)) -> ExperimentState:
    state = ExperimentState.from_app_config(AppConfig())
    state.system.experiment_label = "Synthetic q20 closed-loop demonstration"
    state.system.physical_axicon = PhysicalAxiconState.IN
    state.camera.provider = "replay"
    state.camera.implementation_status = "REPLAY_VALIDATED"
    state.camera.shape_yx = shape_yx
    state.camera.pixel_size_um = 5.5
    state.camera.full_scale = 4095.0
    state.camera.exposure_us = 1000.0
    state.camera.gain = 0.0
    state.camera.settings_id = "synthetic-replay-1000us-g0"
    state.camera.z_reference = "synthetic axicon-tip reference"
    state.slm1.phase.switches.vortex = False
    state.slm1.phase.vortex_charge = 0
    state.slm2.phase.switches.zernike_z40 = True
    state.slm2.phase.z22_cos_amp_waves = 0.0
    return state


def create_synthetic_replay_dataset(root: str | Path, *, repeats: int = 3) -> Path:
    """Create deterministic matrices; this is labelled synthetic everywhere."""

    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Synthetic replay destination must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    store = ExperimentStore(_base_state())
    dummy = DummyCameraProvider(store.snapshot, shape_yx=(192, 192), seed=20260915)
    dummy.connect()

    def save_group(folder: Path, *, charge: int, z_mm: float, astig: float) -> None:
        store.update(
            lambda state: (
                setattr(state.slm1.phase.switches, "vortex", charge != 0),
                setattr(state.slm1.phase, "vortex_charge", charge),
                setattr(state.slm2.phase.switches, "zernike_z40", True),
                setattr(state.slm2.phase, "z22_cos_amp_waves", astig),
                setattr(state.camera, "current_z_mm", z_mm),
            ),
            source="synthetic_fixture",
            reason="Configure deterministic synthetic fixture frame",
        )
        folder.mkdir(parents=True, exist_ok=True)
        for repeat in range(1, repeats + 1):
            frame = dummy.acquire_frame()
            np.save(folder / f"z{z_mm:g}_r{repeat:02d}.npy", frame.data.astype(np.float32))

    for charge, label in ((0, "q0"), (20, "q20")):
        for z in Z_PLANES_MM:
            save_group(root / label / f"z{z:g}", charge=charge, z_mm=z, astig=0.0)
    for role, delta in (
        [("control_start", 0.0)]
        + [(f"candidate_{delta:+.2f}", delta) for delta in SWEEP_DELTAS]
        + [("control_end", 0.0)]
    ):
        save_group(root / "sweep" / role, charge=20, z_mm=38.0, astig=delta)
    for delta in SWEEP_DELTAS:
        save_group(root / "verification" / f"candidate_{delta:+.2f}", charge=20, z_mm=38.0, astig=delta)
    _json(
        root / "dataset_manifest.json",
        {
            "schema_version": 1,
            "data_kind": "SYNTHETIC",
            "warning": "Generated camera-like matrices for software demonstration; not optical propagation or experimental evidence.",
            "shape_yx": [192, 192],
            "z_planes_mm": list(Z_PLANES_MM),
            "repeats": repeats,
            "sweep_deltas": list(SWEEP_DELTAS),
        },
    )
    return root


def _paths(folder: Path) -> list[Path]:
    paths = sorted(folder.glob("*.npy"))
    if not paths:
        raise ValueError(f"No quantitative replay matrices in {folder}")
    return paths


def _replay_metrics(
    controller: LabController,
    paths: Iterable[Path],
    *,
    z_mm: float,
    metric_engine: MetricEngine,
    source_data_kind: str,
) -> list[tuple[str, BeamMetrics]]:
    path_list = list(paths)
    provider = ReplayCameraProvider(
        path_list,
        loop=False,
        exposure_us=1000.0,
        gain=0.0,
        full_scale=4095.0,
        z_by_name={path.name: z_mm for path in path_list},
        source_data_kind=source_data_kind,
    )
    controller.set_camera_provider(provider)
    controller.connect_camera()
    controller.start_camera()
    values = []
    for _ in path_list:
        frame = controller.acquire_frame(fresh=True)
        values.append((frame.frame_id, metric_engine.analyse(frame, controller.store.snapshot())))
    controller.stop_camera()
    provider.disconnect()
    return values


def _walk(
    controller: LabController,
    dataset: Path,
    label: str,
    charge: int,
    metric_engine: MetricEngine,
    source_data_kind: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    controller.store.set_vortex("SLM1", charge != 0, charge, source="recover_q20")
    stage = DummyStageProvider()
    stage.connect()
    observations = []
    metric_rows = []
    for z in Z_PLANES_MM:
        stage.request_move(z)
        stage.confirm_position(z)
        controller.store.set_camera_z(z, source="recover_q20")
        values = _replay_metrics(
            controller,
            _paths(dataset / label / f"z{z:g}"),
            z_mm=z,
            metric_engine=metric_engine,
            source_data_kind=source_data_kind,
        )
        for frame_id, metrics in values:
            observations.append(
                PlaneObservation(
                    frame_id=frame_id,
                    z_mm=z,
                    centre_y_px=metrics.centre_yx_px[0],
                    centre_x_px=metrics.centre_yx_px[1],
                    ring_radius_px=(
                        float(metrics.values["principal_ring_radius_px"])
                        if "principal_ring_radius_px" in metrics.values
                        else None
                    ),
                    power_proxy=float(metrics.values["total_signal"]),
                )
            )
            metric_rows.append(metrics.to_dict())
    result = fit_beam_walk(
        observations,
        pixel_size_um=controller.store.snapshot().camera.pixel_size_um,
        camera_axis_calibrated=False,
    )
    return result.to_dict(), metric_rows


def _sweep_folder(dataset: Path, role: str, delta: float) -> Path:
    return dataset / "sweep" / (role.lower() if role.startswith("CONTROL") else f"candidate_{delta:+.2f}")


def _run_sensorless(
    controller: LabController,
    dataset: Path,
    metric_engine: MetricEngine,
    source_data_kind: str,
) -> tuple[OptimisationRun, bool]:
    optimiser = SensorlessOptimiser(controller.store)
    run = optimiser.plan(
        "astig_x",
        SWEEP_DELTAS,
        seed=7,
        higher_is_better=False,
        minimum_fractional_improvement=0.02,
        max_control_drift_fraction=0.08,
    )
    for trial in list(run.trials):
        optimiser.apply_trial(run, trial.trial_id)
        folder = _sweep_folder(dataset, trial.role, trial.delta)
        values = _replay_metrics(
            controller,
            _paths(folder),
            z_mm=38.0,
            metric_engine=metric_engine,
            source_data_kind=source_data_kind,
        )
        score = float(np.mean([metrics.values["azimuthal_cv"] for _, metrics in values]))
        optimiser.record_score(run, trial.trial_id, score, capture_id=f"replay:{folder}")
    verification = optimiser.recommend(run)
    verification_folder = dataset / "verification" / f"candidate_{verification.delta:+.2f}"
    values = _replay_metrics(
        controller,
        _paths(verification_folder),
        z_mm=38.0,
        metric_engine=metric_engine,
        source_data_kind=source_data_kind,
    )
    score = float(np.mean([metrics.values["azimuthal_cv"] for _, metrics in values]))
    accepted = optimiser.verify_and_decide(
        run, score, capture_id=f"fresh-replay:{verification_folder}"
    )
    return run, accepted


def _seed_prerequisites(registry: CalibrationRegistry, dataset: Path) -> None:
    evidence = (str(dataset / "dataset_manifest.json"),)
    for key in (
        CalibrationKey.SLM_SERIAL_IDENTITY,
        CalibrationKey.SLM_PANEL_GEOMETRY,
        CalibrationKey.SLM_PHASE_PATH,
        CalibrationKey.SLM_NATIVE_ORIENTATION,
        CalibrationKey.CARRIER_CONVENTION,
        CalibrationKey.RELAY_GEOMETRY,
        CalibrationKey.SLM_TO_OPTICAL_MAPPING,
        CalibrationKey.CAMERA_PIXEL_SCALE,
        CalibrationKey.CAMERA_SENSOR_GEOMETRY,
        CalibrationKey.CAMERA_RESPONSE,
        CalibrationKey.CAMERA_TRAVEL_AXIS,
        CalibrationKey.AXICON_PLACEMENT,
    ):
        registry.mark(
            key,
            CalibrationStatus.VALID,
            calibration_id=f"demo-scope-{key.value}",
            evidence=evidence,
            notes="Valid only inside this replay demonstration scope; not current-bench calibration.",
        )


def run_closed_loop_demo(
    output: str | Path,
    *,
    replay_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run recovery headlessly; generated sources remain explicitly synthetic."""

    run_root = _safe_run_root(Path(output))
    if replay_root is None:
        dataset = create_synthetic_replay_dataset(run_root / "synthetic_replay_dataset")
        source_data_kind = "SYNTHETIC"
    else:
        dataset = Path(replay_root)
        manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
        source_data_kind = str(manifest.get("data_kind", "REPLAY"))
    state = _base_state()
    store = ExperimentStore(state)
    controller = LabController(store, run_root)
    metrics = MetricEngine()
    registry = CalibrationRegistry()
    _seed_prerequisites(registry, dataset)
    registry.publish(store, reason="Loaded scoped recovery prerequisites")
    recipe_engine = RecipeEngine(store, registry)
    recipe = recipe_engine.start(
        "recover_q20",
        {"z_mm": list(Z_PLANES_MM), "repeats": 3, "source": str(dataset)},
    )
    recipe_engine.complete_step(recipe, {"physical_axicon": "IN", "source_data_kind": source_data_kind})

    q0_walk, q0_metrics = _walk(controller, dataset, "q0", 0, metrics, source_data_kind)
    recipe_engine.complete_step(recipe, {"fit": q0_walk["fit"]})
    q20_walk, q20_metrics = _walk(controller, dataset, "q20", 20, metrics, source_data_kind)
    recipe_engine.complete_step(recipe, {"fit": q20_walk["fit"]})
    comparison = compare_beam_walk(
        fit_beam_walk(
            [PlaneObservation(**{
                "frame_id": row["frame_id"], "z_mm": row["z_mm"],
                "centre_y_px": row["centre_y_px"], "centre_x_px": row["centre_x_px"],
                "ring_radius_px": row.get("ring_radius_px"), "power_proxy": row.get("power_proxy")
            }) for row in q0_walk["per_frame"]],
            pixel_size_um=5.5,
        ),
        fit_beam_walk(
            [PlaneObservation(**{
                "frame_id": row["frame_id"], "z_mm": row["z_mm"],
                "centre_y_px": row["centre_y_px"], "centre_x_px": row["centre_x_px"],
                "ring_radius_px": row.get("ring_radius_px"), "power_proxy": row.get("power_proxy")
            }) for row in q20_walk["per_frame"]],
            pixel_size_um=5.5,
        ),
    )
    recipe_engine.complete_step(recipe, comparison)

    optimisation, accepted = _run_sensorless(controller, dataset, metrics, source_data_kind)
    recipe_engine.complete_step(
        recipe,
        {
            "run_id": optimisation.run_id,
            "recommended_trial_id": optimisation.recommended_trial_id,
            "terminology": "accepted correction command; not a measured physical aberration",
        },
    )
    recipe_engine.complete_step(
        recipe,
        {"verification_trial_id": optimisation.verification_trial_id, "accepted": accepted},
    )
    recipe_engine.complete_step(recipe, {"decision": "ACCEPT" if accepted else "ROLLBACK"})

    registry.mark(
        CalibrationKey.Q0_PROPAGATION_REFERENCE,
        CalibrationStatus.VALID,
        calibration_id="synthetic-q0-walk",
        evidence=("demo_result.json:q0_walk",),
    )
    registry.mark(
        CalibrationKey.VORTEX_PROPAGATION_REFERENCE,
        CalibrationStatus.VALID,
        calibration_id="synthetic-q20-walk",
        evidence=("demo_result.json:q20_walk",),
    )
    if accepted:
        for key in (
            CalibrationKey.LOW_ORDER_CORRECTION,
            CalibrationKey.GOLDEN_VORTEX_REFERENCE,
            CalibrationKey.VORTEX_VERIFICATION,
        ):
            registry.mark(
                key,
                CalibrationStatus.VALID,
                calibration_id=f"synthetic-{key.value}",
                evidence=(f"sensorless:{optimisation.run_id}",),
                notes="Synthetic/replay evidence only; this does not establish physical lab readiness.",
            )
    readiness = registry.readiness()
    registry.publish(store, reason="Synthetic recovery evidence updated")
    recipe_engine.complete_step(recipe, readiness.to_dict())
    final_bundle = controller.generate()

    validation_scope = (
        "synthetic demonstration only"
        if source_data_kind == "SYNTHETIC"
        else "replay analysis only; not automatically current-lab readiness"
    )
    warning = (
        "This run uses synthetic/replayed numeric matrices and is not physical lab validation."
        if source_data_kind == "SYNTHETIC"
        else "This run re-analyses recorded matrices; it does not prove the current physical bench is ready."
    )
    result = {
        "schema_version": 1,
        "run_root": str(run_root),
        "created_utc": utc_now(),
        "data_kind": source_data_kind,
        "warning": warning,
        "recipe": recipe.to_dict(),
        "q0_walk": q0_walk,
        "q20_walk": q20_walk,
        "walk_comparison": comparison,
        "sensorless": optimisation.to_dict(),
        "accepted": accepted,
        "readiness": {
            **readiness.to_dict(),
            "scope": validation_scope,
        },
        "final_phase_hashes": dict(final_bundle.hashes),
        "slm_cast_status": "NOT_CAST_SYNTHETIC_DEMO",
        "q0_metric_frames": len(q0_metrics),
        "q20_metric_frames": len(q20_metrics),
    }
    _json(run_root / "demo_result.json", result)
    _json(run_root / "final_state.json", store.snapshot().to_dict())
    _json(run_root / "calibration_registry.json", registry.to_dict())
    RecipeEngine.save(recipe, run_root / "recipe_run.json")
    (run_root / "DEMO_REPORT.md").write_text(
        "\n".join(
            [
                "# Synthetic closed-loop recovery demonstration",
                "",
                "**Not physical lab evidence.** Quantitative matrices are deterministic synthetic data routed through ReplayCameraProvider.",
                "",
                f"- q=0 relative walk: {q0_walk['fit']['total_mrad']:.4g} mrad",
                f"- q=20 relative walk: {q20_walk['fit']['total_mrad']:.4g} mrad",
                f"- Classification: {comparison['classification']}",
                f"- Correction decision: {'accepted after fresh verification' if accepted else 'rolled back'}",
                f"- Demonstration readiness: {'READY' if readiness.ready else 'NOT READY'} (synthetic scope only)",
                "- SLM cast: not performed by this replay demonstration",
                "",
                "Beam-walk values are measured beam-camera relative propagation mismatch because the camera travel axis is not a physical calibrated rail.",
                "The selected sensorless value is an accepted correction command, not a measured aberration coefficient.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return result
