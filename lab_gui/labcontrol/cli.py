"""Headless CLI for state, capture, beam-walk, replay and recipes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from slm_lab_control.config import AppConfig

from .beam_walk import PlaneObservation, fit_beam_walk
from .calibration import CalibrationRegistry
from .capture import FormalCaptureService, SessionRepository
from .controller import LabController
from .demo import run_closed_loop_demo
from .devices.camera import DummyCameraProvider, ReplayCameraProvider
from .metrics import MetricEngine
from .reporting import build_session_report
from .state import ExperimentState, ExperimentStore


def _base_controller(root: Path) -> LabController:
    state = ExperimentState.from_app_config(AppConfig())
    state.application.output_root = str(root / "casts")
    store = ExperimentStore(state)
    return LabController(store, root)


def _infer_z(path: Path) -> float:
    match = re.search(r"(?:^|[^0-9])z?(\d+(?:\.\d+)?)(?:[^0-9]|$)", path.stem, re.I)
    if not match:
        raise ValueError(f"Cannot infer physical z in mm from {path.name}; include zNN in the name.")
    return float(match.group(1))


def command_status(args) -> int:
    if args.state:
        store = ExperimentStore.load(args.state)
    else:
        store = ExperimentStore(ExperimentState.from_app_config(AppConfig()))
    registry = CalibrationRegistry.load(args.calibration) if args.calibration else CalibrationRegistry()
    payload = {
        "state": store.snapshot().to_dict(),
        "readiness": registry.readiness().to_dict(),
    }
    print(json.dumps(payload, indent=2))
    return 0


def command_capture(args) -> int:
    root = Path(args.output)
    controller = _base_controller(root)
    if args.camera == "dummy":
        provider = DummyCameraProvider(controller.store.snapshot, shape_yx=(512, 512))
    else:
        if not args.replay:
            raise ValueError("--replay is required for the replay camera.")
        provider = ReplayCameraProvider(args.replay, source_data_kind=args.source_data_kind)
    controller.set_camera_provider(provider)
    controller.store.update(
        lambda state: (
            setattr(state.camera, "settings_id", args.settings_id),
            setattr(state.camera, "z_reference", args.z_reference),
            setattr(state.camera, "current_z_mm", args.z_mm),
        ),
        source="cli",
        reason="Configured formal CLI capture",
    )
    controller.connect_camera()
    controller.start_camera()
    try:
        record = FormalCaptureService(
            controller, SessionRepository(root / "session")
        ).capture(
            args.trial,
            repeats=args.repeats,
            role=args.role,
            settle_s=args.settle,
        )
    finally:
        controller.stop_camera()
    print(record.trial_root)
    return 0


def command_beam_walk(args) -> int:
    folder = Path(args.folder)
    paths = sorted(
        path
        for path in folder.iterdir()
        if path.suffix.lower() in {".bmg", ".npy", ".txt", ".csv", ".bmp", ".png", ".tif", ".tiff"}
    )
    if not paths:
        raise ValueError(f"No quantitative frames in {folder}")
    z_map = {path.name: _infer_z(path) for path in paths}
    controller = _base_controller(Path(args.output))
    controller.store.set_vortex("SLM1", args.charge != 0, args.charge, source="cli")
    controller.set_camera_provider(
        ReplayCameraProvider(
            paths,
            loop=False,
            z_by_name=z_map,
            source_data_kind=args.source_data_kind,
        )
    )
    controller.connect_camera()
    controller.start_camera()
    metric_engine = MetricEngine()
    observations = []
    for _ in paths:
        frame = controller.acquire_frame()
        metrics = metric_engine.analyse(frame, controller.store.snapshot())
        observations.append(
            PlaneObservation(
                frame.frame_id,
                float(frame.z_mm),
                metrics.centre_yx_px[0],
                metrics.centre_yx_px[1],
                metrics.values.get("principal_ring_radius_px"),
                float(metrics.values["total_signal"]),
            )
        )
    controller.stop_camera()
    result = fit_beam_walk(
        observations,
        pixel_size_um=args.pixel_um,
        camera_axis_calibrated=args.camera_axis_calibrated,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "beam_walk.json"
    target.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(result.fit, indent=2))
    print(target)
    return 0


def command_recover(args) -> int:
    replay_root = Path(args.replay_root) if args.replay_root else None
    result = run_closed_loop_demo(args.output, replay_root=replay_root)
    print(json.dumps({
        "run_root": result["run_root"],
        "data_kind": result["data_kind"],
        "accepted": result["accepted"],
        "readiness": result["readiness"],
        "warning": result["warning"],
    }, indent=2))
    return 0


def command_replay_session(args) -> int:
    report = build_session_report(args.session)
    print(report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="labcontrol", description="Headless optical experiment control")
    commands = parser.add_subparsers(dest="command", required=True)
    status = commands.add_parser("status", help="Print authoritative state and readiness")
    status.add_argument("--state", type=Path)
    status.add_argument("--calibration", type=Path)
    status.set_defaults(function=command_status)

    capture = commands.add_parser("capture", help="Perform a formal provenance-linked capture")
    capture.add_argument("--output", required=True)
    capture.add_argument("--trial", default="trial_0001")
    capture.add_argument("--camera", choices=("dummy", "replay"), default="dummy")
    capture.add_argument("--replay", nargs="+")
    capture.add_argument("--source-data-kind", choices=("REPLAY", "EXPERIMENT", "SYNTHETIC"), default="REPLAY")
    capture.add_argument("--settings-id", default="cli-fixed-settings")
    capture.add_argument("--z-mm", type=float, default=38.0)
    capture.add_argument("--z-reference", default="operator-declared reference")
    capture.add_argument("--repeats", type=int, default=3)
    capture.add_argument("--settle", type=float, default=0.1)
    capture.add_argument("--role", default="CURRENT")
    capture.set_defaults(function=command_capture)

    analyse = commands.add_parser("analyse", help="Analyse quantitative data")
    analyse_commands = analyse.add_subparsers(dest="analysis", required=True)
    walk = analyse_commands.add_parser("beam-walk")
    walk.add_argument("folder")
    walk.add_argument("--output", required=True)
    walk.add_argument("--pixel-um", type=float, default=5.5)
    walk.add_argument("--charge", type=int, default=0)
    walk.add_argument("--source-data-kind", choices=("REPLAY", "EXPERIMENT", "SYNTHETIC"), default="REPLAY")
    walk.add_argument("--camera-axis-calibrated", action="store_true")
    walk.set_defaults(function=command_beam_walk)

    run = commands.add_parser("run", help="Run an experiment recipe")
    run_commands = run.add_subparsers(dest="recipe", required=True)
    recovery = run_commands.add_parser("recover-q20", help="Run q20 closed-loop recovery in replay mode")
    recovery.add_argument("--camera", choices=("replay",), default="replay")
    recovery.add_argument("--replay-root", help="Optional compatible replay dataset; default generates labelled synthetic matrices")
    recovery.add_argument("--output", required=True)
    recovery.set_defaults(function=command_recover)

    replay = commands.add_parser("replay", help="Build a human-readable report from a formal session")
    replay.add_argument("session")
    replay.set_defaults(function=command_replay_session)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except Exception as exc:
        parser.exit(2, f"labcontrol: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
