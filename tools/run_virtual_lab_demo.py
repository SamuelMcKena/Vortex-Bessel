"""Headless smoke/demo runner for the Unified GUI Virtual Lab path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB_GUI = ROOT / "lab_gui"
for path in (ROOT, LAB_GUI):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from slm_lab_control.config import AppConfig

from labcontrol.devices.camera import DummyCameraProvider
from labcontrol.mode_controller import ModeAwareLabController
from labcontrol.state import ExperimentState, ExperimentStore
from labcontrol.virtual_lab import (
    BlindCorrectionRunner,
    OperatingMode,
    ScenarioKind,
    VirtualBenchEngine,
    VirtualLabGeometry,
    capture_virtual_z_stack,
)


def parse_z_plan(text: str) -> tuple[float, ...]:
    values = tuple(float(v.strip()) for v in text.split(",") if v.strip())
    if len(values) < 2:
        raise argparse.ArgumentTypeError("--z must contain at least two comma-separated positions")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the same VirtualCameraProvider/CameraFrame route used by the GUI."
    )
    parser.add_argument("--scenario", choices=[x.value for x in ScenarioKind], default="MIXED")
    parser.add_argument("--seed", type=int, default=1048)
    parser.add_argument("--charge", type=int, default=20)
    parser.add_argument("--z", type=parse_z_plan, default=parse_z_plan("31,34,37,40,43,46"))
    parser.add_argument("--quality", choices=("preview", "validation"), default="preview")
    parser.add_argument("--optimise", action="store_true", help="Run one SLM2 low-order correction pass.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "virtual_lab" / "headless_demo.json",
    )
    args = parser.parse_args()

    state = ExperimentState.from_app_config(AppConfig())
    store = ExperimentStore(state)
    engine = VirtualBenchEngine(VirtualLabGeometry(), quality=args.quality)
    controller = ModeAwareLabController(
        store,
        LAB_GUI,
        camera_provider=DummyCameraProvider(store.snapshot, shape_yx=(64, 64)),
        virtual_engine=engine,
    )
    controller.set_operating_mode(OperatingMode.VIRTUAL_LAB)
    store.set_vortex("SLM1", bool(args.charge), int(args.charge), source="virtual_demo")
    handle = engine.generate_scenario(args.scenario, args.seed)
    controller.cast(("SLM1", "SLM2"), persist=False)

    baseline = capture_virtual_z_stack(controller, args.z)
    payload = {
        "mode": controller.operating_mode.value,
        "scenario": {
            "scenario_id": handle.scenario_id,
            "kind": handle.kind,
            "seed": handle.seed,
            "truth_hash": handle.truth_hash,
        },
        "geometry": engine.geometry.public_summary(),
        "baseline": baseline.summary(),
    }

    if args.optimise:
        runner = BlindCorrectionRunner(controller)
        result = runner.run(
            args.z,
            targets=("SLM2",),
            parameters=("astig_x", "astig_xy", "coma_x", "coma_y"),
            probe_amplitude_waves=0.15,
            passes=1,
            seed=args.seed,
        )
        payload["correction"] = result.to_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
