#!/usr/bin/env python
"""Read-only/low-impact diagnostic for the PC-Beamage named-pipe provider."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAB_GUI = ROOT / "lab_gui"
for candidate in (str(LAB_GUI), str(ROOT)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from labcontrol.devices.beamage_pipe import BeamagePipeClient  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also request one vendor BMP and print its shape/safety metadata.",
    )
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args(argv)

    client = BeamagePipeClient(timeout_s=args.timeout)
    try:
        print(f"Connecting to PC-Beamage at {client.pipe_path} ...")
        client.connect()
        print(f"Identity: {client.probe_identity()}")
        print("Capture state:", client.get_capture_state())
        print("Measurements:", json.dumps(client.measurements(), indent=2, sort_keys=True))
        print("Positions:", json.dumps(client.positions(), indent=2, sort_keys=True))
        if args.preview:
            client.start()
            array, metadata = client.read_quantitative_frame(args.timeout)
            print(f"Preview shape: {array.shape}; dtype after ingestion: {array.dtype}")
            print("Preview safety:", metadata["measurement_scope"], metadata["warning"])
            client.stop()
        print("PIPE CHECK PASSED (communication only; quantitative BMP validation remains required).")
        return 0
    except Exception as exc:
        print(f"PIPE CHECK FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
