from __future__ import annotations

import argparse
import json
from pathlib import Path

from hexapod_lab.cad_loader import inspect_step_geometry


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a Stewart-platform STEP assembly")
    parser.add_argument("step", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data = inspect_step_geometry(args.step)
    text = json.dumps(data, indent=2)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
