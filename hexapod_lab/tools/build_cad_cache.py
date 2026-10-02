from __future__ import annotations

import argparse
from pathlib import Path

from hexapod_lab.cad_loader import build_step_mesh_cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Tessellate the Stewart-platform STEP into cached per-solid meshes")
    parser.add_argument("step", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--tolerance-mm", type=float, default=2.0)
    parser.add_argument("--angular-tolerance-rad", type=float, default=0.8)
    args = parser.parse_args()
    result = build_step_mesh_cache(
        args.step,
        cache_root=args.cache,
        tolerance_mm=args.tolerance_mm,
        angular_tolerance_rad=args.angular_tolerance_rad,
    )
    print(f"source  : {result.source_path}")
    print(f"sha256  : {result.source_sha256}")
    print(f"cache   : {result.cache_dir}")
    print(f"solids  : {len(result.mesh_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
