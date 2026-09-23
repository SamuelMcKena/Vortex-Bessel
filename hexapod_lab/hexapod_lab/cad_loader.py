from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any


@dataclass(frozen=True, slots=True)
class CadCacheResult:
    source_path: Path
    cache_dir: Path
    mesh_paths: tuple[Path, ...]
    source_sha256: str


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build_step_mesh_cache(
    step_path: str | Path,
    *,
    cache_root: str | Path | None = None,
    tolerance_mm: float = 2.0,
    angular_tolerance_rad: float = 0.8,
) -> CadCacheResult:
    """Tessellate the supplied STEP assembly into one cached mesh per solid.

    CadQuery is imported lazily so the rest of the controller can operate with the
    fast CAD-derived rig even when CadQuery is not installed.
    """

    try:
        from cadquery import exporters, importers
    except ImportError as exc:
        raise RuntimeError(
            "Exact STEP rendering needs CadQuery. Install requirements-cad.txt or use the built-in CAD-derived rig."
        ) from exc

    src = Path(step_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    digest = sha256_file(src)
    root = Path(cache_root) if cache_root is not None else Path(tempfile.gettempdir()) / "hexapod_lab_cad_cache"
    cache = root / digest[:16]
    cache.mkdir(parents=True, exist_ok=True)
    manifest_path = cache / "manifest.json"

    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            paths = tuple(cache / p for p in manifest["meshes"])
            if manifest.get("source_sha256") == digest and all(p.is_file() for p in paths):
                return CadCacheResult(src, cache, paths, digest)
        except Exception:
            pass

    obj = importers.importStep(str(src)).val()
    solids = obj.Solids()
    if len(solids) < 8:
        raise RuntimeError(f"STEP file contains only {len(solids)} solids; expected a Stewart-platform assembly")

    mesh_paths: list[Path] = []
    for i, solid in enumerate(solids):
        out = cache / f"solid_{i:02d}.stl"
        exporters.export(
            solid,
            str(out),
            tolerance=float(tolerance_mm),
            angularTolerance=float(angular_tolerance_rad),
        )
        mesh_paths.append(out)

    manifest_path.write_text(
        json.dumps(
            {
                "source": src.name,
                "source_sha256": digest,
                "solid_count": len(solids),
                "tolerance_mm": float(tolerance_mm),
                "angular_tolerance_rad": float(angular_tolerance_rad),
                "meshes": [p.name for p in mesh_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return CadCacheResult(src, cache, tuple(mesh_paths), digest)


def inspect_step_geometry(step_path: str | Path) -> dict[str, Any]:
    """Return a compact, reproducible geometry summary for diagnostics."""

    try:
        from cadquery import importers
    except ImportError as exc:
        raise RuntimeError("CadQuery is required for STEP inspection") from exc

    src = Path(step_path).expanduser().resolve()
    obj = importers.importStep(str(src)).val()
    solids = obj.Solids()
    rows = []
    for index, solid in enumerate(solids):
        centre = solid.Center()
        bb = solid.BoundingBox()
        rows.append(
            {
                "index": index,
                "volume_mm3": float(solid.Volume()),
                "center_mm": [float(centre.x), float(centre.y), float(centre.z)],
                "bbox_mm": [float(bb.xlen), float(bb.ylen), float(bb.zlen)],
            }
        )
    return {
        "source": src.name,
        "sha256": sha256_file(src),
        "solid_count": len(solids),
        "solids": rows,
    }
