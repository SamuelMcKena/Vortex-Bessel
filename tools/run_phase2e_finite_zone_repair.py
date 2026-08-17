"""Generate the Phase 2E finite-zone propagation repair."""

from __future__ import annotations

import json

from vbb_study.digital_twin.phase2e_finite_zone_repair import generate_finite_zone_repair


def main() -> None:
    print(json.dumps(generate_finite_zone_repair(), indent=2))


if __name__ == "__main__":
    main()
