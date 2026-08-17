"""Generate the isolated Phase 2E propagation forensic repair package."""

from __future__ import annotations

import json

from vbb_study.digital_twin.phase2e_propagation_repair import (
    generate_phase2e_propagation_repair,
)


def main() -> None:
    outcome = generate_phase2e_propagation_repair()
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
