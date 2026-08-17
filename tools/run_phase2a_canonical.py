"""Generate PHASE 2A canonical lab-realism artifacts."""

from __future__ import annotations

import json

from vbb_study.digital_twin.phase2a_canonical import write_phase2a_outputs


def main() -> None:
    result = write_phase2a_outputs()
    print(json.dumps(result["outcome"], indent=2))


if __name__ == "__main__":
    main()
