"""Generate the PHASE 2B visual diagnostics and beam-volume package."""

from __future__ import annotations

import json

from vbb_study.digital_twin.phase2b_visual_diagnostics import write_phase2b_outputs


def main() -> None:
    result = write_phase2b_outputs()
    print(json.dumps(result["outcome"], indent=2))


if __name__ == "__main__":
    main()
