"""Finalize the Phase 2E finite-zone repair after pre-formation extension."""

from __future__ import annotations

import json

from vbb_study.digital_twin.phase2e_finite_zone_repair import finalize_finite_zone_repair


if __name__ == "__main__":
    print(json.dumps(finalize_finite_zone_repair(), indent=2))
