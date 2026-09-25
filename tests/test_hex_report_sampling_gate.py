"""A legacy hexagon class is not, by itself, a resolved report figure."""
from __future__ import annotations

import unittest
from dataclasses import replace

from vbb_study.digital_twin.nathan_vector_hexagon import (
    NathanSourceParityConfig,
    mode2n_source_target,
    mode2q_strict_hexagon_gate,
    run_mode2n_v0_reference,
)


class HexReportSamplingGateTest(unittest.TestCase):
    def test_under_resolved_source_scale_hex_is_not_report_eligible(self):
        cfg = replace(NathanSourceParityConfig(), grid_n=384, z_planes=3,
                      z_start_m=.059, z_end_m=.061)
        data = mode2n_source_target(cfg, grid_n=384, z_planes=3)
        v0 = run_mode2n_v0_reference(data)
        gate = mode2q_strict_hexagon_gate(v0.reference_plane, data["grid"])
        self.assertLess(gate["ring_radius_pixels"], 12)
        self.assertFalse(gate["report_sampling_ok"])
        self.assertFalse(gate["report_eligible_hexagon"])
        self.assertIn("low-resolution", gate["report_claim_boundary"])


if __name__ == "__main__":
    unittest.main()
