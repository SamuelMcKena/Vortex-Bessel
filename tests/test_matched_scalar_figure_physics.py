"""Regression gates for the report-facing scalar comparison."""
from __future__ import annotations

import unittest
import numpy as np

from tools.render_matched_scalar_figures import calculate, _crop


class MatchedScalarFigurePhysicsTest(unittest.TestCase):
    def test_native_512_grid_is_rejected_before_interpolation(self):
        with self.assertRaisesRegex(ValueError, "input phase under-resolved"):
            calculate("V1", 1, grid_n=512, z_m=.06, pad_factor=2)


    def test_matched_routes_are_resolved_and_not_claimed_as_bench_calibration(self):
        ideal, nominal = calculate("V1", 1, grid_n=1024, z_m=.06, pad_factor=2)
        for result in (ideal, nominal):
            self.assertGreaterEqual(result["source_samples_per_period"], 6)
            self.assertGreaterEqual(result["output_samples_per_period"], 12)
            self.assertFalse(result["bench_calibrated"])
            self.assertAlmostEqual(result["z_mm"], 60)
        self.assertEqual(ideal["kr_rad_per_m"], nominal["kr_rad_per_m"])
        self.assertEqual(ideal["output_dx_um"], nominal["output_dx_um"])
        self.assertGreater(ideal["retained_power_au"], nominal["retained_power_au"])
        # Throughput-only fill-factor model, not pixel-border diffraction.
        a = ideal["intensity"] / ideal["retained_power_au"]
        b = nominal["intensity"] / nominal["retained_power_au"]
        overlap = np.sum(a*b) / np.sqrt(np.sum(a*a)*np.sum(b*b))
        self.assertGreater(overlap, .999)


    def test_invalid_sas_distance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SAS validity/output sampling gate failed"):
            calculate("V1", 1, grid_n=1024, z_m=10.0, pad_factor=2)

    def test_vortex_output_stable_when_input_grid_is_refined(self):
        low = calculate("V1", 1, grid_n=1024, z_m=.06, pad_factor=2)
        high = calculate("V1", 1, grid_n=1536, z_m=.06, pad_factor=2)
        for a, b in zip(low, high):
            aa, _ = _crop(a["intensity"]/a["retained_power_au"], a["grid"], .24e-3)
            bb, _ = _crop(b["intensity"]/b["retained_power_au"], b["grid"], .24e-3)
            self.assertEqual(aa.shape, bb.shape)
            self.assertGreater(np.corrcoef(aa.ravel(), bb.ravel())[0, 1], .995)


if __name__ == "__main__":
    unittest.main()
