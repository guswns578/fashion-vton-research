from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from measurement_geometry import (
    DEFAULT_BODY_CM,
    DEFAULT_GARMENT_CM,
    body_proxy_from_cm,
    load_pilot_geometry,
    resolve_mass_gsm,
    panel_pattern_from_cm,
    vertex_mass_from_gsm,
)


class MeasurementGeometryTests(unittest.TestCase):
    def test_pattern_preserves_direct_measurements(self) -> None:
        body = body_proxy_from_cm(DEFAULT_BODY_CM)
        pattern = panel_pattern_from_cm(DEFAULT_GARMENT_CM, body)
        self.assertAlmostEqual(pattern.shoulder_half_m * 2, 0.48)
        self.assertAlmostEqual(pattern.chest_half_m * 2, 0.55)
        self.assertAlmostEqual(pattern.hem_half_m * 2, 0.55)
        self.assertAlmostEqual(pattern.shoulder_z_m - pattern.hem_z_m, 0.70)
        self.assertEqual(len(pattern.outline_xz_m), 13)
        for left, right in zip(pattern.outline_xz_m[:6], reversed(pattern.outline_xz_m[6:12])):
            self.assertAlmostEqual(left[0], -right[0])
            self.assertAlmostEqual(left[1], right[1])
        self.assertAlmostEqual(pattern.outline_xz_m[0][1] - pattern.hem_z_m, 0.70)
        self.assertEqual(pattern.outline_xz_m[-1][0], 0.0)

    def test_body_proxy_matches_chest_ellipse_contract(self) -> None:
        body = body_proxy_from_cm(DEFAULT_BODY_CM)
        self.assertGreater(body.torso_radius_x_m, body.torso_radius_y_m)
        self.assertGreater(body.arm_half_length_m, 0.25)
        self.assertGreater(body.arm_radius_m, 0.04)

    def test_rejects_garment_tighter_than_body(self) -> None:
        body = body_proxy_from_cm(DEFAULT_BODY_CM)
        garment = dict(DEFAULT_GARMENT_CM)
        garment["chest_flat"] = 40.0
        with self.assertRaisesRegex(ValueError, "Garment circumference"):
            panel_pattern_from_cm(garment, body)

    def test_loads_complete_pilot(self) -> None:
        payload = {
            "subject": {"measurements_cm": DEFAULT_BODY_CM},
            "garment": {"measurements_cm": DEFAULT_GARMENT_CM},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pilot.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            body, pattern, source = load_pilot_geometry(path)
        self.assertEqual(body.source_cm["height"], 172.0)
        self.assertEqual(pattern.source_cm["body_length"], 70.0)
        self.assertTrue(source.endswith("pilot.json"))

    def test_vertex_mass_reconstructs_total_gsm_mass(self) -> None:
        vertex_mass, total_mass = vertex_mass_from_gsm(1.25, 2000, 200.0)
        self.assertAlmostEqual(total_mass, 0.25)
        self.assertAlmostEqual(vertex_mass * 2000, total_mass)

    def test_mass_resolution_precedence(self) -> None:
        value, source = resolve_mass_gsm({"mass_gsm": 180, "garment_mass_g": 300}, 1.5, 220)
        self.assertEqual((value, source), (180.0, "pilot.material.mass_gsm"))
        value, source = resolve_mass_gsm({"mass_gsm": None, "garment_mass_g": 300}, 1.5, 220)
        self.assertAlmostEqual(value, 200.0)
        self.assertIn("garment_mass_g", source)
        self.assertEqual(resolve_mass_gsm({}, 1.5, 220), (220, "preset_fallback"))


if __name__ == "__main__":
    unittest.main()
