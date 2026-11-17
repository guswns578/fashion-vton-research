import unittest

from garmentcode_adapter import body_overrides, design_overrides, update_calibration
from measurement_geometry import DEFAULT_BODY_CM, DEFAULT_GARMENT_CM


class GarmentCodeAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.neutral = {
            "height": 171.99,
            "back_width": 47.6761,
            "shoulder_w": 36.4568,
            "head_l": 26.3262,
            "waist_line": 36.8913,
        }

    def test_body_direct_mapping(self) -> None:
        mapped, contract = body_overrides(DEFAULT_BODY_CM, self.neutral)
        self.assertEqual(mapped["height"], 172.0)
        self.assertEqual(mapped["bust"], 92.0)
        self.assertEqual(mapped["shoulder_w"], 42.0)
        self.assertEqual(mapped["waist_line"], 44.0)
        self.assertIn("back_width", contract["derived"])

    def test_design_targets_long_sleeve_and_chest_ease(self) -> None:
        design = {
            "meta": {"upper": {"v": None}, "bottom": {"v": "Pants"}, "wb": {"v": "StraightWB"}},
            "shirt": {"width": {"v": 1.0}, "length": {"v": 1.0}, "flare": {"v": 1.0}},
            "sleeve": {
                "sleeveless": {"v": True},
                "armhole_shape": {"v": "ArmholeSquare"},
                "length": {"v": 0.3},
                "end_width": {"v": 1.0},
            },
        }
        mapped, _ = design_overrides(
            design, DEFAULT_BODY_CM, DEFAULT_GARMENT_CM, {"waist_line": 44.0}
        )
        self.assertAlmostEqual(mapped["shirt"]["width"]["v"], 110 / 92)
        self.assertAlmostEqual(mapped["shirt"]["length"]["v"], 70 / 44)
        self.assertEqual(mapped["sleeve"]["length"]["v"], 1.0)
        self.assertFalse(mapped["sleeve"]["sleeveless"]["v"])

    def test_calibration_update_corrects_all_supported_axes(self) -> None:
        design = {
            "shirt": {"width": {"v": 1.2}, "length": {"v": 1.5}, "flare": {"v": 1.2}},
            "sleeve": {
                "length": {"v": 1.0},
                "end_width": {"v": 0.5},
                "standing_shoulder": {"v": False},
                "standing_shoulder_len": {"v": 5.0},
                "sleeve_angle": {"v": 10.0},
            },
        }
        realized = {
            "shoulder": 40.0,
            "chest_flat": 60.0,
            "body_length": 74.0,
            "sleeve_length": 58.0,
            "cuff_flat": 10.0,
            "hem_flat": 66.0,
        }
        update_calibration(
            design, {"waist_line": 44.0, "arm_length": 60.0}, DEFAULT_GARMENT_CM, realized
        )
        self.assertAlmostEqual(design["shirt"]["width"]["v"], 1.1)
        self.assertAlmostEqual(design["shirt"]["flare"]["v"], 1.2 * 55 / 66 / (55 / 60))
        self.assertTrue(design["sleeve"]["standing_shoulder"]["v"])
        self.assertAlmostEqual(design["sleeve"]["standing_shoulder_len"]["v"], 4.0)


if __name__ == "__main__":
    unittest.main()
