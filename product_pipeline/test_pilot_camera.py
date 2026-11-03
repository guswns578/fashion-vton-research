import unittest

from pilot_camera import build_camera


class PilotCameraTests(unittest.TestCase):
    def test_builds_metric_perspective_camera(self) -> None:
        camera = build_camera(172.0, 0.35, 300.0, 26.0, (3024, 4032))
        self.assertEqual(camera["projection"], "perspective")
        self.assertEqual(camera["location"][1], -3.0)
        self.assertEqual(camera["focal_length_mm"], 26.0)
        self.assertEqual(camera["reference_resolution"], [3024, 4032])
        self.assertAlmostEqual(camera["target"][2], 0.35 - 0.32 * 1.72)

    def test_rejects_nonpositive_capture_values(self) -> None:
        with self.assertRaises(ValueError):
            build_camera(172.0, 0.35, 0.0, 26.0, (3024, 4032))


if __name__ == "__main__":
    unittest.main()
