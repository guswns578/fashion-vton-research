import unittest

from capture_alignment import compare_points


class CaptureAlignmentTests(unittest.TestCase):
    def points(self, dx=0.0, dy=0.0):
        names = [
            "neck", "shoulder_r", "elbow_r", "wrist_r", "shoulder_l",
            "elbow_l", "wrist_l", "hip_r", "hip_l", "knee_r", "knee_l",
        ]
        return {name: (0.3 + index * 0.03 + dx, 0.2 + index * 0.05 + dy) for index, name in enumerate(names)}

    def test_small_shift_passes(self):
        result = compare_points(self.points(), self.points(dx=0.003), 1000, 1500)
        self.assertEqual(result["status"], "pass")

    def test_large_shift_fails(self):
        result = compare_points(self.points(), self.points(dx=0.08), 1000, 1500)
        self.assertEqual(result["status"], "fail")

    def test_too_few_points_fails(self):
        left = {"neck": (0.5, 0.2), "shoulder_l": (0.4, 0.3)}
        result = compare_points(left, left, 1000, 1500)
        self.assertEqual(result["status"], "fail")


if __name__ == "__main__":
    unittest.main()
