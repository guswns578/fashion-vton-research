import math
import unittest

from garmentcode_pattern_audit import edge_length, edge_point, edge_point_by_length_fraction


class GarmentCodePatternAuditTests(unittest.TestCase):
    def test_line_length(self) -> None:
        panel = {"vertices": [[0, 0], [3, 4]], "edges": [{"endpoints": [0, 1]}]}
        self.assertAlmostEqual(edge_length(panel, 0), 5.0)

    def test_straight_cubic_length(self) -> None:
        panel = {
            "vertices": [[0, 0], [10, 0]],
            "edges": [
                {
                    "endpoints": [0, 1],
                    "curvature": {"type": "cubic", "params": [[1 / 3, 0], [2 / 3, 0]]},
                }
            ],
        }
        self.assertAlmostEqual(edge_length(panel, 0), 10.0, places=5)

    def test_semicircle_length(self) -> None:
        panel = {
            "vertices": [[-2, 0], [2, 0]],
            "edges": [{"endpoints": [0, 1], "curvature": {"type": "circle", "params": [2, 0, 1]}}],
        }
        self.assertAlmostEqual(edge_length(panel, 0), 2 * 3.141592653589793, places=6)
        midpoint = edge_point(panel, 0, 0.5)
        self.assertAlmostEqual(midpoint[0], 0.0, places=6)
        self.assertAlmostEqual(abs(midpoint[1]), 2.0, places=6)

    def test_quadratic_arc_length_sampling_is_nearly_uniform(self) -> None:
        panel = {
            "vertices": [[0.0, 0.0], [10.0, 0.0]],
            "edges": [{"endpoints": [0, 1], "curvature": [0.05, 1.2]}],
        }
        points = [edge_point_by_length_fraction(panel, 0, index / 8) for index in range(9)]
        distances = [math.dist(left, right) for left, right in zip(points, points[1:])]
        self.assertLess(max(distances) / min(distances), 1.05)


if __name__ == "__main__":
    unittest.main()
