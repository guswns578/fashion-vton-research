import unittest

from garmentcode_mesh_contract import (
    edge_segment_counts,
    harmonic_preclose_vertices,
    point_in_polygon,
    preclose_vertices,
    relax_preclosed_vertices,
    structural_edge_ratios,
)


class GarmentCodeMeshContractTests(unittest.TestCase):
    def test_point_in_concave_polygon(self) -> None:
        polygon = [(0, 0), (2, 0), (2, 2), (1, 1), (0, 2)]
        self.assertTrue(point_in_polygon((0.5, 1.0), polygon))
        self.assertFalse(point_in_polygon((1.0, 1.75), polygon))
        self.assertTrue(point_in_polygon((1.0, 1.0), polygon))

    def test_stitched_edges_share_segment_count(self) -> None:
        pattern = {
            "panels": {
                "a": {"vertices": [[0, 0], [10, 0]], "edges": [{"endpoints": [0, 1]}]},
                "b": {"vertices": [[0, 0], [12, 0]], "edges": [{"endpoints": [0, 1]}]},
            },
            "stitches": [[{"panel": "a", "edge": 0}, {"panel": "b", "edge": 0}]],
        }
        counts = edge_segment_counts(pattern, 2.0)
        self.assertEqual(counts[("a", 0)], 6)
        self.assertEqual(counts[("b", 0)], 6)

    def test_preclose_keeps_duplicate_topology_and_joins_transitive_seams(self) -> None:
        vertices = [[-1, 0, 0], [1, 0, 0], [3, 0, 0], [8, 0, 0]]
        result = preclose_vertices(vertices, [(0, 1), (1, 2)])
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], [1.0, 0.0, 0.0])
        self.assertEqual(result[1], [1.0, 0.0, 0.0])
        self.assertEqual(result[2], [1.0, 0.0, 0.0])
        self.assertEqual(result[3], [8.0, 0.0, 0.0])

    def test_harmonic_preclose_spreads_seam_motion_through_each_panel(self) -> None:
        vertices = [
            [0, 0, 0], [1, 0, 0], [0, 1, 0],
            [3, 0, 0], [4, 0, 0], [3, 1, 0],
        ]
        result = harmonic_preclose_vertices(
            vertices,
            [[0, 1, 2], [3, 4, 5]],
            {"left": [0, 3], "right": [3, 6]},
            [(1, 3)],
        )
        self.assertEqual(result[1], result[3])
        self.assertAlmostEqual(result[1][0], 2.0)
        self.assertAlmostEqual(result[1][0] - result[0][0], 1.0)
        self.assertAlmostEqual(result[4][0] - result[3][0], 1.0)

    def test_relaxation_reduces_structural_strain_without_opening_seam(self) -> None:
        rest = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0], [4, 0, 0], [3, 1, 0]]
        faces = [[0, 1, 2], [3, 4, 5]]
        preclosed = [list(point) for point in rest]
        preclosed[1] = [2.0, 0.0, 0.0]
        preclosed[3] = [2.0, 0.0, 0.0]
        before = structural_edge_ratios(preclosed, rest, faces)
        result = relax_preclosed_vertices(
            preclosed, rest, faces, [(1, 3)], iterations=200, tether=0.0
        )
        after = structural_edge_ratios(result, rest, faces)
        self.assertLess(abs(after - 1.0).mean(), abs(before - 1.0).mean())
        self.assertEqual(result[1], result[3])


if __name__ == "__main__":
    unittest.main()
