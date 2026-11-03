import unittest

import numpy as np

from compare_cloth_runs import mask_iou


class CompareClothRunsTests(unittest.TestCase):
    def test_mask_iou(self) -> None:
        left = np.asarray([[True, True], [False, False]])
        right = np.asarray([[True, False], [True, False]])
        self.assertAlmostEqual(mask_iou(left, right), 1 / 3)

    def test_empty_masks_are_identical(self) -> None:
        empty = np.zeros((2, 3), dtype=bool)
        self.assertEqual(mask_iou(empty, empty), 1.0)


if __name__ == "__main__":
    unittest.main()
