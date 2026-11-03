import unittest

from cloth_calibration import build_trials


class ClothCalibrationTests(unittest.TestCase):
    def test_grid_is_complete_and_ids_are_unique(self) -> None:
        trials = build_trials()
        self.assertEqual(len(trials), 27)
        self.assertEqual(len({trial["trial_id"] for trial in trials}), 27)
        self.assertEqual({trial["solver_mass_scale"] for trial in trials}, {1500.0, 2500.0, 4000.0})
        self.assertEqual({trial["damping_scale"] for trial in trials}, {0.8, 1.0, 1.2})
        self.assertEqual({trial["bending_stiffness"] for trial in trials}, {0.15, 0.35, 0.8})


if __name__ == "__main__":
    unittest.main()
