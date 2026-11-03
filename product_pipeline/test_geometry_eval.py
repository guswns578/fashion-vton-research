import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from geometry_eval import evaluate


class GeometryEvalTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def save_mask(self, name, box):
        mask = np.zeros((200, 160), dtype=np.uint8)
        x0, y0, x1, y1 = box
        mask[y0:y1, x0:x1] = 255
        path = self.root / name
        Image.fromarray(mask).save(path)
        return path

    def save_landmarks(self, name, x_shift=0.0):
        points = {
            "shoulder_l": [0.35 + x_shift, 0.25],
            "shoulder_r": [0.65 + x_shift, 0.25],
            "cuff_l": [0.25 + x_shift, 0.55],
            "cuff_r": [0.75 + x_shift, 0.55],
            "hem_l": [0.38 + x_shift, 0.75],
            "hem_r": [0.62 + x_shift, 0.75]
        }
        path = self.root / name
        path.write_text(json.dumps({"landmarks": points}), encoding="utf-8")
        return path

    def test_identical_geometry_passes(self):
        real = self.save_mask("real.png", (40, 40, 120, 160))
        rendered = self.save_mask("render.png", (40, 40, 120, 160))
        real_lm = self.save_landmarks("real.json")
        render_lm = self.save_landmarks("render.json")
        report = evaluate(real, rendered, real_lm, render_lm, self.root / "overlay.png", 0.8, 0.05)
        self.assertEqual(report["decision"]["status"], "pass")
        self.assertEqual(report["mask_iou"], 1.0)
        self.assertTrue((self.root / "overlay.png").exists())

    def test_shifted_geometry_fails(self):
        real = self.save_mask("real.png", (40, 40, 120, 160))
        rendered = self.save_mask("render.png", (70, 40, 150, 160))
        real_lm = self.save_landmarks("real.json")
        render_lm = self.save_landmarks("render.json", x_shift=0.2)
        report = evaluate(real, rendered, real_lm, render_lm, self.root / "overlay.png", 0.8, 0.05)
        self.assertEqual(report["decision"]["status"], "fail")
        self.assertLess(report["mask_iou"], 0.8)

    def test_missing_landmarks_is_incomplete(self):
        real = self.save_mask("real.png", (40, 40, 120, 160))
        rendered = self.save_mask("render.png", (40, 40, 120, 160))
        report = evaluate(real, rendered, None, None, self.root / "overlay.png", 0.8, 0.05)
        self.assertEqual(report["decision"]["status"], "incomplete")
        self.assertFalse(report["decision"]["overall_pass"])


if __name__ == "__main__":
    unittest.main()
