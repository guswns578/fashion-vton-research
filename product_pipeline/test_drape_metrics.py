import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from drape_metrics import evaluate_manifest, measure_image
from catalog_eval import hard_composite


class DrapeMetricsTests(unittest.TestCase):
    def test_meso_frequency_responds_to_wrinkles(self):
        n = 256
        mask = np.zeros((n, n), np.float32)
        mask[24:232, 40:216] = 1
        flat = np.full((n, n, 3), 128, np.uint8)
        x = np.arange(n)[None, :]
        wave = 35 * np.sin(2 * np.pi * x / 16)
        wrinkled = np.clip(flat.astype(float) + wave[..., None], 0, 255).astype(np.uint8)
        a = measure_image(Image.fromarray(flat), mask, size=256)
        b = measure_image(Image.fromarray(wrinkled), mask, size=256)
        self.assertGreater(b["meso_energy_ratio"], a["meso_energy_ratio"])
        self.assertGreater(b["fold_contrast"], a["fold_contrast"])

    def test_hem_curvature_detects_curved_bottom(self):
        n = 256
        flat = np.zeros((n, n), np.float32)
        curved = np.zeros_like(flat)
        flat[30:200, 40:216] = 1
        for x in range(40, 216):
            xn = (x - 128) / 88
            bottom = round(190 + 22 * (1 - xn * xn))
            curved[30:bottom, x] = 1
        im = Image.new("RGB", (n, n), (128, 128, 128))
        a = measure_image(im, flat, size=256)
        b = measure_image(im, curved, size=256)
        self.assertLess(abs(a["hem_curvature"]), 0.01)
        self.assertGreater(abs(b["hem_curvature"]), 0.05)

    def test_original_frame_metrics_detect_length_change(self):
        n = 256
        short = np.zeros((n, n), np.float32)
        long = np.zeros_like(short)
        short[30:170, 50:206] = 1
        long[30:220, 50:206] = 1
        im = Image.new("RGB", (n, n), (128, 128, 128))
        a = measure_image(im, short, size=256)
        b = measure_image(im, long, size=256)
        self.assertGreater(b["garment_height_image_ratio"], a["garment_height_image_ratio"])
        self.assertGreater(b["hem_y_image_ratio"], a["hem_y_image_ratio"])
        self.assertGreater(b["garment_aspect_ratio"], a["garment_aspect_ratio"])

    def test_manifest_with_no_existing_ground_truth_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest = {
                "samples": [
                    {
                        "sample_id": "missing",
                        "product_id": "p1",
                        "ground_truth": {
                            "image": str(root / "missing.png"),
                            "mask": str(root / "missing_mask.png"),
                        },
                    }
                ]
            }
            mp = root / "manifest.json"
            mp.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "zero evaluable records"):
                evaluate_manifest(mp, size=192)

    def test_hard_composite_preserves_every_outside_pixel(self):
        rng = np.random.default_rng(4)
        base = rng.integers(0, 256, (80, 90, 3), dtype=np.uint8)
        generated = 255 - base
        mask = np.zeros((80, 90), np.uint8)
        mask[20:60, 25:70] = 255
        out = np.asarray(hard_composite(Image.fromarray(base), Image.fromarray(generated), mask))
        self.assertTrue(np.array_equal(out[mask == 0], base[mask == 0]))
        self.assertFalse(np.array_equal(out[mask > 0], base[mask > 0]))

    def test_manifest_evaluation_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            n = 160
            image = np.full((n, n, 3), 128, np.uint8)
            mask = np.zeros((n, n), np.uint8)
            mask[20:140, 30:130] = 255
            Image.fromarray(image).save(root / "gt.png")
            Image.fromarray(image).save(root / "model.png")
            Image.fromarray(mask).save(root / "mask.png")
            manifest = {
                "samples": [
                    {
                        "sample_id": "p1_a",
                        "product_id": "p1",
                        "ground_truth": {
                            "image": str(root / "gt.png"),
                            "mask": str(root / "mask.png"),
                        },
                        "variants": {
                            "model": {
                                "image": str(root / "model.png"),
                                "mask": str(root / "mask.png"),
                                "change_mask": str(root / "mask.png"),
                            }
                        },
                    }
                ]
            }
            mp = root / "manifest.json"
            mp.write_text(json.dumps(manifest), encoding="utf-8")
            result = evaluate_manifest(mp, size=192)
            self.assertEqual(result["counts"]["samples"], 1)
            self.assertIn("model", result["comparisons"])
            self.assertEqual(
                result["comparisons"]["model"]["preservation"]["outside_changed_fraction"][
                    "median"
                ],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
