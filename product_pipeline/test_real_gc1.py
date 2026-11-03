from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from run_real_gc1 import same_resolved_asset, simulation_command, stage_blockers
from validate_geometry_pilot import build_readiness_report


HERE = Path(__file__).resolve().parent


class RealGc1Tests(unittest.TestCase):
    def complete_pilot(self, root: Path) -> dict:
        pilot = json.loads((HERE / "geometry_pilot.example.json").read_text(encoding="utf-8"))
        pilot["subject"]["measurements_cm"] = {
            "height": 172,
            "neck": 37,
            "shoulder_breadth": 42,
            "chest": 92,
            "waist": 76,
            "hip": 94,
            "arm_length": 60,
            "upper_arm": 29,
            "wrist": 16,
            "back_waist_length": 44,
        }
        pilot["garment"]["measurements_cm"] = {
            "shoulder": 48,
            "chest_flat": 55,
            "body_length": 70,
            "sleeve_length": 60,
            "upper_arm_flat": 22,
            "cuff_flat": 12,
            "hem_flat": 55,
            "armhole": 25,
        }
        pilot["garment"]["material"].update({
            "composition": "cotton 100%",
            "mass_gsm": 210,
            "bending_class": "medium",
            "surface_class": "matte",
        })
        pilot["subject"]["body_mesh"].update({
            "path": "body.obj",
            "format": "obj",
            "scale_unit": "m",
            "pose": "a_pose",
            "alignment": "prealigned",
        })
        pilot["garment"]["assets"].update({
            "flat_front": "flat_front.png",
            "flat_back": "flat_back.png",
        })
        pilot["capture"].update({
            "calibration_image": "calibration.png",
            "reference_pose_image": "reference.png",
            "worn_ground_truth_images": ["worn.png"],
            "camera_distance_cm": 250,
            "focal_length_35mm_eq": 26,
        })
        for name in (
            "body.obj",
            "flat_front.png",
            "flat_back.png",
            "calibration.png",
            "reference.png",
            "worn.png",
        ):
            (root / name).touch()
        return pilot

    def test_readiness_separates_simulation_capture_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = self.complete_pilot(root)
            report = build_readiness_report(root, pilot)
            self.assertTrue(report["readiness"]["pattern_generation"])
            self.assertTrue(report["readiness"]["simulation"])
            self.assertTrue(report["readiness"]["capture_pair"])
            self.assertTrue(report["readiness"]["render"])
            self.assertFalse(report["readiness"]["silhouette_evaluation"])
            self.assertEqual(stage_blockers("pattern", report), [])

    def test_material_mass_is_required_for_real_simulation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = self.complete_pilot(root)
            pilot = deepcopy(pilot)
            pilot["garment"]["material"]["mass_gsm"] = None
            pilot["garment"]["material"]["garment_mass_g"] = None
            report = build_readiness_report(root, pilot)
            self.assertFalse(report["readiness"]["measurements"])
            self.assertIn("mass_gsm_or_garment_mass_g", report["missing"]["minimum_material"])

    def test_unconverted_smpl_body_is_not_simulation_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = self.complete_pilot(root)
            pilot["subject"]["body_mesh"]["format"] = "smpl"
            report = build_readiness_report(root, pilot)
            self.assertFalse(report["readiness"]["body_mesh"])
            self.assertIn(
                "format_supported_by_blender_import",
                report["missing"]["body_mesh_metadata"],
            )

    def test_simulation_command_keeps_external_body_safety_contract(self):
        command = simulation_command(
            Path("blender.exe"),
            Path("pilot.json"),
            Path("contract.json"),
            Path("body.obj"),
            "m",
            "prealigned",
            Path("out"),
            True,
            "medium",
        )
        self.assertIn("--resolve-external-penetration", command)
        self.assertEqual(command[command.index("--penetration-clearance") + 1], "0.007")
        self.assertEqual(command[command.index("--body-collision-thickness") + 1], "0.003")
        self.assertEqual(command[command.index("--gravity-scale") + 1], "1")
        self.assertEqual(command[command.index("--min-frames") + 1], "140")

    def test_self_evaluation_path_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertTrue(same_resolved_asset(root, "mask.png", "mask.png"))
            self.assertFalse(same_resolved_asset(root, "real.png", "render.png"))


if __name__ == "__main__":
    unittest.main()
