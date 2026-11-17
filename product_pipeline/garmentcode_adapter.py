"""Generate GarmentCode sewing patterns from the geometry-pilot measurement contract."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

import yaml

from garmentcode_pattern_audit import audit, realized_measurements
from measurement_geometry import DEFAULT_BODY_CM, DEFAULT_GARMENT_CM, load_pilot_geometry

ROOT = Path(__file__).resolve().parents[1]
GARMENTCODE = ROOT / "third_party" / "GarmentCode"
PINNED_COMMIT = "d449629979028123a5c4dc9e732a2ec19b7fce31"


def body_overrides(
    body_cm: dict[str, float], neutral: dict[str, float]
) -> tuple[dict[str, float], dict]:
    shoulder_ratio = neutral["back_width"] / neutral["shoulder_w"]
    height_ratio = body_cm["height"] / neutral["height"]
    overrides = {
        "height": body_cm["height"],
        "bust": body_cm["chest"],
        "waist": body_cm["waist"],
        "hips": body_cm["hip"],
        "shoulder_w": body_cm["shoulder_breadth"],
        "arm_length": body_cm["arm_length"],
        "wrist": body_cm["wrist"],
        "neck_w": body_cm["neck"] / 2.0,
        "back_width": body_cm["shoulder_breadth"] * shoulder_ratio,
        "head_l": neutral["head_l"] * height_ratio,
        "waist_line": body_cm["back_waist_length"],
        "arm_pose_angle": 8.0,
    }
    contract = {
        "direct": [
            "height",
            "bust",
            "waist",
            "hips",
            "shoulder_w",
            "arm_length",
            "wrist",
            "waist_line",
        ],
        "derived": {
            "neck_w": "neck circumference / 2",
            "back_width": f"shoulder_breadth * neutral ratio {shoulder_ratio:.6f}",
            "head_l": f"neutral head length * height ratio {height_ratio:.6f}",
            "arm_pose_angle": "8 degrees to match the current Blender body-proxy capture pose",
        },
        "neutral_retained": sorted(set(neutral) - set(overrides)),
    }
    return overrides, contract


def design_overrides(
    design: dict,
    body_cm: dict[str, float],
    garment_cm: dict[str, float],
    body_params: dict[str, float],
) -> tuple[dict, dict]:
    result = deepcopy(design)
    shirt_width = 2.0 * garment_cm["chest_flat"] / body_cm["chest"]
    shirt_length = garment_cm["body_length"] / body_params["waist_line"]
    sleeve_length = garment_cm["sleeve_length"] / body_cm["arm_length"]
    result["meta"]["upper"]["v"] = "Shirt"
    result["meta"]["bottom"]["v"] = None
    result["meta"]["wb"]["v"] = None
    result["shirt"]["width"]["v"] = shirt_width
    result["shirt"]["length"]["v"] = shirt_length
    result["shirt"]["flare"]["v"] = 2.0 * garment_cm["hem_flat"] / body_cm["chest"]
    result["sleeve"]["sleeveless"]["v"] = False
    result["sleeve"]["armhole_shape"]["v"] = "ArmholeCurve"
    result["sleeve"]["length"]["v"] = sleeve_length
    result["sleeve"]["end_width"]["v"] = max(
        0.2,
        2.0 * garment_cm["cuff_flat"] / max(garment_cm["upper_arm_flat"] * 2.0, 1e-6),
    )
    contract = {
        "direct_targets_cm": garment_cm,
        "mapped": {
            "shirt.width": shirt_width,
            "shirt.length": shirt_length,
            "shirt.flare": result["shirt"]["flare"]["v"],
            "sleeve.length": sleeve_length,
            "sleeve.end_width": result["sleeve"]["end_width"]["v"],
        },
        "warning": (
            "GarmentCode parameters are an initialization. Validate realized panel seam lengths and "
            "fit against flat captures before simulation."
        ),
    }
    return result, contract


def update_calibration(
    design: dict,
    body_params: dict[str, float],
    targets: dict[str, float],
    realized: dict[str, float],
) -> dict:
    """Apply one transparent proportional/Newton-like update to design parameters."""
    old_width = float(design["shirt"]["width"]["v"])
    width_factor = targets["chest_flat"] / realized["chest_flat"]
    design["shirt"]["width"]["v"] = float(old_width * width_factor)
    design["shirt"]["flare"]["v"] = float(
        design["shirt"]["flare"]["v"] * (targets["hem_flat"] / realized["hem_flat"] / width_factor)
    )
    design["shirt"]["length"]["v"] = float(
        design["shirt"]["length"]["v"]
        + (targets["body_length"] - realized["body_length"]) / body_params["waist_line"]
    )
    design["sleeve"]["length"]["v"] = float(
        design["sleeve"]["length"]["v"]
        + (targets["sleeve_length"] - realized["sleeve_length"]) / body_params["arm_length"]
    )
    design["sleeve"]["end_width"]["v"] = float(
        design["sleeve"]["end_width"]["v"] * targets["cuff_flat"] / realized["cuff_flat"]
    )

    shoulder_delta = targets["shoulder"] - realized["shoulder"]
    current_extension = (
        float(design["sleeve"]["standing_shoulder_len"]["v"])
        if design["sleeve"]["standing_shoulder"]["v"]
        else 0.0
    )
    extension = float(max(0.0, current_extension + shoulder_delta / 2.0))
    design["sleeve"]["standing_shoulder"]["v"] = extension > 0.05
    design["sleeve"]["standing_shoulder_len"]["v"] = extension
    if extension > 0.05:
        design["sleeve"]["sleeve_angle"]["v"] = max(
            30.0, float(body_params.get("_shoulder_incl", 10.0)) + 10.0
        )
    return design


def calibrate_design(
    MetaGarment, body, design: dict, targets: dict[str, float], iterations: int = 6
):
    history = []
    final_garment = None
    final_pattern = None
    for iteration in range(iterations + 1):
        final_garment = MetaGarment("metric_long_sleeve", body, design)
        final_pattern = final_garment.assembly()
        measured = realized_measurements(final_pattern.pattern)
        errors = {name: measured[name] - targets[name] for name in measured if name in targets}
        history.append({"iteration": iteration, "realized_cm": measured, "error_cm": errors})
        if iteration == iterations or max(abs(value) for value in errors.values()) <= 0.1:
            break
        update_calibration(design, body.params, targets, measured)
    return final_garment, final_pattern, design, history


def generate(pilot_path: Path | None, out: Path) -> Path:
    if not GARMENTCODE.exists():
        raise FileNotFoundError(GARMENTCODE)
    if str(GARMENTCODE) not in sys.path:
        sys.path.insert(0, str(GARMENTCODE))
    from assets.bodies.body_params import BodyParameters
    from assets.garment_programs.meta_garment import MetaGarment

    body_geometry, _, source = load_pilot_geometry(pilot_path)
    if pilot_path:
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
        garment_cm = {
            key: float(value) for key, value in pilot["garment"]["measurements_cm"].items()
        }
    else:
        garment_cm = dict(DEFAULT_GARMENT_CM)
    body_cm = body_geometry.source_cm

    neutral_path = GARMENTCODE / "assets" / "bodies" / "mean_all.yaml"
    design_path = GARMENTCODE / "assets" / "design_params" / "t-shirt.yaml"
    neutral_raw = yaml.safe_load(neutral_path.read_text(encoding="utf-8"))["body"]
    design_raw = yaml.safe_load(design_path.read_text(encoding="utf-8"))["design"]
    body = BodyParameters(neutral_path)
    overrides, body_contract = body_overrides(body_cm, neutral_raw)
    body.load_from_dict(overrides)
    design, design_contract = design_overrides(design_raw, body_cm, garment_cm, body.params)

    garment, pattern, design, calibration_history = calibrate_design(
        MetaGarment, body, design, garment_cm
    )
    if garment.is_self_intersecting():
        raise ValueError("Generated GarmentCode pattern is self-intersecting.")
    out.mkdir(parents=True, exist_ok=True)
    folder = pattern.serialize(
        out,
        tag="_pattern",
        to_subfolder=True,
        with_3d=False,
        with_text=False,
        view_ids=False,
        with_printable=True,
    )
    body.save(folder)
    generated_design = Path(folder) / "design.yaml"
    plain_design = json.loads(json.dumps(design, default=float))
    generated_design.write_text(
        yaml.safe_dump({"design": plain_design}, sort_keys=False), encoding="utf-8"
    )
    contract = {
        "schema_version": 1,
        "source": source,
        "garmentcode": {
            "repository": "https://github.com/maria-korosteleva/GarmentCode",
            "commit": PINNED_COMMIT,
            "license": "MIT",
            "simulation_code_used": False,
        },
        "body_mapping": body_contract,
        "design_mapping": design_contract,
        "calibration": {
            "method": "iterative realized-pattern measurement correction",
            "history": calibration_history,
        },
        "output_folder": str(Path(folder).resolve()),
    }
    (Path(folder) / "adapter_contract.json").write_text(
        json.dumps(contract, indent=2, default=float), encoding="utf-8"
    )
    specification_files = list(Path(folder).glob("*_specification.json"))
    if len(specification_files) != 1:
        raise RuntimeError(f"Expected one pattern specification, got {specification_files}")
    report = audit(specification_files[0], garment_cm)
    (Path(folder) / "pattern_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.copy2(GARMENTCODE / "LICENSE", Path(folder) / "GARMENTCODE_LICENSE.txt")
    return Path(folder)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pilot = args.pilot.resolve() if args.pilot else None
    print(generate(pilot, args.out.resolve()))


if __name__ == "__main__":
    main()
