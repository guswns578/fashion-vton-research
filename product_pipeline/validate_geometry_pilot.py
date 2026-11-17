"""Validate geometry-pilot inputs and report experiment readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jsonschema

DEFAULT_ROOT = Path(r"fashion_pipeline/out/geometry_pilot_v1")


def resolve_asset(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def complete_values(mapping: dict[str, Any]) -> tuple[bool, list[str]]:
    missing = [key for key, value in mapping.items() if value is None]
    return not missing, missing


def asset_status(root: Path, value: str | None) -> dict[str, Any]:
    path = resolve_asset(root, value)
    return {
        "configured": path is not None,
        "exists": bool(path and path.exists()),
        "path": str(path) if path else None,
    }


def build_readiness_report(root: Path, pilot: dict[str, Any]) -> dict[str, Any]:
    body_ok, missing_body = complete_values(pilot["subject"]["measurements_cm"])
    garment_ok, missing_garment = complete_values(pilot["garment"]["measurements_cm"])
    material = pilot["garment"]["material"]
    minimum_material_fields = ("composition", "bending_class", "surface_class")
    missing_material = [key for key in minimum_material_fields if material[key] is None]
    if material.get("mass_gsm") is None and material.get("garment_mass_g") is None:
        missing_material.append("mass_gsm_or_garment_mass_g")
    consent_ok = all(
        pilot["consent"][key]
        for key in ("subject_is_researcher", "body_scan", "image_generation", "commercial_research")
    )
    body_mesh = pilot["subject"]["body_mesh"]
    required_body_mesh_metadata = ("format", "scale_unit", "pose", "alignment")
    missing_body_mesh_metadata = [
        key for key in required_body_mesh_metadata if not body_mesh.get(key)
    ]
    if body_mesh.get("format") not in {"obj", "fbx", "glb", "ply"}:
        missing_body_mesh_metadata.append("format_supported_by_blender_import")
    capture = pilot["capture"]
    missing_camera = [
        key for key in ("camera_distance_cm", "focal_length_35mm_eq") if capture.get(key) is None
    ]

    assets = {
        "body_mesh": asset_status(root, body_mesh["path"]),
        "garment_flat_front": asset_status(root, pilot["garment"]["assets"]["flat_front"]),
        "garment_flat_back": asset_status(root, pilot["garment"]["assets"]["flat_back"]),
        "pattern_or_mesh": asset_status(root, pilot["garment"]["assets"]["pattern_or_mesh"]),
        "calibration_image": asset_status(root, capture["calibration_image"]),
        "reference_pose_image": asset_status(root, capture["reference_pose_image"]),
        "camera_config": asset_status(root, capture.get("camera_config")),
        "real_worn_mask": asset_status(root, pilot["evaluation"]["real_worn_mask"]),
        "real_worn_landmarks": asset_status(root, pilot["evaluation"]["real_worn_landmarks"]),
        "garment_silhouette": asset_status(root, pilot["geometry_outputs"]["garment_silhouette"]),
        "depth": asset_status(root, pilot["geometry_outputs"]["depth"]),
        "normal": asset_status(root, pilot["geometry_outputs"]["normal"]),
        "render_landmarks": asset_status(root, pilot["geometry_outputs"]["landmarks"]),
        "render_metadata": asset_status(root, pilot["geometry_outputs"]["render_metadata"]),
        "fashn_candidate_manifest": asset_status(
            root, pilot["evaluation"]["fashn_candidate_manifest"]
        ),
    }
    worn_images = [asset_status(root, value) for value in capture["worn_ground_truth_images"]]
    detail_images = [
        asset_status(root, value) for value in pilot["garment"]["assets"]["detail_closeups"]
    ]

    measurements_ready = body_ok and garment_ok and not missing_material
    pattern_ready = all(
        (
            consent_ok,
            measurements_ready,
            assets["garment_flat_front"]["exists"],
            assets["garment_flat_back"]["exists"],
        )
    )
    body_mesh_ready = all(
        (
            consent_ok,
            assets["body_mesh"]["exists"],
            not missing_body_mesh_metadata,
        )
    )
    simulation_ready = pattern_ready and body_mesh_ready
    capture_pair_ready = all(
        (
            consent_ok,
            assets["calibration_image"]["exists"],
            assets["reference_pose_image"]["exists"],
            bool(worn_images) and all(item["exists"] for item in worn_images),
            not missing_camera,
        )
    )
    render_outputs_ready = all(
        assets[name]["exists"]
        for name in ("garment_silhouette", "depth", "normal", "render_metadata")
    )
    render_ready = simulation_ready and capture_pair_ready
    silhouette_eval_ready = all(
        (
            render_ready,
            render_outputs_ready,
            assets["real_worn_mask"]["exists"],
        )
    )
    geometry_eval_ready = all(
        (
            silhouette_eval_ready,
            assets["real_worn_landmarks"]["exists"],
            assets["render_landmarks"]["exists"],
        )
    )
    selector_ready = geometry_eval_ready and assets["fashn_candidate_manifest"]["exists"]

    return {
        "schema_valid": True,
        "pilot_id": pilot["pilot_id"],
        "readiness": {
            "consent": consent_ok,
            "measurements": measurements_ready,
            "pattern_generation": pattern_ready,
            "body_mesh": body_mesh_ready,
            "simulation": simulation_ready,
            "capture_pair": capture_pair_ready,
            "render": render_ready,
            "render_outputs": render_outputs_ready,
            "silhouette_evaluation": silhouette_eval_ready,
            "geometry_evaluation": geometry_eval_ready,
            "selector_experiment": selector_ready,
            "capture_complete": simulation_ready and capture_pair_ready,
        },
        "missing": {
            "body_measurements": missing_body,
            "garment_measurements": missing_garment,
            "minimum_material": missing_material,
            "body_mesh_metadata": missing_body_mesh_metadata,
            "camera": missing_camera,
        },
        "assets": assets,
        "worn_ground_truth_images": worn_images,
        "detail_closeups": detail_images,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    pilot_path = args.pilot or root / "pilot.json"
    schema_path = args.schema or root / "geometry_pilot.schema.json"
    report_path = args.report or root / "reports" / "readiness.json"

    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(pilot, schema)

    report = build_readiness_report(root, pilot)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["readiness"], ensure_ascii=False))
    print(report_path)


if __name__ == "__main__":
    main()
