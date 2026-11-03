"""Build a perspective Blender camera contract from fixed smartphone capture metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from measurement_geometry import load_pilot_geometry


FOCAL_35MM_EXIF_TAG = 41989


def focal_from_exif(path: Path) -> float | None:
    with Image.open(path) as source:
        value = source.getexif().get(FOCAL_35MM_EXIF_TAG)
    return float(value) if value else None


def build_camera(
    height_cm: float,
    pattern_shoulder_z_m: float,
    camera_distance_cm: float,
    focal_length_35mm_eq: float,
    image_size: tuple[int, int],
) -> dict:
    if min(height_cm, camera_distance_cm, focal_length_35mm_eq, *image_size) <= 0:
        raise ValueError("Height, distance, focal length, and image dimensions must be positive.")
    height_m = height_cm / 100.0
    feet_z = pattern_shoulder_z_m - 0.82 * height_m
    body_center_z = feet_z + 0.5 * height_m
    distance_m = camera_distance_cm / 100.0
    return {
        "schema_version": 1,
        "projection": "perspective",
        "location": [0.0, -distance_m, body_center_z],
        "target": [0.0, 0.0, body_center_z],
        "focal_length_mm": focal_length_35mm_eq,
        "sensor_width_mm": 36.0,
        "sensor_fit": "AUTO",
        "shift_x": 0.0,
        "shift_y": 0.0,
        "clip_start": 0.01,
        "clip_end": max(20.0, distance_m * 4.0),
        "reference_resolution": list(image_size),
        "assumptions": {
            "subject_centered": True,
            "camera_level": True,
            "shoulder_height_ratio": 0.82,
            "focal_interpretation": "35mm-equivalent lens on a 36mm virtual sensor",
        },
        "warning": (
            "This is an initialization. Validate against body landmarks and optimize camera shift, "
            "target, and distance before using silhouette IoU."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pilot_path = args.pilot.resolve()
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    root = pilot_path.parent
    reference_value = args.reference or pilot["capture"].get("reference_pose_image")
    if not reference_value:
        raise ValueError("A fixed-camera reference image is required.")
    reference = Path(reference_value)
    if not reference.is_absolute():
        reference = root / reference
    if not reference.exists():
        raise FileNotFoundError(reference)
    with Image.open(reference) as source:
        image_size = source.size
    capture = pilot["capture"]
    distance = capture.get("camera_distance_cm")
    focal = capture.get("focal_length_35mm_eq") or focal_from_exif(reference)
    if distance is None or focal is None:
        raise ValueError("camera_distance_cm and focal_length_35mm_eq (or matching EXIF) are required.")
    body, pattern, _ = load_pilot_geometry(pilot_path)
    camera = build_camera(
        body.source_cm["height"],
        pattern.shoulder_z_m,
        float(distance),
        float(focal),
        image_size,
    )
    camera["reference_image"] = str(reference.resolve())
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(camera, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
