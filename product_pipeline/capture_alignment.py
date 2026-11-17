"""Check fixed-camera pose alignment for paired VTON pilot captures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

DEFAULT_ROOT = Path(r"fashion_pipeline/out/geometry_pilot_v1")
FASHN_SRC = Path(r"C:\Dev\fashn-vton-1.5\src")
DWPOSE_WEIGHTS = Path(r"C:\Dev\fashn-vton-1.5\weights\dwpose")
BODY_NAMES = (
    "nose",
    "neck",
    "shoulder_r",
    "elbow_r",
    "wrist_r",
    "shoulder_l",
    "elbow_l",
    "wrist_l",
    "hip_r",
    "knee_r",
    "ankle_r",
    "hip_l",
    "knee_l",
    "ankle_l",
    "eye_r",
    "eye_l",
    "ear_r",
    "ear_l",
)
CORE_NAMES = {
    "neck",
    "shoulder_r",
    "elbow_r",
    "wrist_r",
    "shoulder_l",
    "elbow_l",
    "wrist_l",
    "hip_r",
    "knee_r",
    "ankle_r",
    "hip_l",
    "knee_l",
    "ankle_l",
}


def resolve_asset(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def extract_body_points(pose: dict[str, Any]) -> dict[str, tuple[float, float]]:
    candidate = np.asarray(pose["bodies"]["candidate"])
    subset = np.asarray(pose["bodies"]["subset"])
    if candidate.shape[0] < len(BODY_NAMES) or subset.size < len(BODY_NAMES):
        return {}
    visibility = subset.reshape(-1)[: len(BODY_NAMES)] >= 0
    result = {}
    for index, name in enumerate(BODY_NAMES):
        x, y = candidate[index, :2]
        if visibility[index] and 0 <= x <= 1 and 0 <= y <= 1:
            result[name] = (float(x), float(y))
    return result


def compare_points(
    reference: dict[str, tuple[float, float]],
    worn: dict[str, tuple[float, float]],
    width: int,
    height: int,
    median_threshold: float = 0.02,
    p90_threshold: float = 0.04,
    minimum_points: int = 8,
) -> dict[str, Any]:
    common = sorted((set(reference) & set(worn)) & CORE_NAMES)
    rows = []
    for name in common:
        rx, ry = reference[name]
        wx, wy = worn[name]
        distance_px = float(np.hypot((rx - wx) * width, (ry - wy) * height))
        rows.append(
            {
                "name": name,
                "distance_px": distance_px,
                "distance_image_height_ratio": distance_px / height,
            }
        )
    ratios = [row["distance_image_height_ratio"] for row in rows]
    if not ratios:
        return {
            "status": "fail",
            "reason": "No common core body keypoints.",
            "common_count": 0,
            "points": [],
        }
    median = float(np.median(ratios))
    p90 = float(np.percentile(ratios, 90))
    passed = len(rows) >= minimum_points and median <= median_threshold and p90 <= p90_threshold
    return {
        "status": "pass" if passed else "fail",
        "common_count": len(rows),
        "median_image_height_ratio": median,
        "p90_image_height_ratio": p90,
        "maximum_image_height_ratio": max(ratios),
        "thresholds": {
            "minimum_common_points": minimum_points,
            "median_image_height_ratio": median_threshold,
            "p90_image_height_ratio": p90_threshold,
        },
        "points": sorted(rows, key=lambda row: row["distance_image_height_ratio"], reverse=True),
    }


def load_pose_detector(device: str):
    sys.path.insert(0, str(FASHN_SRC))
    from fashn_vton.dwpose import DWposeDetector

    return DWposeDetector(checkpoints_dir=str(DWPOSE_WEIGHTS), device=device)


def detect(detector, path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return bgr, detector(bgr)


def save_pose_overlay(
    reference_pose: dict[str, Any],
    worn_pose: dict[str, Any],
    height: int,
    width: int,
    path: Path,
) -> None:
    sys.path.insert(0, str(FASHN_SRC))
    from fashn_vton.dwpose import draw_pose

    ref = draw_pose(reference_pose, height, width, grayscale=True) > 0
    worn = draw_pose(worn_pose, height, width, grayscale=True) > 0
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    canvas[ref] = (235, 70, 70)
    canvas[worn] = (55, 205, 235)
    canvas[np.logical_and(ref, worn)] = (240, 240, 240)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--worn", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--median-threshold", type=float, default=0.02)
    parser.add_argument("--p90-threshold", type=float, default=0.04)
    args = parser.parse_args()

    root = args.root.resolve()
    pilot_path = args.pilot or root / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    reference = args.reference or resolve_asset(root, pilot["capture"]["reference_pose_image"])
    worn_values = pilot["capture"]["worn_ground_truth_images"]
    worn = args.worn or (resolve_asset(root, worn_values[0]) if worn_values else None)
    if not reference or not worn:
        raise ValueError("Both reference and worn capture images are required.")

    detector = load_pose_detector(args.device)
    ref_image, ref_pose = detect(detector, reference)
    worn_image, worn_pose = detect(detector, worn)
    height, width = ref_image.shape[:2]
    if worn_image.shape[:2] != (height, width):
        raise ValueError("Reference and worn captures must have identical image dimensions.")

    report = compare_points(
        extract_body_points(ref_pose),
        extract_body_points(worn_pose),
        width,
        height,
        median_threshold=args.median_threshold,
        p90_threshold=args.p90_threshold,
    )
    report["inputs"] = {
        "reference": str(reference.resolve()),
        "worn": str(worn.resolve()),
        "device": args.device,
    }
    out_path = args.out or root / "reports" / "capture_alignment.json"
    overlay_path = args.overlay or root / "reports" / "capture_pose_overlay.png"
    save_pose_overlay(ref_pose, worn_pose, height, width, overlay_path)
    report["overlay"] = str(overlay_path.resolve())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "common_count")}, ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
