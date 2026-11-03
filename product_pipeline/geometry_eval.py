"""Evaluate an aligned 3D garment render against a real worn garment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


DEFAULT_ROOT = Path(r"fashion_pipeline/out/geometry_pilot_v1")


def resolve_asset(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("L")
        if size and image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        mask = np.asarray(image) >= 128
    if not mask.any():
        raise ValueError(f"Empty mask: {path}")
    return mask


def bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def silhouette_stats(mask: np.ndarray) -> dict[str, float]:
    x0, y0, x1, y1 = bbox(mask)
    height, width = mask.shape
    garment_h = y1 - y0
    garment_w = x1 - x0
    return {
        "height_image_ratio": garment_h / height,
        "width_image_ratio": garment_w / width,
        "aspect_ratio": garment_w / garment_h,
        "hem_y_image_ratio": (y1 - 1) / max(height - 1, 1),
    }


def mask_iou(real: np.ndarray, rendered: np.ndarray) -> float:
    union = np.logical_or(real, rendered).sum()
    return float(np.logical_and(real, rendered).sum() / union) if union else 1.0


def boundary(mask: np.ndarray) -> np.ndarray:
    binary = mask.astype(np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    return (binary - eroded).astype(bool)


def boundary_metrics(real: np.ndarray, rendered: np.ndarray, tolerance_px: int) -> dict[str, float]:
    real_edge = boundary(real)
    render_edge = boundary(rendered)
    real_distance = cv2.distanceTransform((~real_edge).astype(np.uint8), cv2.DIST_L2, 3)
    render_distance = cv2.distanceTransform((~render_edge).astype(np.uint8), cv2.DIST_L2, 3)
    precision = float((real_distance[render_edge] <= tolerance_px).mean())
    recall = float((render_distance[real_edge] <= tolerance_px).mean())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    symmetric_distance = float(
        (real_distance[render_edge].mean() + render_distance[real_edge].mean()) / 2
    )
    torso_height = max(bbox(real)[3] - bbox(real)[1], 1)
    return {
        "tolerance_px": tolerance_px,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "symmetric_distance_px": symmetric_distance,
        "symmetric_distance_torso_ratio": symmetric_distance / torso_height,
    }


def load_landmarks(path: Path) -> dict[str, tuple[float, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    points = raw.get("landmarks", raw)
    result = {}
    for name, value in points.items():
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Invalid landmark {name!r} in {path}")
        x, y = float(value[0]), float(value[1])
        if not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f"Landmark {name!r} must use normalized [0, 1] coordinates")
        result[name] = (x, y)
    return result


def landmark_metrics(
    real_path: Path | None,
    render_path: Path | None,
    image_size: tuple[int, int],
    torso_height: int,
) -> dict[str, Any]:
    if not real_path or not render_path:
        return {"available": False, "reason": "Both real and rendered landmark files are required."}
    real = load_landmarks(real_path)
    rendered = load_landmarks(render_path)
    common = sorted(set(real) & set(rendered))
    if not common:
        return {"available": False, "reason": "No common landmark names."}
    width, height = image_size
    rows = []
    for name in common:
        rx, ry = real[name]
        px, py = rendered[name]
        distance = float(np.hypot((rx - px) * width, (ry - py) * height))
        rows.append({
            "name": name,
            "distance_px": distance,
            "distance_torso_ratio": distance / max(torso_height, 1),
        })
    values = [row["distance_torso_ratio"] for row in rows]
    return {
        "available": True,
        "common_count": len(rows),
        "median_torso_ratio": float(np.median(values)),
        "p90_torso_ratio": float(np.percentile(values, 90)),
        "points": rows,
    }


def save_overlay(real: np.ndarray, rendered: np.ndarray, path: Path) -> None:
    canvas = np.full((*real.shape, 3), 32, dtype=np.uint8)
    overlap = np.logical_and(real, rendered)
    real_only = np.logical_and(real, ~rendered)
    render_only = np.logical_and(rendered, ~real)
    canvas[overlap] = (235, 235, 235)
    canvas[real_only] = (235, 70, 70)
    canvas[render_only] = (55, 205, 235)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(path)


def evaluate(
    real_mask_path: Path,
    render_mask_path: Path,
    real_landmarks_path: Path | None,
    render_landmarks_path: Path | None,
    overlay_path: Path,
    iou_threshold: float,
    landmark_threshold: float,
) -> dict[str, Any]:
    with Image.open(real_mask_path) as source:
        size = source.size
    real = load_mask(real_mask_path)
    rendered = load_mask(render_mask_path, size=size)
    real_stats = silhouette_stats(real)
    render_stats = silhouette_stats(rendered)
    stat_error = {
        key: abs(render_stats[key] - real_stats[key])
        for key in real_stats
    }
    real_box = bbox(real)
    torso_height = real_box[3] - real_box[1]
    tolerance = max(2, round(0.005 * np.hypot(*real.shape)))
    landmarks = landmark_metrics(
        real_landmarks_path,
        render_landmarks_path,
        size,
        torso_height,
    )
    iou = mask_iou(real, rendered)
    silhouette_pass = iou >= iou_threshold
    landmark_pass = bool(
        landmarks.get("available")
        and landmarks["common_count"] >= 4
        and landmarks["median_torso_ratio"] <= landmark_threshold
    )
    save_overlay(real, rendered, overlay_path)
    return {
        "inputs": {
            "real_mask": str(real_mask_path.resolve()),
            "render_mask": str(render_mask_path.resolve()),
            "real_landmarks": str(real_landmarks_path.resolve()) if real_landmarks_path else None,
            "render_landmarks": str(render_landmarks_path.resolve()) if render_landmarks_path else None,
        },
        "thresholds": {
            "mask_iou": iou_threshold,
            "median_landmark_torso_ratio": landmark_threshold,
            "minimum_common_landmarks": 4,
        },
        "mask_iou": iou,
        "boundary": boundary_metrics(real, rendered, tolerance),
        "real_silhouette": real_stats,
        "render_silhouette": render_stats,
        "absolute_silhouette_error": stat_error,
        "landmarks": landmarks,
        "decision": {
            "silhouette_pass": silhouette_pass,
            "landmark_pass": landmark_pass,
            "overall_pass": silhouette_pass and landmark_pass,
            "status": "pass" if silhouette_pass and landmark_pass else (
                "incomplete" if not landmarks.get("available") else "fail"
            ),
        },
        "overlay": str(overlay_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--real-mask", type=Path)
    parser.add_argument("--render-mask", type=Path)
    parser.add_argument("--real-landmarks", type=Path)
    parser.add_argument("--render-landmarks", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.80)
    parser.add_argument("--landmark-threshold", type=float, default=0.05)
    args = parser.parse_args()

    root = args.root.resolve()
    pilot_path = args.pilot or root / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    real_mask = args.real_mask or resolve_asset(root, pilot["evaluation"]["real_worn_mask"])
    render_mask = args.render_mask or resolve_asset(root, pilot["geometry_outputs"]["garment_silhouette"])
    real_landmarks = args.real_landmarks or resolve_asset(root, pilot["evaluation"]["real_worn_landmarks"])
    render_landmarks = args.render_landmarks or resolve_asset(root, pilot["geometry_outputs"]["landmarks"])
    if not real_mask or not render_mask:
        raise ValueError("Both real and rendered garment masks are required.")

    out_path = args.out or root / "reports" / "geometry_eval.json"
    overlay_path = args.overlay or root / "reports" / "geometry_overlay.png"
    report = evaluate(
        real_mask,
        render_mask,
        real_landmarks,
        render_landmarks,
        overlay_path,
        args.iou_threshold,
        args.landmark_threshold,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False))
    print(out_path)


if __name__ == "__main__":
    main()
