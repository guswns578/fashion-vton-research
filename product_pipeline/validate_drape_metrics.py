"""Validate drape metrics against controlled perturbations of real garments.

The report answers two questions before a metric is used as a gate:

1. Does a known length/width/blur perturbation move the intended metric in the
   expected direction?
2. Is that movement larger than the previously measured codec/resampling/mask
   noise floor?
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from drape_metrics import _as_mask, _as_rgb, _bbox, measure_image, measurement_noise


def _scale_mask(mask: np.ndarray, axis: str, factor: float) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        raise ValueError("empty garment mask")
    x0, y0, x1, y1 = box
    crop = Image.fromarray(((mask[y0:y1, x0:x1] > 0.5) * 255).astype(np.uint8), "L")
    width, height = crop.size
    if axis == "height":
        target = (width, max(2, round(height * factor)))
        anchor_x, anchor_y = x0, y0
    elif axis == "width":
        target = (max(2, round(width * factor)), height)
        anchor_x = round((x0 + x1 - target[0]) / 2)
        anchor_y = y0
    else:
        raise ValueError(f"unknown axis: {axis}")
    resized = np.asarray(crop.resize(target, Image.Resampling.NEAREST), np.uint8)
    out = np.zeros_like(mask, np.float32)
    dst_x0, dst_y0 = max(0, anchor_x), max(0, anchor_y)
    dst_x1 = min(mask.shape[1], anchor_x + target[0])
    dst_y1 = min(mask.shape[0], anchor_y + target[1])
    src_x0, src_y0 = dst_x0 - anchor_x, dst_y0 - anchor_y
    src_x1, src_y1 = src_x0 + dst_x1 - dst_x0, src_y0 + dst_y1 - dst_y0
    if dst_x1 > dst_x0 and dst_y1 > dst_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = resized[src_y0:src_y1, src_x0:src_x1] / 255.0
    return out


def _median(values: list[float]) -> float | None:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else None


def _monotonic_rate(rows: list[list[float]], increasing: bool) -> float | None:
    valid = [r for r in rows if len(r) >= 2 and np.all(np.isfinite(r))]
    if not valid:
        return None
    passed = []
    for row in valid:
        diff = np.diff(row)
        passed.append(bool(np.all(diff > 0) if increasing else np.all(diff < 0)))
    return float(np.mean(passed))


def validate(manifest_path: Path, size: int = 384, limit: int = 0) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", manifest if isinstance(manifest, list) else [])
    length_factors = [0.90, 1.00, 1.10]
    width_factors = [0.90, 1.00, 1.10]
    blur_radii = [0.0, 1.5, 3.0]
    length_rows: dict[str, list[list[float]]] = defaultdict(list)
    width_rows: dict[str, list[list[float]]] = defaultdict(list)
    blur_rows: dict[str, list[list[float]]] = defaultdict(list)
    noise: dict[str, list[float]] = defaultdict(list)
    processed: list[str] = []
    skipped: list[dict[str, str]] = []

    for sample in samples:
        if limit and len(processed) >= limit:
            break
        gt = sample.get("ground_truth") or {}
        image_path, mask_path = Path(gt.get("image", "")), Path(gt.get("mask", ""))
        if not image_path.is_file() or not mask_path.is_file():
            skipped.append({"sample_id": sample.get("sample_id", ""), "reason": "missing_gt"})
            continue
        image = _as_rgb(image_path)
        mask = _as_mask(mask_path, image.size)
        base = measure_image(image, mask, size=size)
        for repeated in measurement_noise(image, mask, size=size):
            for metric, value in base.items():
                other = repeated.get(metric)
                if isinstance(value, (int, float)) and isinstance(other, (int, float)):
                    if np.isfinite(value) and np.isfinite(other):
                        noise[metric].append(abs(float(other) - float(value)))

        length_metrics = [
            measure_image(image, _scale_mask(mask, "height", factor), size=size)
            for factor in length_factors
        ]
        for metric in ("garment_height_image_ratio", "hem_y_image_ratio", "garment_aspect_ratio"):
            length_rows[metric].append([float(row[metric]) for row in length_metrics])

        width_metrics = [
            measure_image(image, _scale_mask(mask, "width", factor), size=size)
            for factor in width_factors
        ]
        for metric in ("garment_width_image_ratio",):
            width_rows[metric].append([float(row[metric]) for row in width_metrics])

        blur_metrics = []
        for radius in blur_radii:
            variant = image if radius == 0 else image.filter(ImageFilter.GaussianBlur(radius))
            blur_metrics.append(measure_image(variant, mask, size=size))
        for metric in ("fold_contrast", "wrinkle_strength_p90", "meso_energy_ratio", "micro_energy_ratio"):
            blur_rows[metric].append([float(row[metric]) for row in blur_metrics])
        processed.append(sample.get("sample_id", ""))

    if not processed:
        raise ValueError(f"zero valid ground-truth samples in {manifest_path}")

    def section(rows: dict[str, list[list[float]]], increasing: bool) -> dict[str, Any]:
        result = {}
        for metric, values in rows.items():
            endpoint = [abs(row[-1] - row[0]) for row in values]
            floor = _median(noise.get(metric, []))
            signal = _median(endpoint)
            monotonic = _monotonic_rate(values, increasing)
            snr = None if floor in (None, 0) or signal is None else signal / floor
            above_noise = bool(
                signal is not None and (
                    (floor == 0 and signal > 0)
                    or (snr is not None and snr >= 3.0)
                )
            )
            result[metric] = {
                "n": len(values),
                "strict_monotonic_rate": monotonic,
                "median_endpoint_signal": signal,
                "median_noise_floor": floor,
                "signal_to_noise": snr,
                "above_noise_floor": above_noise,
                "passes_default_gate": bool(
                    monotonic is not None and monotonic >= 0.80 and above_noise
                ),
                "median_curve": [float(np.median([row[i] for row in values])) for i in range(len(values[0]))],
            }
        return result

    report = {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "normalised_size": size,
        "counts": {"processed": len(processed), "skipped": len(skipped)},
        "factors": {"length": length_factors, "width": width_factors, "blur_radius": blur_radii},
        "length_validation": section(length_rows, increasing=True),
        "width_validation": section(width_rows, increasing=True),
        "blur_validation": section(blur_rows, increasing=False),
        "skipped": skipped,
    }
    verdicts = {}
    for section_name in ("length_validation", "width_validation", "blur_validation"):
        for metric, result in report[section_name].items():
            verdicts[metric] = result["passes_default_gate"]
    report["verdict"] = {
        "criteria": {"strict_monotonic_rate_min": 0.80, "signal_to_noise_min": 3.0},
        "validated_metrics": sorted(metric for metric, passed in verdicts.items() if passed),
        "failed_metrics": sorted(metric for metric, passed in verdicts.items() if not passed),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    report = validate(args.manifest, size=args.size, limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False))
    print(args.out)


if __name__ == "__main__":
    main()
