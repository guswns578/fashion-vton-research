"""Test whether a future 3D silhouette can select better FASHN candidates."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT = Path(r"fashion_pipeline")
BENCHMARK = PROJECT / "out" / "startup_benchmark_v1" / "manifest.json"
OUT = PROJECT / "out" / "seed_selector_gate"
WEIGHTS = Path(r"C:\Dev\fashn-vton-1.5\weights")
DEFAULT_SEEDS = (42, 137, 271, 911, 2026)


def mask_iou(left: str | Path, right: str | Path) -> float:
    with Image.open(left) as source:
        a = np.asarray(source.convert("L")) > 0
    with Image.open(right) as source:
        b_image = source.convert("L")
        if b_image.size != (a.shape[1], a.shape[0]):
            b_image = b_image.resize((a.shape[1], a.shape[0]), Image.Resampling.NEAREST)
        b = np.asarray(b_image) > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def command_run(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(PROJECT / "dataset"))
    sys.path.insert(0, r"C:\Dev\fashn-vton-1.5\src")
    from enhance import set_resolution, upper_roi
    from fashn_vton import TryOnPipeline
    from catalog_eval import _parser_and_ids, garment_mask, hard_composite, save_mask

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    sample = next(row for row in benchmark["samples"] if row["sku_id"] == args.sku)
    base = Image.open(sample["person"]).convert("RGB")
    garment = Image.open(sample["garment"]).convert("RGB")
    gt_mask = Path(sample["ground_truth"]["mask"])
    if not gt_mask.exists():
        raise FileNotFoundError(gt_mask)
    box = upper_roi(base, "tops")
    crop = base.crop(box)

    parser, ids = _parser_and_ids()
    old_mask = garment_mask(parser, ids, base)
    pipe = TryOnPipeline(weights_dir=str(WEIGHTS))
    height, width = (int(value) for value in args.res.lower().split("x"))
    set_resolution(pipe, height, width)

    args.out.mkdir(parents=True, exist_ok=True)
    variants = {}
    for index, seed in enumerate(args.seeds, 1):
        name = f"seed_{seed}"
        raw_path = args.out / "raw" / f"{args.sku}_{name}.png"
        image_path = args.out / "images" / f"{args.sku}_{name}.png"
        mask_path = args.out / "masks" / f"{args.sku}_{name}.png"
        if seed == 42 and Path(sample["variants"]["fashn"]["image"]).exists():
            image_path = Path(sample["variants"]["fashn"]["image"])
            mask_path = Path(sample["variants"]["fashn"]["mask"])
            raw_path = Path(sample["variants"]["fashn"]["raw"])
        elif not (args.resume and image_path.exists() and mask_path.exists()):
            started = time.time()
            result = pipe(
                person_image=crop,
                garment_image=garment,
                category="tops",
                garment_photo_type="flat-lay",
                num_samples=1,
                num_timesteps=args.steps,
                guidance_scale=1.5,
                seed=seed,
                segmentation_free=False,
            )
            generated_crop = result.images[0].resize(crop.size, Image.Resampling.LANCZOS)
            raw_full = base.copy()
            raw_full.paste(generated_crop, box[:2])
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_full.save(raw_path)
            new_mask = garment_mask(parser, ids, raw_full)
            union = np.maximum(old_mask, new_mask)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            hard_composite(base, raw_full, union).save(image_path)
            save_mask(new_mask, mask_path)
            print(f"[{index}/{len(args.seeds)}] {name} {time.time()-started:.0f}s", flush=True)
        variants[name] = {
            "seed": seed,
            "raw": str(raw_path.resolve()),
            "image": str(image_path.resolve()),
            "mask": str(mask_path.resolve()),
        }

    result_manifest = {
        "schema_version": 1,
        "dataset": "seed_selector_gate_v1",
        "protocol": "real_ground_truth_mask_as_temporary_3d_silhouette_oracle",
        "warning": (
            "Ground-truth masks are unavailable at deployment. This gate only estimates the upper bound "
            "of candidate selection before a 3D silhouette source exists."
        ),
        "sample": {
            "sku_id": args.sku,
            "person": sample["person"],
            "garment": sample["garment"],
            "ground_truth_mask": str(gt_mask.resolve()),
            "variants": variants,
        },
    }
    path = args.out / "manifest.json"
    path.write_text(json.dumps(result_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path)


def command_report(args: argparse.Namespace) -> None:
    manifest = json.loads((args.out / "manifest.json").read_text(encoding="utf-8"))
    sample = manifest["sample"]
    target_mask = args.target_mask or Path(sample["ground_truth_mask"])
    if not target_mask.exists():
        raise FileNotFoundError(target_mask)
    rows = []
    for name, variant in sample["variants"].items():
        rows.append({
            "variant": name,
            "seed": variant["seed"],
            "mask_iou": mask_iou(target_mask, variant["mask"]),
            "image": variant["image"],
        })
    rows.sort(key=lambda row: (-row["mask_iou"], row["seed"]))
    baseline = next(row for row in rows if row["seed"] == 42)
    best = rows[0]
    report = {
        "schema_version": 1,
        "sku_id": sample["sku_id"],
        "target_mask": str(target_mask.resolve()),
        "target_kind": "external_geometry" if args.target_mask else "ground_truth_oracle",
        "candidate_count": len(rows),
        "baseline_seed": 42,
        "baseline_mask_iou": baseline["mask_iou"],
        "best_seed": best["seed"],
        "best_mask_iou": best["mask_iou"],
        "oracle_improvement": best["mask_iou"] - baseline["mask_iou"],
        "interpretation": (
            "Positive improvement means target-silhouette candidate selection has headroom. "
            "Selection alone does not establish geometry simulation quality or material fidelity."
        ),
        "ranking": rows,
    }
    path = args.out / args.report_name
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "baseline_iou": report["baseline_mask_iou"],
        "best_iou": report["best_mask_iou"],
        "improvement": report["oracle_improvement"],
    }, ensure_ascii=False))
    print(path)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--benchmark", type=Path, default=BENCHMARK)
    run.add_argument("--sku", default="p2428")
    run.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    run.add_argument("--steps", type=int, default=30)
    run.add_argument("--res", default="1296x864")
    run.add_argument("--resume", action="store_true")
    run.set_defaults(func=command_run)
    report = sub.add_parser("report")
    report.add_argument(
        "--target-mask",
        type=Path,
        help="Optional 3D-rendered garment silhouette. Defaults to the held-out ground-truth mask.",
    )
    report.add_argument("--report-name", default="report.json")
    report.set_defaults(func=command_report)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
