"""Reconstruct and evaluate the interrupted framing/resolution gate."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(r"fashion_pipeline/out/res_gate")
MANIFEST = ROOT / "manifest.json"
DRAPE = ROOT / "drape_metrics.json"
METRIC_VALIDATION = Path(r"fashion_pipeline/out/catalog_eval/metric_validation.json")
NAME_RE = re.compile(r"^(p\d+)_(m\d+_g\d+)_(gt|vton)\.png$")


def build_manifest(root: Path) -> dict:
    grouped: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)
    ignored = []
    for path in sorted(root.glob("*.png")):
        match = NAME_RE.match(path.name)
        if not match:
            ignored.append(str(path.resolve()))
            continue
        product_id, condition, role = match.groups()
        grouped[(product_id, condition)][role] = path.resolve()

    samples = []
    incomplete = []
    for (product_id, condition), paths in sorted(grouped.items()):
        if not {"gt", "vton"}.issubset(paths):
            incomplete.append({
                "product_id": product_id,
                "condition": condition,
                "present": sorted(paths),
            })
            continue
        sample_id = f"{product_id}_{condition}"
        samples.append({
            "sample_id": sample_id,
            "product_id": product_id,
            "condition": condition,
            "ground_truth": {
                "image": str(paths["gt"]),
                "mask": str((root / "masks" / "ground_truth" / f"{sample_id}.png").resolve()),
            },
            "variants": {
                "vton": {
                    "image": str(paths["vton"]),
                    "mask": str((root / "masks" / "vton" / f"{sample_id}.png").resolve()),
                }
            },
        })
    return {
        "schema_version": 1,
        "dataset": "resolution_gate_reconstruction",
        "protocol": "aligned_condition_specific_reconstruction",
        "counts": {
            "complete_pairs": len(samples),
            "products": len({sample["product_id"] for sample in samples}),
            "incomplete_conditions": len(incomplete),
        },
        "incomplete": incomplete,
        "ignored": ignored,
        "samples": samples,
    }


def command_build(args: argparse.Namespace) -> None:
    manifest = build_manifest(args.root)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(args.manifest)


def command_masks(args: argparse.Namespace) -> None:
    import fashn_human_parser as fhp

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    parser = fhp.FashnHumanParser()
    ids = [fhp.LABELS_TO_IDS[name] for name in fhp.BODY_COVERAGE_TO_LABELS["upper"]]
    made = 0
    for sample in data["samples"]:
        for role, spec in (
            ("ground_truth", sample["ground_truth"]),
            ("vton", sample["variants"]["vton"]),
        ):
            dst = Path(spec["mask"])
            if args.resume and dst.exists():
                continue
            with Image.open(spec["image"]) as source:
                image = source.convert("RGB")
            segmentation = parser.predict(image)
            mask = np.isin(segmentation, ids).astype(np.uint8) * 255
            if mask.shape != (image.height, image.width):
                mask = np.asarray(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST))
            dst.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, "L").save(dst)
            made += 1
            print(f"{sample['sample_id']} {role}", flush=True)
    print(f"made={made}")


def mask_iou(left: str, right: str) -> float:
    with Image.open(left) as source:
        a = np.asarray(source.convert("L")) > 0
    with Image.open(right) as source:
        b_image = source.convert("L")
        if b_image.size != (a.shape[1], a.shape[0]):
            b_image = b_image.resize((a.shape[1], a.shape[0]), Image.Resampling.NEAREST)
        b = np.asarray(b_image) > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def validation_scales(validation: dict) -> dict[str, float]:
    scales = {}
    for section in ("length_validation", "width_validation", "blur_validation"):
        for metric, row in validation.get(section, {}).items():
            signal = row.get("median_endpoint_signal")
            if signal is not None and float(signal) > 0:
                scales[metric] = float(signal)
    return scales


def median_or_none(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def command_report(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    drape = json.loads(args.drape.read_text(encoding="utf-8"))
    validation = json.loads(args.metric_validation.read_text(encoding="utf-8"))
    trusted = validation["verdict"]["validated_metrics"]
    scales = validation_scales(validation)
    records = {(row["sample_id"], row["variant"]): row for row in drape["records"]}

    rows = []
    for sample in manifest["samples"]:
        sample_id = sample["sample_id"]
        gt = records[(sample_id, "ground_truth")]["metrics"]
        output = records[(sample_id, "vton")]["metrics"]
        deltas = {
            metric: float(output[metric] - gt[metric])
            for metric in trusted
            if gt.get(metric) is not None and output.get(metric) is not None
        }
        normalised = [abs(value) / scales[key] for key, value in deltas.items() if key in scales]
        detail_ratio = {}
        for metric in ("fold_contrast", "micro_energy_ratio", "wrinkle_strength_p90"):
            if gt.get(metric) not in (None, 0) and output.get(metric) is not None:
                detail_ratio[metric] = float(output[metric] / gt[metric])
        rows.append({
            "sample_id": sample_id,
            "product_id": sample["product_id"],
            "condition": sample["condition"],
            "mask_iou": mask_iou(sample["ground_truth"]["mask"], sample["variants"]["vton"]["mask"]),
            "mean_normalised_error": float(np.mean(normalised)) if normalised else None,
            "trusted_metric_delta": deltas,
            "detail_ratio": detail_ratio,
        })

    by_condition: dict[str, list[dict]] = defaultdict(list)
    by_product: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]].append(row)
        by_product[row["product_id"]][row["condition"]] = row
    conditions = {}
    for condition, values in sorted(by_condition.items()):
        conditions[condition] = {
            "n_products": len(values),
            "mask_iou_median": median_or_none([row["mask_iou"] for row in values]),
            "mean_normalised_error_median": median_or_none([
                row["mean_normalised_error"] for row in values if row["mean_normalised_error"] is not None
            ]),
            "detail_ratio_median": {
                metric: median_or_none([
                    row["detail_ratio"][metric] for row in values if metric in row["detail_ratio"]
                ])
                for metric in ("fold_contrast", "micro_energy_ratio", "wrinkle_strength_p90")
            },
        }

    comparisons = {}
    for candidate, reference in (("m15_g864", "m00_g864"), ("m30_g864", "m00_g864"),
                                 ("m00_g1008", "m00_g864")):
        paired = [values for values in by_product.values() if candidate in values and reference in values]
        comparisons[f"{candidate}_minus_{reference}"] = {
            "n_products": len(paired),
            "mask_iou_delta_median": median_or_none([
                values[candidate]["mask_iou"] - values[reference]["mask_iou"] for values in paired
            ]),
            "normalised_error_delta_median": median_or_none([
                values[candidate]["mean_normalised_error"] - values[reference]["mean_normalised_error"]
                for values in paired
            ]),
        }

    report = {
        "schema_version": 1,
        "counts": manifest["counts"],
        "caveat": (
            "The original runner script was missing. Conditions are reconstructed from filenames; "
            "results establish observed trade-offs but do not prove the intended generation parameters."
        ),
        "trusted_metrics": trusted,
        "conditions": conditions,
        "paired_comparisons": comparisons,
        "per_sample": rows,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": len(rows), "conditions": len(conditions)}, ensure_ascii=False))
    print(args.out)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    sub = ap.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.set_defaults(func=command_build)
    masks = sub.add_parser("masks")
    masks.add_argument("--resume", action="store_true")
    masks.set_defaults(func=command_masks)
    report = sub.add_parser("report")
    report.add_argument("--drape", type=Path, default=DRAPE)
    report.add_argument("--metric-validation", type=Path, default=METRIC_VALIDATION)
    report.add_argument("--out", type=Path, default=ROOT / "report.json")
    report.set_defaults(func=command_report)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
