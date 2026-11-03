"""Build and run a product-grouped shop VTO evaluation set.

The archive contains real flat garment photos and real worn photos of the same
SKU.  It does *not* contain the counterfactual, pixel-aligned photograph needed
for a perfect cross-garment VTO ground truth.  The generated manifest therefore
labels this protocol honestly as ``self_reconstruction_proxy`` and keeps every
pose of one product in the same statistical group.

Commands
--------
Build a deterministic 4-pose-per-product manifest (NumPy + Pillow only)::

    python catalog_eval.py build

Create ground-truth garment masks (run in the FASHN environment)::

    python catalog_eval.py masks --target ground_truth

Generate the current local FASHN baseline and hard-composite it back into the
original photograph (run in the FASHN environment)::

    python catalog_eval.py fashn --limit 2

After FLUX VTO v2 raw outputs have been downloaded, parse their garment masks
and make face/background pixels exact::

    python catalog_eval.py finalise --variant flux_vto_v2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter


PROJECT = Path(r"fashion_pipeline")
RAW = PROJECT / "dataset" / "_raw"
DEFAULT_OUT = PROJECT / "out" / "catalog_eval"
WEIGHTS = Path(r"C:\Dev\fashn-vton-1.5\weights")

BOTTOM_KW = ("denim", " pt", "pants", "shorts", "trouser", "cargo", "half pt", "wide pt")
TOP_KW = ("tee", "shirt", "knit", "sleeveless", "sweat", "long sleeve", "longsleeve", "hoodie")


def is_top(name: str) -> bool:
    s = name.lower()
    return not any(k in s for k in BOTTOM_KW) and any(k in s for k in TOP_KW)


def resolve_image(product_dir: Path, name: str) -> Path | None:
    for p in (product_dir / name, product_dir / "detail" / name):
        if p.exists():
            return p.resolve()
    return None


def _thumb(path: Path, size: int = 32) -> np.ndarray:
    with Image.open(path) as im:
        # JPEG decoder draft mode avoids fully decoding every 4K archive image.
        im.draft("RGB", (size, size))
        a = np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), np.float32)
    # Normalise brightness so farthest-point selection favours composition and
    # pose more than a simple exposure change.
    a = a / 255.0
    return (a - a.mean((0, 1), keepdims=True)) / (a.std((0, 1), keepdims=True) + 0.05)


def choose_diverse_worn(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    # Diversity over every archive frame is expensive and mostly compares near
    # duplicates.  First retain a generous quality shortlist.
    candidates = sorted(candidates, key=lambda x: (-x["quality"], x["name"]))[: max(24, count * 8)]
    if len(candidates) <= count:
        return sorted(candidates, key=lambda x: x["name"])
    thumbs = [_thumb(Path(c["path"])) for c in candidates]
    quality = np.asarray([c["quality"] for c in candidates], float)
    quality = (quality - quality.min()) / max(float(np.ptp(quality)), 1e-8)
    chosen = [int(np.argmax(quality))]
    while len(chosen) < count:
        dist = np.full(len(candidates), np.inf)
        for j in chosen:
            d = np.mean(np.abs(np.stack(thumbs) - thumbs[j]), axis=(1, 2, 3))
            dist = np.minimum(dist, d)
        score = dist + quality * 0.15
        score[chosen] = -np.inf
        chosen.append(int(np.argmax(score)))
    return [candidates[i] for i in sorted(chosen, key=lambda j: candidates[j]["name"])]


def nms(boxes: list[tuple[float, ...]], iou_threshold: float = 0.5) -> list[tuple[float, ...]]:
    keep = []
    for box in sorted(boxes, key=lambda x: -x[4]):
        accepted = True
        for old in keep:
            iw = max(0.0, min(box[2], old[2]) - max(box[0], old[0]))
            ih = max(0.0, min(box[3], old[3]) - max(box[1], old[1]))
            inter = iw * ih
            union = ((box[2] - box[0]) * (box[3] - box[1]) +
                     (old[2] - old[0]) * (old[3] - old[1]) - inter)
            if union > 0 and inter / union > iou_threshold:
                accepted = False
                break
        if accepted:
            keep.append(box)
    return keep


def screen_real_people(
    candidates: list[dict[str, Any]], parser: Any, labels: dict[str, int],
    person_detector: Any, count: int,
) -> list[dict[str, Any]]:
    """Reject fabric macros and flat-lays that ``classified.json`` calls worn.

    The archive heuristic can assign ``person_frac=0.995`` to a full-frame
    fabric macro.  A valid worn frame must contain face/hair and a meaningful
    amount of non-garment human labels in the semantic parse.
    """
    pool = sorted(candidates, key=lambda x: (-x["quality"], x["name"]))[: max(32, count * 12)]
    garment_ids = {labels[n] for n in ("top", "dress", "scarf")}
    person_ids = {labels[n] for n in ("face", "hair", "arms", "hands", "legs", "feet", "torso")}
    valid = []
    for row in pool:
        try:
            raw_boxes, (image_w, image_h) = person_detector(row["path"])
            boxes = nms(raw_boxes)
            if len(boxes) != 1:
                continue
            x0, y0, x1, y1 = boxes[0][:4]
            box_h, box_w = y1 - y0, max(x1 - x0, 1.0)
            if box_h / image_h < 0.25 or box_h / box_w < 1.30:
                continue
            im = Image.open(row["path"]).convert("RGB")
            seg = np.asarray(parser.predict(im))
        except Exception:
            continue
        garment = float(np.isin(seg, list(garment_ids)).mean())
        person = float(np.isin(seg, list(person_ids)).mean())
        face_hair = float(np.isin(seg, [labels["face"], labels["hair"]]).mean())
        if not (0.008 < garment < 0.70 and person > 0.018 and face_hair > 0.0008):
            continue
        row = dict(row)
        row["screen"] = {"garment_fraction": garment, "person_fraction": person,
                         "face_hair_fraction": face_hair, "detected_people": 1,
                         "person_box_height_fraction": box_h / image_h,
                         "person_box_aspect": box_h / box_w}
        valid.append(row)
        if len(valid) >= max(12, count * 3):
            break
    return valid


def best_garment(product_dir: Path, classified: dict[str, Any]) -> Path | None:
    choices = []
    for name, meta in classified.items():
        if meta.get("class") != "garment":
            continue
        p = resolve_image(product_dir, name)
        if p is None:
            continue
        coverage = float(meta.get("coverage", 0))
        if not 0.08 < coverage < 0.80:
            continue
        score = float(meta.get("bg_mean", 0)) - 4.0 * float(meta.get("bg_std", 99))
        choices.append((score, p))
    return max(choices, default=(0.0, None), key=lambda x: x[0])[1]


def worn_candidates(product_dir: Path, classified: dict[str, Any], min_person_frac: float) -> list[dict[str, Any]]:
    out = []
    for name, meta in classified.items():
        if meta.get("class") != "worn" or float(meta.get("person_frac", 0)) < min_person_frac:
            continue
        p = resolve_image(product_dir, name)
        if p is None:
            continue
        try:
            with Image.open(p) as im:
                area = im.width * im.height
        except OSError:
            continue
        quality = float(meta.get("person_frac", 0)) * math.log1p(area) + 0.02 * float(meta.get("hf", 0))
        out.append({"name": name, "path": str(p), "quality": quality,
                    "person_frac": float(meta.get("person_frac", 0))})
    return out


def build_manifest(
    raw: Path, out_dir: Path, poses_per_product: int, n_products: int, min_person_frac: float,
    person_parser: Any | None = None, parser_labels: dict[str, int] | None = None,
    person_detector: Any | None = None,
) -> dict[str, Any]:
    catalogue = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
    samples, products = [], []
    for item in sorted(catalogue, key=lambda x: int(x["product_id"])):
        if len(products) >= n_products or not is_top(item.get("name", "")):
            continue
        product_id = f"p{item['product_id']}"
        product_dir = raw / product_id
        classified_path = product_dir / "classified.json"
        if not classified_path.exists():
            continue
        classified = json.loads(classified_path.read_text(encoding="utf-8"))
        garment = best_garment(product_dir, classified)
        worn = worn_candidates(product_dir, classified, min_person_frac)
        if person_parser is not None and parser_labels is not None and person_detector is not None:
            worn = screen_real_people(worn, person_parser, parser_labels, person_detector,
                                      poses_per_product)
        if garment is None or not worn:
            continue
        chosen = choose_diverse_worn(worn, poses_per_product)
        products.append(product_id)
        for pose_index, row in enumerate(chosen):
            sample_id = f"{product_id}_{Path(row['name']).stem}"
            mask_root = out_dir / "masks"
            variant_root = out_dir / "variants"
            samples.append({
                "sample_id": sample_id,
                "product_id": product_id,
                # Filled after selection so folds are product-grouped *and*
                # balanced even for a small 26-product archive.
                "fold": None,
                "pose_index": pose_index,
                "protocol": "self_reconstruction_proxy",
                "person": row["path"],
                "garment": str(garment),
                "name": item.get("name"),
                "fabric_text": item.get("fabric_text"),
                "model_size": item.get("model_size"),
                "selection_screen": row.get("screen"),
                "ground_truth": {
                    "image": row["path"],
                    "mask": str((mask_root / "ground_truth" / f"{sample_id}.png").resolve()),
                },
                "variants": {
                    "fashn": {
                        "raw": str((variant_root / "fashn_raw" / f"{sample_id}.png").resolve()),
                        "image": str((variant_root / "fashn" / f"{sample_id}.png").resolve()),
                        "mask": str((mask_root / "fashn" / f"{sample_id}.png").resolve()),
                        "change_mask": str((mask_root / "fashn_change" / f"{sample_id}.png").resolve()),
                    },
                    "flux_vto_v2": {
                        "raw": str((variant_root / "flux_vto_v2_raw" / f"{sample_id}.png").resolve()),
                        "image": str((variant_root / "flux_vto_v2" / f"{sample_id}.png").resolve()),
                        "mask": str((mask_root / "flux_vto_v2" / f"{sample_id}.png").resolve()),
                        "change_mask": str((mask_root / "flux_vto_v2_change" / f"{sample_id}.png").resolve()),
                    },
                },
            })
    fold_by_product = {product_id: i % 5 for i, product_id in enumerate(products)}
    for sample in samples:
        sample["fold"] = fold_by_product[sample["product_id"]]
    return {
        "schema_version": 1,
        "dataset": "catalog_archive",
        "raw_root": str(raw.resolve()),
        "protocol": "self_reconstruction_proxy",
        "warning": ("No pixel-aligned cross-garment counterfactual exists. Product is the independent "
                    "statistical unit; multiple poses are repeated measurements, not extra garments."),
        "counts": {"products": len(products), "pose_observations": len(samples),
                   "poses_per_product_max": poses_per_product},
        "samples": samples,
    }


def _parser_and_ids() -> tuple[Any, list[int]]:
    import fashn_human_parser as fhp
    parser = fhp.FashnHumanParser()
    ids = [fhp.LABELS_TO_IDS[n] for n in fhp.BODY_COVERAGE_TO_LABELS["upper"]]
    return parser, ids


def garment_mask(parser: Any, ids: list[int], image: Image.Image) -> np.ndarray:
    seg = parser.predict(image)
    mask = np.isin(seg, ids).astype(np.uint8) * 255
    if mask.shape != (image.height, image.width):
        mask = np.asarray(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST))
    return mask


def save_mask(mask: np.ndarray, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), "L").save(p)


def feather_inside(binary: np.ndarray, radius: float = 2.0) -> np.ndarray:
    """Feather inward while keeping every pixel outside the union exactly zero."""
    b = (binary > 0).astype(np.uint8) * 255
    soft = np.asarray(Image.fromarray(b, "L").filter(ImageFilter.GaussianBlur(radius)), np.float32) / 255.0
    return soft * (binary > 0)


def hard_composite(base: Image.Image, generated: Image.Image, union_mask: np.ndarray) -> Image.Image:
    if generated.size != base.size:
        generated = generated.resize(base.size, Image.Resampling.LANCZOS)
    a, b = np.asarray(base, np.uint8), np.asarray(generated, np.uint8)
    m = feather_inside(union_mask)[..., None]
    out = a.copy()
    active = m[..., 0] > 0
    blend = np.rint(a.astype(np.float32) * (1 - m) + b.astype(np.float32) * m).clip(0, 255).astype(np.uint8)
    out[active] = blend[active]
    return Image.fromarray(out, "RGB")


def finalise_one(sample: dict[str, Any], variant: str, parser: Any, ids: list[int]) -> bool:
    spec = sample["variants"][variant]
    raw_path = Path(spec["raw"])
    if not raw_path.exists():
        return False
    base = Image.open(sample["person"]).convert("RGB")
    generated = Image.open(raw_path).convert("RGB")
    if generated.size != base.size:
        generated = generated.resize(base.size, Image.Resampling.LANCZOS)
    old = garment_mask(parser, ids, base)
    new = garment_mask(parser, ids, generated)
    union = np.maximum(old, new)
    if (new > 0).mean() < 0.005:
        print(f"  {sample['sample_id']} new garment mask too small; skipped")
        return False
    save_mask(new, spec["mask"])
    save_mask(union, spec["change_mask"])
    dst = Path(spec["image"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    hard_composite(base, generated, union).save(dst)
    return True


def command_build(args: argparse.Namespace) -> None:
    person_parser = parser_labels = person_detector = None
    if args.screen_person:
        import fashn_human_parser as fhp
        sys.path.insert(0, str(PROJECT / "dataset"))
        from person_detect import person_boxes
        person_parser = fhp.FashnHumanParser()
        parser_labels = fhp.LABELS_TO_IDS
        person_detector = person_boxes
    result = build_manifest(args.raw, args.out_dir, args.poses_per_product,
                            args.n_products, args.min_person_frac, person_parser, parser_labels,
                            person_detector)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    c = result["counts"]
    print(f"products={c['products']} pose_observations={c['pose_observations']} -> {args.manifest}")


def command_masks(args: argparse.Namespace) -> None:
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    parser, ids = _parser_and_ids()
    made = 0
    for sample in data["samples"][: args.limit]:
        if args.target == "ground_truth":
            spec = sample["ground_truth"]
            if args.resume and Path(spec["mask"]).exists():
                continue
            im = Image.open(spec["image"]).convert("RGB")
            save_mask(garment_mask(parser, ids, im), spec["mask"])
            made += 1
            print(f"  {sample['sample_id']}", flush=True)
        else:
            if finalise_one(sample, args.target, parser, ids):
                made += 1
                print(f"  {sample['sample_id']}", flush=True)
    print(f"made={made}")


def command_fashn(args: argparse.Namespace) -> None:
    # Keep imports here: build and manifest inspection do not need the large VTO stack.
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(PROJECT / "dataset"))
    sys.path.insert(0, r"C:\Dev\fashn-vton-1.5\src")
    from enhance import set_resolution, upper_roi
    from fashn_vton import TryOnPipeline

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    parser, ids = _parser_and_ids()
    pipe = TryOnPipeline(weights_dir=str(WEIGHTS))
    h, w = (int(v) for v in args.res.lower().split("x"))
    set_resolution(pipe, h, w)
    selected = data["samples"][: args.limit]
    t0 = time.time()
    for i, sample in enumerate(selected, 1):
        spec = sample["variants"]["fashn"]
        if args.resume and Path(spec["image"]).exists():
            continue
        base = Image.open(sample["person"]).convert("RGB")
        garment = Image.open(sample["garment"]).convert("RGB")
        box = upper_roi(base, "tops")
        crop = base.crop(box)
        started = time.time()
        result = pipe(person_image=crop, garment_image=garment, category="tops",
                      garment_photo_type="flat-lay", num_samples=1,
                      num_timesteps=args.steps, guidance_scale=1.5, seed=args.seed,
                      segmentation_free=False)
        generated_crop = result.images[0].resize(crop.size, Image.Resampling.LANCZOS)
        raw_full = base.copy()
        raw_full.paste(generated_crop, box[:2])
        raw_path = Path(spec["raw"])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_full.save(raw_path)
        finalise_one(sample, "fashn", parser, ids)
        elapsed = time.time() - t0
        remaining = elapsed / i * (len(selected) - i) / 60
        print(f"[{i}/{len(selected)}] {sample['sample_id']} {time.time()-started:.0f}s "
              f"remaining~{remaining:.0f}m", flush=True)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_OUT / "manifest.json")
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", type=Path, default=RAW)
    b.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    b.add_argument("--poses-per-product", type=int, default=4)
    b.add_argument("--n-products", type=int, default=100)
    b.add_argument("--min-person-frac", type=float, default=0.25)
    b.add_argument("--screen-person", action="store_true",
                   help="semantic-parser gate for real worn frames (run in FASHN environment)")
    b.set_defaults(func=command_build)
    m = sub.add_parser("masks")
    m.add_argument("--target", default="ground_truth",
                   choices=("ground_truth", "fashn", "flux_vto_v2"))
    m.add_argument("--limit", type=int, default=10**9)
    m.add_argument("--resume", action="store_true")
    m.set_defaults(func=command_masks)
    f = sub.add_parser("fashn")
    f.add_argument("--limit", type=int, default=10**9)
    f.add_argument("--steps", type=int, default=30)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--res", default="1296x864")
    f.add_argument("--resume", action="store_true")
    f.set_defaults(func=command_fashn)
    z = sub.add_parser("finalise")
    z.add_argument("--variant", required=True, choices=("fashn", "flux_vto_v2"))
    z.add_argument("--limit", type=int, default=10**9)
    z.add_argument("--resume", action="store_true")
    z.set_defaults(func=lambda a: command_masks(argparse.Namespace(
        manifest=a.manifest, target=a.variant, limit=a.limit, resume=a.resume)))
    return ap


def main() -> None:
    args = parser().parse_args()
    # The global --manifest default follows --out-dir for the common build call.
    if args.command == "build" and args.manifest == DEFAULT_OUT / "manifest.json":
        args.manifest = args.out_dir / "manifest.json"
    args.func(args)


if __name__ == "__main__":
    main()
