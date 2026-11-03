"""FASHN VTON 1.5 segmentation-free single-pair runner with full QC artifacts.

본선 구조를 그대로 따른다 (run_segfree_ab.py 와 동일한 추론 설정).
    person -> upper_roi crop -> FASHN segfree -> 원래 프레임 복귀
    -> old/new upper-garment mask 합집합만 채택 -> 나머지는 원본 픽셀 하드 합성

run_segfree_ab.py 와 다른 점은 배치가 아니라 **한 pair 를 깊게 검사**한다는 것이다.
9장 품질검사 규칙이 요구하는 중간 산출물과 manifest 를 전부 남기고,
보호 영역(합집합 밖)이 실제로 원본과 같은지 수치로 증명한다.

FASHN 은 자기 venv 에만 있다. 반드시 다음으로 실행한다:
    C:\\Dev\\fashn-vton-1.5\\.venv\\Scripts\\python.exe run_fashn_pair_v2.py ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(r"fashion_pipeline")
WEIGHTS = Path(r"C:\Dev\fashn-vton-1.5\weights")


def _import_project():
    import sys

    for p in (
        str(PROJECT),
        str(PROJECT / "dataset"),
        str(PROJECT / "refiner"),
        r"C:\Dev\fashn-vton-1.5\src",
        str(Path(__file__).parent),
    ):
        if p not in sys.path:
            sys.path.insert(0, p)


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, "L").save(path)


def preservation_stats(base: Image.Image, final: Image.Image, union: np.ndarray) -> dict:
    """합집합 밖이 원본과 byte-identical 인지 검증한다.

    ★hard_composite 는 union 안쪽으로만 feather 하므로 이론상 밖은 무손실이다.
      이론을 믿지 말고 매번 측정한다 -- crop/resize 경로가 바뀌면 조용히 깨진다.
    """
    a = np.asarray(base, np.int16)
    b = np.asarray(final, np.int16)
    outside = ~(union > 0)
    diff = np.abs(a - b).max(axis=2)
    out_diff = diff[outside]
    changed = int((out_diff > 0).sum())
    return {
        "outside_pixels": int(outside.sum()),
        "outside_changed_pixels": changed,
        "outside_changed_ratio": float(changed / max(int(outside.sum()), 1)),
        "outside_mae": float(out_diff.mean()) if out_diff.size else 0.0,
        "outside_max_abs_error": int(out_diff.max()) if out_diff.size else 0,
        "byte_identical_outside": changed == 0,
        "union_pixels": int((union > 0).sum()),
        "union_ratio": float((union > 0).mean()),
    }


def contact_sheet(panels, path: Path, height: int = 900) -> None:
    """같은 높이로 맞춘 가로 대조 시트. 라벨은 상단에 굽는다."""
    from PIL import ImageDraw

    scaled = []
    for label, img in panels:
        w = max(1, int(img.width * height / img.height))
        scaled.append((label, img.convert("RGB").resize((w, height), Image.Resampling.LANCZOS)))
    pad, bar = 8, 34
    total_w = sum(im.width for _, im in scaled) + pad * (len(scaled) + 1)
    sheet = Image.new("RGB", (total_w, height + bar + pad * 2), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    x = pad
    for label, im in scaled:
        sheet.paste(im, (x, bar + pad))
        draw.text((x + 4, 9), label, fill=(255, 255, 255))
        x += im.width + pad
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=95)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--person", required=True, type=Path)
    ap.add_argument("--garment", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--baseline", type=Path, default=None,
                    help="기존 채택 결과. 주면 같은 크기 대조 시트를 만든다.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--res", default="1296x864")
    ap.add_argument("--category", default="tops")
    ap.add_argument("--garment-photo-type", default="flat-lay")
    args = ap.parse_args()

    _import_project()
    import fashn_human_parser as fhp
    from enhance import set_resolution, upper_roi
    from fashn_vton import TryOnPipeline
    from catalog_eval import cross_zone_masks, garment_mask, hard_composite

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    base = Image.open(args.person).convert("RGB")
    garment = Image.open(args.garment).convert("RGB")
    base.save(out / "01_person_original.png")
    garment.save(out / "02_garment_flatlay.png")

    box = upper_roi(base, args.category)
    crop = base.crop(box)
    crop.save(out / "03_person_crop.png")

    height, width = (int(v) for v in args.res.lower().split("x"))
    pipe = TryOnPipeline(weights_dir=str(WEIGHTS))
    set_resolution(pipe, height, width)

    started = time.time()
    result = pipe(
        person_image=crop,
        garment_image=garment,
        category=args.category,
        garment_photo_type=args.garment_photo_type,
        num_samples=1,
        num_timesteps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        segmentation_free=True,
    )
    elapsed = time.time() - started

    generated_crop = result.images[0].resize(crop.size, Image.Resampling.LANCZOS)
    generated_crop.save(out / "04_fashn_crop.png")
    raw = base.copy()
    raw.paste(generated_crop, box[:2])
    raw.save(out / "05_fashn_raw_fullframe.png")

    parser = fhp.FashnHumanParser()
    garment_ids = [fhp.LABELS_TO_IDS[n] for n in fhp.BODY_COVERAGE_TO_LABELS["upper"]]
    old_mask = garment_mask(parser, garment_ids, base)
    new_mask = garment_mask(parser, garment_ids, raw)
    zones = cross_zone_masks(old_mask, new_mask)
    union = np.maximum(old_mask, new_mask)

    save_mask(old_mask, out / "06_mask_old_garment.png")
    save_mask(new_mask, out / "07_mask_new_garment.png")
    save_mask(union, out / "08_mask_union_protect.png")
    save_mask(zones["erase"], out / "09_mask_erase_only.png")
    save_mask(zones["boundary"], out / "10_mask_boundary.png")

    final = hard_composite(base, raw, union)
    final_path = out / "11_final_hard_composite.png"
    final.save(final_path)

    stats = preservation_stats(base, final, union)

    panels = [("person", base), ("garment", garment), ("fashn_raw", raw), ("final_v2", final)]
    if args.baseline and args.baseline.exists():
        panels.append(("baseline_adopted", Image.open(args.baseline).convert("RGB")))
    contact_sheet(panels, out / "12_contact_sheet.jpg")

    # 1:1 확대 시트 -- 넥/어깨 경계는 축소하면 안 보인다.
    ys, xs = np.nonzero(union > 0)
    if xs.size:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        crops = [("person", base), ("final_v2", final)]
        if args.baseline and args.baseline.exists():
            crops.append(("baseline", Image.open(args.baseline).convert("RGB")))
        zoom = [(n, im.crop((x0, y0, x1, y1))) for n, im in crops]
        contact_sheet(zoom, out / "13_contact_sheet_zoom.jpg", height=1100)

    manifest = {
        "person": str(args.person),
        "garment": str(args.garment),
        "output": str(final_path),
        "model": "FASHN VTON 1.5",
        "weights": str(WEIGHTS),
        "category": args.category,
        "garment_photo_type": args.garment_photo_type,
        "segmentation_free": True,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance,
        "resolution": args.res,
        "upper_roi": [int(v) for v in box],
        "person_size": list(base.size),
        "elapsed_seconds": round(elapsed, 3),
        "outside_change_mask_policy": "original pixels copied exactly",
        "preservation": stats,
        "baseline_compared": str(args.baseline) if args.baseline else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
