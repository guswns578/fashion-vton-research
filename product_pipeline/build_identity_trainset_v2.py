"""Rebuild the identity training set with real per-image captions and face crops.

v1 failed. Two causes, both fixed here.

1. All 40 captions were the SAME string, and several of its claims were false for
   the image ("full body photo, standing, natural light" on half-body, seated,
   indoor-artificial shots). The rule for an identity LoRA is to caption
   everything you do NOT want folded into the trigger token; v1 captioned nothing
   that varied, so outfit, framing, pose and lighting were all absorbed into
   `hgnst_model` instead of the face.

   Every varying attribute here is recoverable from metadata already on disk --
   no VLM needed:
     - garment   <- product_name in the curated pair manifest (40/40 matched)
     - framing   <- face_height_px / image height
     - direction <- frontal_score / yaw_proxy
     - lighting  <- mean_luma / contrast

2. v1 trained on body crops only, where the face is ~130px and collapses to ~16px
   in SDXL latent space. v1's face-crop bucket was 1024, which needed a 2.93x
   median upsample, so it was skipped entirely. At a 512 bucket the median
   upsample is ~1.5x, which is acceptable -- the earlier "face crops unusable"
   call was against the wrong target size.

Writes to a NEW dataset directory; v1 is left intact for comparison.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image

BODY_BUCKET = (896, 1152)
FACE_BUCKET = (512, 512)

CURATED = Path(
    r"fashion_pipeline/dataset/catalog_curated/tops_v1/pairs_selected.jsonl"
)


def product_names() -> dict[str, str]:
    names: dict[str, str] = {}
    with CURATED.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            names.setdefault(r["product_id"], r["product_name"])
    return names


def clean_garment(name: str) -> str:
    """Strip promo/pricing noise and brand prefixes from a product title."""
    n = re.sub(r"\[.*?\]", " ", name)            # [가을진행], [2장 구매 시 …]
    n = re.sub(r"\d+\s*원", " ", n)
    n = re.sub(r"[^A-Za-z ]+", " ", n)            # keep the English descriptor
    n = re.sub(r"\s+", " ", n).strip().lower()
    words = n.split()
    # Product titles start with a short house code (e.g. "Od", "Se", "Ay").
    if words and len(words[0]) <= 2:
        words = words[1:]
    return " ".join(words) if words else "top"


def framing(row: dict) -> str:
    ratio = row["face_height_px"] / max(row["height"], 1)
    if ratio < 0.09:
        return "full body photo"
    if ratio < 0.16:
        return "three quarter body photo"
    return "upper body photo"


def direction(row: dict) -> str:
    return "facing the camera" if row.get("frontal_score", 0) >= 0.55 else "turned to the side"


def lighting(row: dict) -> str:
    luma = row.get("mean_luma", 120)
    if luma < 95:
        return "dim indoor light"
    if luma > 150:
        return "bright daylight"
    return "soft indoor light"


def body_crop(im: Image.Image, face: list[float]) -> Image.Image:
    fx, fy, fw, fh = face
    cx = fx + fw / 2
    ar = BODY_BUCKET[0] / BODY_BUCKET[1]
    top = max(0, fy - fh * 1.2)
    ch = min(im.height - top, im.height)
    cw = ch * ar
    if cw > im.width:
        cw = im.width
        ch = cw / ar
        top = max(0, min(fy - fh * 1.2, im.height - ch))
    left = max(0, min(cx - cw / 2, im.width - cw))
    box = (int(left), int(top), int(left + cw), int(top + ch))
    return im.crop(box).resize(BODY_BUCKET, Image.LANCZOS)


def face_crop(im: Image.Image, face: list[float], expand: float):
    fx, fy, fw, fh = face
    cx, cy = fx + fw / 2, fy + fh / 2
    side = min(max(fw, fh) * expand, im.width, im.height)
    left = max(0, min(cx - side / 2, im.width - side))
    top = max(0, min(cy - side / 2, im.height - side))
    crop = im.crop((int(left), int(top), int(left + side), int(top + side)))
    return crop.resize(FACE_BUCKET, Image.LANCZOS), FACE_BUCKET[0] / max(1, crop.width)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split-dir",
        type=Path,
        default=Path(r"fashion_pipeline/dataset/seller_identity_faces_50_v1/splits/group_40_10_v1"),
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=Path(r"fashion_pipeline/dataset/seller_identity_trainset_v2"),
    )
    ap.add_argument("--token", default="hgnst_model")
    ap.add_argument("--class-word", default="man")
    ap.add_argument("--face-expand", type=float, default=2.4)
    ap.add_argument("--max-upsample", type=float, default=2.2)
    args = ap.parse_args()

    names = product_names()
    report = {}

    for split in ("train", "holdout"):
        rows = [json.loads(l) for l in (args.split_dir / f"{split}.jsonl").open(encoding="utf-8") if l.strip()]
        bdir = args.out_dir / split / "body"
        fdir = args.out_dir / split / "face"
        bdir.mkdir(parents=True, exist_ok=True)
        fdir.mkdir(parents=True, exist_ok=True)

        captions, ups, n_face = [], [], 0
        for row in rows:
            src = Path(row["selected_path"])
            stem = src.stem
            garment = clean_garment(names.get(row["product_id"], "top"))
            with Image.open(src) as raw:
                im = raw.convert("RGB")
                face = row["face_box"]

                cap_body = (
                    f"{args.token} {args.class_word}, {framing(row)}, {direction(row)}, "
                    f"wearing {garment}, {lighting(row)}, photograph"
                )
                body_crop(im, face).save(bdir / f"{stem}.png")
                (bdir / f"{stem}.txt").write_text(cap_body, encoding="utf-8")
                captions.append(cap_body)

                fc, up = face_crop(im, face, args.face_expand)
                ups.append(round(up, 2))
                if up <= args.max_upsample:
                    cap_face = (
                        f"{args.token} {args.class_word}, close up face portrait, "
                        f"{direction(row)}, {lighting(row)}, photograph"
                    )
                    fc.save(fdir / f"{stem}.png")
                    (fdir / f"{stem}.txt").write_text(cap_face, encoding="utf-8")
                    n_face += 1

        ups.sort()
        report[split] = {
            "images": len(rows),
            "body_crops": len(rows),
            "face_crops_emitted": n_face,
            "face_upsample": {"min": ups[0], "median": ups[len(ups) // 2], "max": ups[-1]},
            "unique_captions": len(set(captions)),
            "example_captions": sorted(set(captions))[:4],
        }

    manifest = {
        "schema_version": 2,
        "dataset": "seller_identity_trainset_v2",
        "fixes_over_v1": [
            "per-image captions derived from product_name + face geometry + luma",
            "face crops at a 512 bucket instead of 1024",
        ],
        "trigger_token": args.token,
        "body_bucket": BODY_BUCKET,
        "face_bucket": FACE_BUCKET,
        "splits": report,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "trainset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
