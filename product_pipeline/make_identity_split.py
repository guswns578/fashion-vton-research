"""Deterministic 40/10 train/holdout split for seller_identity_faces_50_v1.

Split is GROUPED BY product_id: the 50 images come from only 38 products, and 12
products contribute 2 frames each. Two frames of one product are the same shoot
(same day, lighting, outfit, near-identical face), so letting one land in train
and its sibling in holdout would inflate any holdout identity score and destroy
the split's value as a generalization test.

Groups are stratified into quartiles by face_height_px so the holdout spans the
same pixel-density range as train rather than collecting only small/large faces.

Writes to a NEW directory; never mutates the existing dataset files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image

DATASET = Path(r"fashion_pipeline/dataset/seller_identity_faces_50_v1")
HOLDOUT_TARGET = 10


def load_rows(manifest: Path) -> list[dict]:
    with manifest.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["product_id"], []).append(row)
    for members in groups.values():
        members.sort(key=lambda r: r["final_index"])
    return groups


def stratify(groups: dict[str, list[dict]], bins: int) -> dict[str, list[str]]:
    """Bucket product_ids into `bins` quantiles of mean face height."""
    scored = sorted(
        groups.items(),
        key=lambda kv: sum(r["face_height_px"] for r in kv[1]) / len(kv[1]),
    )
    strata: dict[str, list[str]] = {}
    for pos, (pid, _) in enumerate(scored):
        idx = min(bins - 1, pos * bins // len(scored))
        strata.setdefault(f"q{idx + 1}", []).append(pid)
    return strata


def pick_holdout(groups: dict[str, list[dict]], seed: int, bins: int) -> list[str]:
    """Round-robin across strata until exactly HOLDOUT_TARGET images are held out."""
    rng = random.Random(seed)
    pools = {name: pids[:] for name, pids in stratify(groups, bins).items()}
    for pids in pools.values():
        rng.shuffle(pids)

    chosen: list[str] = []
    count = 0
    order = sorted(pools)
    while count < HOLDOUT_TARGET:
        progressed = False
        for name in order:
            if count >= HOLDOUT_TARGET:
                break
            pool = pools[name]
            # take the first candidate that still fits the remaining budget
            for i, pid in enumerate(pool):
                size = len(groups[pid])
                if count + size <= HOLDOUT_TARGET:
                    chosen.append(pool.pop(i))
                    count += size
                    progressed = True
                    break
        if not progressed:
            raise RuntimeError(f"cannot reach {HOLDOUT_TARGET} images; stuck at {count}")
    return sorted(chosen)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def contact_sheet(rows: list[dict], out_path: Path, cols: int, thumb: int) -> None:
    imgs = []
    for row in rows:
        with Image.open(row["selected_path"]) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb, thumb), Image.LANCZOS)
            imgs.append((im.copy(), Path(row["selected_path"]).stem))
    if not imgs:
        return
    rows_n = (len(imgs) + cols - 1) // cols
    cell_w = max(i.width for i, _ in imgs)
    cell_h = max(i.height for i, _ in imgs)
    sheet = Image.new("RGB", (cols * cell_w, rows_n * cell_h), (24, 24, 24))
    for idx, (im, _) in enumerate(imgs):
        x = (idx % cols) * cell_w + (cell_w - im.width) // 2
        y = (idx // cols) * cell_h + (cell_h - im.height) // 2
        sheet.paste(im, (x, y))
    sheet.save(out_path, quality=92)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--out-dir", type=Path, default=DATASET / "splits" / "group_40_10_v1")
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--verify-sha", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.dataset / "manifest.jsonl")
    groups = group_rows(rows)

    holdout_pids = pick_holdout(groups, args.seed, args.bins)
    holdout = [r for pid in holdout_pids for r in groups[pid]]
    train = [r for r in rows if r["product_id"] not in set(holdout_pids)]

    assert len(holdout) == HOLDOUT_TARGET, len(holdout)
    assert len(train) + len(holdout) == len(rows)
    assert not (set(r["product_id"] for r in train) & set(holdout_pids)), "product leak"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train", train), ("holdout", holdout)):
        with (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in subset:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        contact_sheet(subset, args.out_dir / f"{name}_contact_sheet.jpg", cols=5, thumb=384)

    def stats(subset: list[dict]) -> dict:
        fh_px = sorted(r["face_height_px"] for r in subset)
        return {
            "images": len(subset),
            "products": len(set(r["product_id"] for r in subset)),
            "face_height_px": {
                "min": round(fh_px[0], 2),
                "median": round(fh_px[len(fh_px) // 2], 2),
                "max": round(fh_px[-1], 2),
            },
            "min_megapixels": round(min(r["megapixels"] for r in subset), 4),
        }

    summary = {
        "schema_version": 1,
        "split_id": "group_40_10_v1",
        "source_manifest": str(args.dataset / "manifest.jsonl"),
        "grouping": "product_id (prevents same-shoot leakage across the split)",
        "stratified_by": f"mean face_height_px, {args.bins} quantile bins",
        "seed": args.seed,
        "holdout_product_ids": holdout_pids,
        "train": stats(train),
        "holdout": stats(holdout),
    }

    if args.verify_sha:
        mismatches = [
            r["selected_path"] for r in rows if sha256_of(Path(r["selected_path"])) != r["sha256"]
        ]
        summary["sha256_verified"] = len(rows) - len(mismatches)
        summary["sha256_mismatches"] = mismatches

    (args.out_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
