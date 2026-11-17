"""기존 옷 잔존(§9-2)을 눈이 아니라 수치로 잡는 게이트.

하드 합성은 union 마스크 **밖**을 원본 픽셀로 되돌린다. 따라서 파서가 옛 옷을
조금이라도 덜 잡으면, 그 덜 잡힌 띠가 최종 이미지에 **원본 옷 색 그대로** 남는다.
축소 시트에서는 이게 그림자처럼 보여서 놓치기 쉽다.

측정 방법
    union 을 반경 R 로 팽창시킨 고리(ring) 안에서, 원본이 '옛 옷 색' 인 픽셀을 센다.
    ring 으로 제한하는 이유는 화면 어딘가의 남색 물체를 잔존으로 오인하지 않기
    위해서다. 잔존은 정의상 새 옷 경계에 붙어서 나온다.

★색만으로는 못 가른다 -- 첫 판에서 이걸로 틀렸다.
  이 pair 는 옛 옷이 네이비 스트라이프인데 **바지가 인디고 데님**이라 RGB 가 거의 같다
  (옛 옷 어두운색 27,27,46 vs 데님). 색 조건만 쓰면 밑단 전체가 잔존으로 잡히지만
  실제로는 정상 보존된 바지다. 머리카락도 같은 이유로 오검출된다.
  그래서 원본 파서 라벨로 후보를 거른다: 옛 옷이 덜 잡혔다면 그 화소는 파서가
  top 이 아닌 **arms/hands/torso/face/background** 로 잘못 부른 자리에 있다.
  pants/belt/hair/hat/bag/glasses/jewelry/legs/feet 는 정당하게 어두울 수 있으므로 뺀다.

잔존은 파서가 놓쳐서 생긴다. 그러므로 파서로 잔존을 '찾을' 수는 없고,
파서로 **아닌 것을 지우는** 데에만 쓴다. 라벨별 내역을 전부 출력해
사람이 마지막 판정을 하게 한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ID_TO_LABEL = {
    0: "background",
    1: "face",
    2: "hair",
    3: "top",
    4: "dress",
    5: "skirt",
    6: "pants",
    7: "belt",
    8: "bag",
    9: "hat",
    10: "scarf",
    11: "glasses",
    12: "arms",
    13: "hands",
    14: "legs",
    15: "feet",
    16: "torso",
    17: "jewelry",
}
# 옛 옷이 덜 잡혔을 때 그 화소가 받게 되는 라벨만 후보로 둔다.
# ★background 를 빼면 안 된다 -- 목 뒤로 솟은 폴로 카라는 파서가 통째로 background 로
#   부르고, 그 852px 카라 조각이 최종본에 **목 옆에 뜬 남색 파편**으로 남았다.
#   background 를 후보에서 빼면 이 치명적 결함을 놓친다(실제로 한 번 놓쳤다).
#   대신 회색 벽돌 배경의 오검출은 **연결 성분 크기**로 거른다:
#   진짜 잔존은 뭉친 덩어리, 벽돌 오검출은 흩어진 실오라기다.
CANDIDATE_LABELS = {"background", "face", "arms", "hands", "torso"}
MIN_BLOB_PIXELS = 80


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255, "L")
    for _ in range(max(1, radius // 2)):
        img = img.filter(ImageFilter.MaxFilter(5))
    return np.asarray(img) > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--ring-radius", type=int, default=24)
    ap.add_argument("--tol", type=float, default=26.0)
    ap.add_argument("--fail-pixels", type=int, default=150)
    args = ap.parse_args()

    import sys

    sys.path.insert(0, r"C:\Dev\fashn-vton-1.5\src")
    import fashn_human_parser as fhp

    run = args.run_dir
    person_img = Image.open(run / "01_person_original.png").convert("RGB")
    person = np.asarray(person_img, np.float32)
    final = np.asarray(Image.open(run / "11_final_hard_composite.png").convert("RGB"), np.float32)
    old = np.asarray(Image.open(run / "06_mask_old_garment.png").convert("L")) > 0
    union = np.asarray(Image.open(run / "08_mask_union_protect.png").convert("L")) > 0

    seg = fhp.FashnHumanParser().predict(person_img)
    seg = np.asarray(seg)
    if seg.shape != union.shape:
        seg = np.asarray(
            Image.fromarray(seg.astype(np.uint8)).resize(
                (union.shape[1], union.shape[0]), Image.Resampling.NEAREST
            )
        )
    Image.fromarray((seg.astype(np.uint8) * 14) % 255, "L").save(run / "15_person_labels.png")

    src = person[old]
    if src.size == 0:
        raise SystemExit("old mask is empty")
    lum = src.mean(axis=1)
    split = np.median(lum)
    clusters = [src[lum <= split].mean(axis=0), src[lum > split].mean(axis=0)]

    ring = dilate(union, args.ring_radius) & ~union
    idx = np.nonzero(ring)
    cand = person[idx]
    dist = np.minimum.reduce([np.linalg.norm(cand - c, axis=1) for c in clusters])
    colour_hit = dist < args.tol
    same = np.all(np.abs(final[idx] - person[idx]) < 1.0, axis=1)
    labels = seg[idx]

    raw_hit = colour_hit & same
    breakdown = {}
    for lid in np.unique(labels[raw_hit]):
        breakdown[ID_TO_LABEL.get(int(lid), str(int(lid)))] = int((labels[raw_hit] == lid).sum())

    keep = np.array([ID_TO_LABEL.get(int(l), "?") in CANDIDATE_LABELS for l in labels])
    residual = raw_hit & keep

    cand_mask = np.zeros(union.shape, np.uint8)
    cand_mask[idx[0][residual], idx[1][residual]] = 1

    # 덩어리만 남긴다. 벽돌/피부 하이라이트 오검출은 얇게 흩어져 여기서 죽는다.
    import cv2

    n, lab_im, stats, _ = cv2.connectedComponentsWithStats(cand_mask, connectivity=8)
    resid_mask = np.zeros(union.shape, bool)
    zones = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_BLOB_PIXELS:
            continue
        comp = lab_im == i
        resid_mask |= comp
        dom = {}
        for lid in np.unique(seg[comp]):
            dom[ID_TO_LABEL.get(int(lid), str(int(lid)))] = int((seg[comp] == lid).sum())
        zones.append(
            {
                "pixels": area,
                "bbox": [
                    int(stats[i, cv2.CC_STAT_LEFT]),
                    int(stats[i, cv2.CC_STAT_TOP]),
                    int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]),
                    int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]),
                ],
                "original_labels": dict(sorted(dom.items(), key=lambda kv: -kv[1])),
            }
        )
    zones.sort(key=lambda z: -z["pixels"])
    Image.fromarray(resid_mask.astype(np.uint8) * 255, "L").save(
        run / "14_residual_old_garment.png"
    )

    report = {
        "old_garment_colors_learned": [[round(float(v), 1) for v in c] for c in clusters],
        "ring_radius": args.ring_radius,
        "tolerance": args.tol,
        "ring_pixels": int(ring.sum()),
        "min_blob_pixels": MIN_BLOB_PIXELS,
        "colour_only_hits": int(raw_hit.sum()),
        "colour_only_hits_by_original_label": breakdown,
        "excluded_labels_note": "pants/hair/belt 등은 정당하게 어두울 수 있어 후보에서 제외",
        "residual_pixels": int(resid_mask.sum()),
        "residual_ratio_of_ring": float(resid_mask.sum() / max(int(ring.sum()), 1)),
        "scattered_candidates_dropped": int(int(cand_mask.sum()) - int(resid_mask.sum())),
        "blobs": zones,
        "verdict": "FAIL" if int(resid_mask.sum()) > args.fail_pixels else "PASS",
    }
    (run / "14_residual_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
