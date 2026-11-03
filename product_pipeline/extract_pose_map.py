"""Extract OpenPose-format pose maps from images and gate them on the base contract.

Two jobs:

1. Render a ControlNet-ready OpenPose skeleton (DWPose ships with FASHN, so the
   pose comes from the same estimator family the VTON side already trusts).

2. Decide, from the keypoints alone, whether the pose satisfies the base contract:
   both hands visible and clearly separated from the torso. Prompt-only posing hit
   that contract in 1 of 3 samples, which is why it has to become a gate rather
   than a hope. Once a pose passes it can be reused forever as a fixed condition.

Contract checks (COCO-18 body + 21-point hands, all normalised by shoulder width):
  - both wrists detected
  - both wrists below the shoulder line        -> rejects hand-at-head
  - each wrist stays on its own side of centre -> rejects crossed / folded arms
  - wrist lateral offset >= --min-separation   -> rejects arms clamped to torso
  - enough hand keypoints detected             -> rejects hands in pockets

Runs in the FASHN venv (DWPose + its weights live there):
    C:\\Dev\\fashn-vton-1.5\\.venv\\Scripts\\python.exe extract_pose_map.py ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

FASHN_SRC = r"C:\Dev\fashn-vton-1.5\src"
WEIGHTS = Path(r"C:\Dev\fashn-vton-1.5\weights\dwpose")

# COCO-18 indices used by DWPose.
NOSE, NECK = 0, 1
R_SHO, R_ELB, R_WRI = 2, 3, 4
L_SHO, L_ELB, L_WRI = 5, 6, 7
R_HIP, L_HIP = 8, 11


def _valid(pt) -> bool:
    # numpy comparisons yield np.bool_, which json.dumps refuses; coerce here so
    # every value that reaches the report is a builtin.
    return bool(pt is not None and pt[0] >= 0 and pt[1] >= 0)


def evaluate_contract(pose: dict, min_separation: float, min_hand_points: int) -> dict:
    body = np.asarray(pose["bodies"]["candidate"], dtype=float)
    hands = np.asarray(pose["hands"], dtype=float)

    def kp(i):
        return body[i] if i < len(body) else None

    checks: dict[str, bool] = {}
    detail: dict[str, float | None] = {}

    r_sho, l_sho, r_wri, l_wri = kp(R_SHO), kp(L_SHO), kp(R_WRI), kp(L_WRI)
    shoulders_ok = _valid(r_sho) and _valid(l_sho)
    checks["shoulders_detected"] = shoulders_ok
    checks["both_wrists_detected"] = _valid(r_wri) and _valid(l_wri)

    if shoulders_ok:
        shoulder_w = abs(float(r_sho[0]) - float(l_sho[0])) or 1e-6
        centre_x = (float(r_sho[0]) + float(l_sho[0])) / 2.0
        shoulder_y = (float(r_sho[1]) + float(l_sho[1])) / 2.0
        detail["shoulder_width_norm"] = round(shoulder_w, 4)

        # DWPose "right" is the subject's right, which sits at smaller x when the
        # subject faces the camera; use each shoulder's own sign so a turned pose
        # is still judged correctly.
        for tag, wri, sho in (("right", r_wri, r_sho), ("left", l_wri, l_sho)):
            if not _valid(wri):
                checks[f"{tag}_wrist_below_shoulder"] = False
                checks[f"{tag}_wrist_own_side"] = False
                checks[f"{tag}_wrist_separated"] = False
                detail[f"{tag}_separation"] = None
                continue
            checks[f"{tag}_wrist_below_shoulder"] = float(wri[1]) > shoulder_y
            side = np.sign(float(sho[0]) - centre_x)
            offset = (float(wri[0]) - centre_x) * side
            sep = offset / shoulder_w
            detail[f"{tag}_separation"] = round(float(sep), 4)
            checks[f"{tag}_wrist_own_side"] = sep > 0
            checks[f"{tag}_wrist_separated"] = sep >= min_separation
    else:
        detail["shoulder_width_norm"] = None

    # DWPose returns hands as (n_hands, 21, 2), not a flat stack.
    hb = hands.reshape(-1, 21, 2) if hands.size else np.empty((0, 21, 2))
    for idx, tag in enumerate(("left", "right")):
        if idx < len(hb):
            block = hb[idx]
            n_valid = int(((block[:, 0] >= 0) & (block[:, 1] >= 0)).sum())
        else:
            n_valid = 0
        detail[f"{tag}_hand_points"] = n_valid
        checks[f"{tag}_hand_visible"] = n_valid >= min_hand_points

    checks = {k: bool(v) for k, v in checks.items()}
    return {"passed": bool(all(checks.values())), "checks": checks, "detail": detail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--min-separation", type=float, default=0.28)
    ap.add_argument("--min-hand-points", type=int, default=8)
    args = ap.parse_args()

    if FASHN_SRC not in sys.path:
        sys.path.insert(0, FASHN_SRC)
    from fashn_vton.dwpose import DWposeDetector, draw_pose

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detector = DWposeDetector(checkpoints_dir=str(args.weights), device=args.device)

    records = []
    for img_path in args.images:
        with Image.open(img_path) as im:
            rgb = im.convert("RGB")
            arr = np.asarray(rgb)
        H, W = arr.shape[:2]

        pose = detector(arr, single=True)
        canvas = draw_pose(pose, H, W)
        pose_img = Image.fromarray(np.asarray(canvas, dtype=np.uint8), "RGB")
        pose_path = args.out_dir / f"{img_path.stem}_openpose.png"
        pose_img.save(pose_path)

        verdict = evaluate_contract(pose, args.min_separation, args.min_hand_points)
        rec = {
            "source": str(img_path),
            "pose_map": str(pose_path),
            "size": [W, H],
            **verdict,
        }
        (args.out_dir / f"{img_path.stem}_pose.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        records.append(rec)

        failed = [k for k, v in verdict["checks"].items() if not v]
        status = "PASS" if verdict["passed"] else "FAIL"
        print(f"[{status}] {img_path.name}")
        if failed:
            print(f"        failed: {', '.join(failed)}")
        print(f"        sep L/R = {verdict['detail'].get('left_separation')}/"
              f"{verdict['detail'].get('right_separation')}  "
              f"hand pts L/R = {verdict['detail'].get('left_hand_points')}/"
              f"{verdict['detail'].get('right_hand_points')}")

    summary = {
        "min_separation": args.min_separation,
        "min_hand_points": args.min_hand_points,
        "passed": [r["source"] for r in records if r["passed"]],
        "failed": [r["source"] for r in records if not r["passed"]],
        "records": records,
    }
    (args.out_dir / "pose_gate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[done] {len(summary['passed'])}/{len(records)} passed -> {args.out_dir}")


if __name__ == "__main__":
    main()
