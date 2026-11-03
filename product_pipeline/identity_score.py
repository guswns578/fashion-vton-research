"""Score generated faces against the held-out real photos, objectively.

Every identity judgement so far in this project has been my eye on a contact
sheet. This puts a number on it, in the SAME embedding space the dataset was
curated with: OpenCV YuNet detection + SFace 128-d recognition
(`select_catalog_faces.py` used exactly these two models to build
face_embeddings.npz), so scores are comparable to the curation.

Reference point that matters: the holdout-vs-holdout similarity. Two photos of
the same real person score some value X; a generated face has to land near X to
count as the same identity. Comparing a generated score to an absolute threshold
in the abstract is meaningless, so this reports the real-person baseline first.

OpenCV's published same-identity threshold for SFace cosine is ~0.363, but the
holdout baseline computed here is the honest yardstick for this specific person
and image quality.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

MODEL_ROOT = Path(r"fashion_pipeline/models/opencv_face")
SFACE_THRESHOLD = 0.363  # OpenCV reference for "same identity", cosine


def build(model_root: Path):
    det = cv2.FaceDetectorYN.create(
        str(model_root / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6, 0.3, 5000
    )
    rec = cv2.FaceRecognizerSF.create(
        str(model_root / "face_recognition_sface_2021dec.onnx"), ""
    )
    return det, rec


def embed(path: Path, det, rec) -> np.ndarray | None:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    ok, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    # Largest detected face.
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    aligned = rec.alignCrop(img, faces[0])
    feat = rec.feature(aligned)
    v = np.asarray(feat, dtype=np.float32).ravel()
    return v / (np.linalg.norm(v) + 1e-9)


def collect(paths: list[Path], det, rec) -> dict[str, np.ndarray]:
    out = {}
    for p in paths:
        v = embed(p, det, rec)
        if v is None:
            print(f"  [no face detected] {p.name}")
        else:
            # Different pipeline stages commonly reuse the same basename
            # (for example the 1x and 2x detailer outputs).  A basename key
            # silently overwrote one arm of an A/B comparison, so retain the
            # resolved path as the stable unique identifier.
            out[str(p.resolve())] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", nargs="+", type=Path, required=True,
                    help="generated image files or directories")
    ap.add_argument("--holdout-jsonl", type=Path,
                    default=Path(r"fashion_pipeline/dataset/seller_identity_faces_50_v1/splits/group_40_10_v1/holdout.jsonl"))
    ap.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    det, rec = build(args.model_root)

    hold_paths = [
        Path(json.loads(l)["selected_path"])
        for l in args.holdout_jsonl.open(encoding="utf-8") if l.strip()
    ]
    print(f"[holdout] {len(hold_paths)} real photos")
    hold = collect(hold_paths, det, rec)
    H = np.stack(list(hold.values()))

    # Real-person baseline: how similar are two genuine photos of him?
    pairs = [float(a @ b) for a, b in combinations(H, 2)]
    pairs_sorted = sorted(pairs)
    baseline = {
        "n_pairs": len(pairs),
        "min": round(pairs_sorted[0], 4),
        "p25": round(pairs_sorted[len(pairs) // 4], 4),
        "median": round(pairs_sorted[len(pairs) // 2], 4),
        "mean": round(float(np.mean(pairs)), 4),
        "max": round(pairs_sorted[-1], 4),
    }
    print(f"[baseline] holdout-vs-holdout cosine: median={baseline['median']} "
          f"mean={baseline['mean']} range={baseline['min']}..{baseline['max']}")

    gen_paths: list[Path] = []
    for g in args.generated:
        gen_paths.extend(sorted(g.glob("*.png"))) if g.is_dir() else gen_paths.append(g)
    print(f"[generated] {len(gen_paths)} images")
    gen = collect(gen_paths, det, rec)

    rows = []
    for name, v in gen.items():
        sims = H @ v
        rows.append({
            "image": name,
            "max_vs_holdout": round(float(sims.max()), 4),
            "mean_vs_holdout": round(float(sims.mean()), 4),
            "passes_sface_threshold": bool(sims.mean() >= SFACE_THRESHOLD),
        })
    rows.sort(key=lambda r: -r["mean_vs_holdout"])

    print(f"\n{'image':46s} {'mean':>8s} {'max':>8s}  same-id?")
    for r in rows:
        print(f"{r['image'][:46]:46s} {r['mean_vs_holdout']:8.4f} {r['max_vs_holdout']:8.4f}  "
              f"{'YES' if r['passes_sface_threshold'] else 'no'}")

    if rows:
        gm = float(np.mean([r["mean_vs_holdout"] for r in rows]))
        print(f"\n[verdict] generated mean={gm:.4f} vs real-person baseline "
              f"median={baseline['median']}  ->  "
              f"{round(100*gm/baseline['median'])}% of the real baseline")

    report = {"sface_threshold": SFACE_THRESHOLD, "holdout_baseline": baseline, "generated": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
