"""Build fair 1x-vs-2x visual checks for the generated-person VTON chain.

The baseline is enlarged to the 2x canvas with Lanczos before a 1:1 crop is
made.  Therefore any detail visible only in the 2x arm comes from running the
face detailer and VTON on the larger canvas, not from displaying it larger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

YUNET = Path(r"fashion_pipeline/models/opencv_face/face_detection_yunet_2023mar.onnx")


def labelled_row(items: list[tuple[str, Image.Image]], out: Path, height: int) -> None:
    pad, bar = 10, 42
    scaled: list[tuple[str, Image.Image]] = []
    for label, image in items:
        w = max(1, round(image.width * height / image.height))
        scaled.append((label, image.convert("RGB").resize((w, height), Image.Resampling.LANCZOS)))
    width = sum(im.width for _, im in scaled) + pad * (len(scaled) + 1)
    sheet = Image.new("RGB", (width, height + bar + pad * 2), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    x = pad
    for label, image in scaled:
        draw.text((x + 4, 13), label, fill=(255, 255, 255))
        sheet.paste(image, (x, bar + pad))
        x += image.width + pad
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=96)


def padded_bbox(mask: np.ndarray, margin: int, size: tuple[int, int]) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return 0, 0, size[0], size[1]
    return (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(size[0], int(xs.max()) + margin + 1),
        min(size[1], int(ys.max()) + margin + 1),
    )


def product_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    arr = np.asarray(image.convert("RGB"))
    # Ignore the bottom label and retain the largest non-background component.
    foreground = (arr[: int(arr.shape[0] * 0.90)].min(axis=2) < 238).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    if n <= 1:
        return 0, 0, image.width, int(image.height * 0.90)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[idx, :4]
    return int(x), int(y), int(x + w), int(y + h)


def face_box(image: Image.Image, pad: float = 2.2) -> tuple[int, int, int, int]:
    arr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.6, 0.3, 5000)
    detector.setInputSize((w, h))
    _, faces = detector.detect(arr)
    if faces is None:
        raise RuntimeError("face not detected")
    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4]
    cx, cy = x + fw / 2, y + fh / 2
    side = min(max(fw, fh) * pad, w, h)
    left = int(max(0, min(cx - side / 2, w - side)))
    top = int(max(0, min(cy - side / 2, h - side)))
    return left, top, int(left + side), int(top + side)


def detail_energy(image: Image.Image) -> dict[str, float]:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    high = gray.astype(np.float32) - cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    return {
        "laplacian_variance": round(float(lap.var()), 3),
        "highpass_rms": round(float(np.sqrt(np.mean(high * high))), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--hires", required=True, type=Path)
    ap.add_argument("--hires-mask", required=True, type=Path)
    ap.add_argument("--product", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    baseline = Image.open(args.baseline).convert("RGB")
    hires = Image.open(args.hires).convert("RGB")
    product = Image.open(args.product).convert("RGB")
    if hires.width % baseline.width or hires.height % baseline.height:
        raise ValueError("hires canvas must be an integer multiple of baseline")
    baseline_up = baseline.resize(hires.size, Image.Resampling.LANCZOS)
    hires_down = hires.resize(baseline.size, Image.Resampling.LANCZOS)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    labelled_row(
        [("A 1x adopted", baseline), ("B 2x downsampled", hires_down), ("product", product)],
        args.out_dir / "60_full_ab.jpg",
        1050,
    )

    mask = np.asarray(Image.open(args.hires_mask).convert("L")) > 0
    garment_box = padded_bbox(mask, 36, hires.size)
    product_box = product_bbox(product)
    base_garment = baseline_up.crop(garment_box)
    hires_garment = hires.crop(garment_box)
    product_garment = product.crop(product_box)
    labelled_row(
        [
            ("A 1x enlarged (interpolated)", base_garment),
            ("B 2x native VTON", hires_garment),
            ("product reference", product_garment),
        ],
        args.out_dir / "61_garment_1to1_ab.jpg",
        1100,
    )

    b_face_box = face_box(baseline_up)
    h_face_box = face_box(hires)
    base_face = baseline_up.crop(b_face_box)
    hires_face = hires.crop(h_face_box)
    labelled_row(
        [("A 1x enlarged", base_face), ("B 2x detailer native", hires_face)],
        args.out_dir / "62_face_1to1_ab.jpg",
        900,
    )

    report = {
        "baseline": str(args.baseline),
        "hires": str(args.hires),
        "baseline_size": list(baseline.size),
        "hires_size": list(hires.size),
        "pixel_count_multiplier": round(
            hires.width * hires.height / (baseline.width * baseline.height), 3
        ),
        "garment_box_on_hires": list(garment_box),
        "garment_detail_energy_same_canvas": {
            "baseline_lanczos_2x": detail_energy(base_garment),
            "hires_native": detail_energy(hires_garment),
        },
        "face_boxes": {"baseline_lanczos_2x": list(b_face_box), "hires_native": list(h_face_box)},
        "face_detail_energy_same_canvas": {
            "baseline_lanczos_2x": detail_energy(base_face),
            "hires_native": detail_energy(hires_face),
        },
        "interpretation_limit": (
            "Sharpness metrics measure detail energy, not garment identity or physical correctness; "
            "the 1:1 sheets remain the deciding evidence."
        ),
    }
    (args.out_dir / "63_hires_ab_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
