"""Drape-oriented metrics for virtual try-on evaluation.

This module deliberately separates three quantities that are often mixed:

* ``measurement_noise``: changes caused by resize/JPEG/mask jitter;
* ``pose_variation``: real variation between photos of the same garment;
* ``model_error``: a generated result compared with its matching worn photo.

The measurements are image-space proxies, not material constants.  In
particular, a true cloth-to-body air gap cannot be recovered from one RGB
photo.  ``body_gap_proxy`` is therefore emitted only when a separate estimated
body-without-garment mask is supplied.

The script has no scipy/opencv dependency.  It runs with NumPy and Pillow:

    python drape_metrics.py --manifest eval_manifest.json --out drape_eval.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFilter

METRIC_DIRECTIONS = {
    # These are descriptors, so neither larger nor smaller is intrinsically
    # better.  Model comparison uses absolute distance to ground truth.
    "fold_contrast": "match",
    "wrinkle_density": "match",
    "wrinkle_strength_p90": "match",
    "wrinkle_orientation_entropy": "match",
    "wrinkle_bend_per_100px": "match",
    "macro_energy_ratio": "match",
    "meso_energy_ratio": "match",
    "micro_energy_ratio": "match",
    "fold_frequency_cpg": "match",
    "hem_curvature": "match",
    "hem_sag_ratio": "match",
    "hem_roughness_ratio": "match",
    "garment_height_image_ratio": "match",
    "garment_width_image_ratio": "match",
    "garment_aspect_ratio": "match",
    "hem_y_image_ratio": "match",
    "silhouette_flare_ratio": "match",
    "silhouette_width_cv": "match",
    "body_gap_proxy": "match",
}


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and np.isfinite(x)


def _as_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _as_mask(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as im:
        m = im.convert("L")
        if size is not None and m.size != size:
            m = m.resize(size, Image.Resampling.NEAREST)
        return np.asarray(m, np.float32) / 255.0


def _morph(mask: np.ndarray, radius: int, dilate: bool = False) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    size = radius * 2 + 1
    im = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), "L")
    filt = ImageFilter.MaxFilter(size) if dilate else ImageFilter.MinFilter(size)
    return np.asarray(im.filter(filt), np.float32) / 255.0


def _blur01(a: np.ndarray, radius: float) -> np.ndarray:
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "L")
    return np.asarray(im.filter(ImageFilter.GaussianBlur(radius)), np.float32) / 255.0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    yy, xx = np.nonzero(mask > 0.5)
    if not len(xx):
        return None
    return int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1


def _normalise_roi(
    image: Image.Image, mask: np.ndarray, body_mask: np.ndarray | None, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Crop to garment and pad to a square without aspect-ratio distortion."""
    box = _bbox(mask)
    if box is None:
        raise ValueError("empty garment mask")
    x0, y0, x1, y1 = box
    # Small context makes the hem stable without letting background dominate.
    pad = max(2, round(max(x1 - x0, y1 - y0) * 0.015))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(image.width, x1 + pad), min(image.height, y1 + pad)
    rgb = np.asarray(image, np.uint8)[y0:y1, x0:x1]
    gm = mask[y0:y1, x0:x1]
    bm = None if body_mask is None else body_mask[y0:y1, x0:x1]
    h, w = gm.shape
    scale = size / max(h, w)
    nw, nh = max(2, round(w * scale)), max(2, round(h * scale))
    rgb_i = Image.fromarray(rgb).resize((nw, nh), Image.Resampling.LANCZOS)
    gm_i = Image.fromarray((gm * 255).astype(np.uint8)).resize((nw, nh), Image.Resampling.NEAREST)
    canvas = np.zeros((size, size, 3), np.uint8)
    mcanvas = np.zeros((size, size), np.float32)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = np.asarray(rgb_i)
    mcanvas[oy : oy + nh, ox : ox + nw] = np.asarray(gm_i, np.float32) / 255.0
    bcanvas = None
    if bm is not None:
        bm_i = Image.fromarray((bm * 255).astype(np.uint8)).resize(
            (nw, nh), Image.Resampling.NEAREST
        )
        bcanvas = np.zeros((size, size), np.float32)
        bcanvas[oy : oy + nh, ox : ox + nw] = np.asarray(bm_i, np.float32) / 255.0
    return canvas.astype(np.float32) / 255.0, mcanvas, bcanvas


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _fill_outside(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    valid = mask > 0.5
    if not valid.any():
        return out
    med = np.median(out[valid], axis=0)
    out[~valid] = med
    return out


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not good.any():
        return float("nan")
    v, w = values[good], weights[good]
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, q * c[-1], side="left")])


def _ridge_metrics(y: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    size = y.shape[0]
    small = _blur01(y, max(0.8, size / 384.0))
    broad = _blur01(y, max(3.0, size / 64.0))
    z = small - broad
    gy, gx = np.gradient(z)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    gxy = (gxy + gyx) * 0.5
    tr = gxx + gyy
    disc = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4 * gxy**2, 0))
    e1, e2 = (tr + disc) * 0.5, (tr - disc) * 0.5
    a1, a2 = np.abs(e1), np.abs(e2)
    strong, weak = np.maximum(a1, a2), np.minimum(a1, a2)
    line = (strong - weak) / (strong + weak + 1e-8)
    interior = _morph(mask, max(2, round(size / 128.0))) > 0.5
    vals = strong[interior]
    if vals.size < 64:
        return {
            k: float("nan")
            for k in (
                "fold_contrast",
                "wrinkle_density",
                "wrinkle_strength_p90",
                "wrinkle_orientation_entropy",
                "wrinkle_bend_per_100px",
            )
        }
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    threshold = med + 2.5 * max(mad, 1e-7)
    active = interior & (strong > threshold) & (line > 0.35)
    theta = 0.5 * np.arctan2(2 * gxy, gxx - gyy)
    # Orientation is pi-periodic.  Differentiating cos(2t),sin(2t) avoids
    # artificial jumps at -pi/2 and +pi/2.
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
    dc_y, dc_x = np.gradient(c2)
    ds_y, ds_x = np.gradient(s2)
    bend = 0.5 * np.sqrt(dc_x**2 + dc_y**2 + ds_x**2 + ds_y**2)
    weights = np.where(active, strong, 0.0)
    hist, _ = np.histogram(
        theta[active], bins=12, range=(-math.pi / 2, math.pi / 2), weights=strong[active]
    )
    hp = hist / max(float(hist.sum()), 1e-12)
    nz = hp > 0
    entropy = float(-(hp[nz] * np.log(hp[nz])).sum() / math.log(len(hist))) if nz.any() else 0.0
    return {
        "fold_contrast": float(np.sqrt(np.mean(z[interior] ** 2))),
        "wrinkle_density": float(active[interior].mean()),
        "wrinkle_strength_p90": float(np.quantile(vals, 0.90)),
        "wrinkle_orientation_entropy": entropy,
        "wrinkle_bend_per_100px": _weighted_quantile(bend, weights, 0.50) * 100.0,
    }


def _spectrum_metrics(y: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    n = y.shape[0]
    valid = mask > 0.5
    if valid.sum() < 64:
        return {
            k: float("nan")
            for k in (
                "macro_energy_ratio",
                "meso_energy_ratio",
                "micro_energy_ratio",
                "fold_frequency_cpg",
            )
        }
    # A broad residual removes illumination while retaining garment-scale folds.
    residual = y - _blur01(y, max(4.0, n / 24.0))
    soft = _blur01(_morph(mask, max(2, round(n / 128.0))), max(1.0, n / 192.0))
    win = np.hanning(n)
    window = soft * win[:, None] * win[None, :]
    z = residual * window
    z -= z.sum() / max(window.sum(), 1e-8) * window
    power = np.abs(np.fft.rfft2(z)) ** 2
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.rfftfreq(n)[None, :]
    cycles = np.sqrt(fx**2 + fy**2) * n
    # Cycles per normalised garment box.  Macro/meso represent drape; micro is
    # mostly yarn/codec texture and is kept separate.
    bands = ((2, 8), (8, 32), (32, min(96, n / 2)))
    energy = []
    for lo, hi in bands:
        sel = (cycles >= lo) & (cycles < hi)
        energy.append(float(power[sel].sum()))
    total = max(sum(energy), 1e-12)
    fold_sel = (cycles >= 2) & (cycles < 32)
    centroid = float((cycles[fold_sel] * power[fold_sel]).sum() / max(power[fold_sel].sum(), 1e-12))
    return {
        "macro_energy_ratio": energy[0] / total,
        "meso_energy_ratio": energy[1] / total,
        "micro_energy_ratio": energy[2] / total,
        "fold_frequency_cpg": centroid,
    }


def _hem_metrics(mask: np.ndarray) -> dict[str, float]:
    box = _bbox(mask)
    if box is None:
        return {
            k: float("nan")
            for k in ("hem_curvature", "hem_sag_ratio", "hem_roughness_ratio", "hem_coverage")
        }
    x0, y0, x1, y1 = box
    h, w = y1 - y0, x1 - x0
    xa, xb = x0 + round(w * 0.18), x0 + round(w * 0.82)
    xs, ys = [], []
    binary = mask > 0.5
    for x in range(xa, xb):
        yy = np.flatnonzero(binary[:, x])
        if yy.size:
            xs.append(x)
            ys.append(int(yy[-1]))
    coverage = len(xs) / max(xb - xa, 1)
    if len(xs) < 12 or coverage < 0.55:
        return {
            "hem_curvature": float("nan"),
            "hem_sag_ratio": float("nan"),
            "hem_roughness_ratio": float("nan"),
            "hem_coverage": float(coverage),
        }
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    window = max(3, round(len(ys) / 30))
    if window % 2 == 0:
        window += 1
    smooth = np.convolve(
        np.pad(ys, window // 2, mode="edge"), np.ones(window) / window, mode="valid"
    )
    xn = (xs - xs.mean()) / max((xs.max() - xs.min()) * 0.5, 1.0)
    yn = (smooth - smooth.mean()) / max(h, 1)
    coef = np.polyfit(xn, yn, 2)
    fit = np.polyval(coef, xn)
    center = float(np.polyval(coef, 0.0))
    edge = float((np.polyval(coef, -0.8) + np.polyval(coef, 0.8)) * 0.5)
    return {
        "hem_curvature": float(2 * coef[0]),
        "hem_sag_ratio": center - edge,
        "hem_roughness_ratio": float(np.sqrt(np.mean((yn - fit) ** 2))),
        "hem_coverage": float(coverage),
    }


def _silhouette_metrics(mask: np.ndarray, body_mask: np.ndarray | None) -> dict[str, float | None]:
    box = _bbox(mask)
    if box is None:
        return {
            "silhouette_flare_ratio": float("nan"),
            "silhouette_width_cv": float("nan"),
            "body_gap_proxy": None,
        }
    x0, y0, x1, y1 = box
    h = y1 - y0
    rows: list[tuple[float, float, float, float]] = []
    binary = mask > 0.5
    for y in range(y0 + round(0.20 * h), y0 + round(0.92 * h)):
        xx = np.flatnonzero(binary[y])
        if xx.size:
            rows.append(((y - y0) / max(h, 1), float(y), float(xx[0]), float(xx[-1])))
    if len(rows) < 8:
        return {
            "silhouette_flare_ratio": float("nan"),
            "silhouette_width_cv": float("nan"),
            "body_gap_proxy": None,
        }
    rr = np.asarray(rows)
    widths = rr[:, 3] - rr[:, 2] + 1
    mid = widths[(rr[:, 0] >= 0.38) & (rr[:, 0] < 0.60)]
    bot = widths[(rr[:, 0] >= 0.72) & (rr[:, 0] < 0.90)]
    flare = (
        float(np.median(bot) / max(np.median(mid), 1e-8) - 1)
        if len(mid) and len(bot)
        else float("nan")
    )
    gap: float | None = None
    if body_mask is not None:
        gaps = []
        b = body_mask > 0.5
        for row in rr:
            y = int(row[1])
            gx, bx = np.flatnonzero(binary[y]), np.flatnonzero(b[y])
            if gx.size and bx.size:
                outside = max(0.0, bx[0] - gx[0]) + max(0.0, gx[-1] - bx[-1])
                gaps.append(outside / max(gx[-1] - gx[0] + 1, 1))
        if gaps:
            gap = float(np.median(gaps))
    return {
        "silhouette_flare_ratio": flare,
        "silhouette_width_cv": float(np.std(widths) / max(np.mean(widths), 1e-8)),
        "body_gap_proxy": gap,
    }


def measure_image(
    image: Image.Image,
    mask: np.ndarray,
    body_mask: np.ndarray | None = None,
    size: int = 384,
) -> dict[str, float | None]:
    """Measure one image.  All frequency values use a normalised garment box."""
    if mask.shape != (image.height, image.width):
        mask = (
            np.asarray(
                Image.fromarray((mask * 255).astype(np.uint8)).resize(
                    image.size, Image.Resampling.NEAREST
                ),
                np.float32,
            )
            / 255.0
        )
    if body_mask is not None and body_mask.shape != mask.shape:
        body_mask = (
            np.asarray(
                Image.fromarray((body_mask * 255).astype(np.uint8)).resize(
                    image.size, Image.Resampling.NEAREST
                ),
                np.float32,
            )
            / 255.0
        )
    original_box = _bbox(mask)
    if original_box is None:
        raise ValueError("empty garment mask")
    x0, y0, x1, y1 = original_box
    garment_h = y1 - y0
    garment_w = x1 - x0
    rgb, gm, bm = _normalise_roi(image, mask, body_mask, size)
    rgb = _fill_outside(rgb, gm)
    y = _luminance(rgb)
    out: dict[str, float | None] = {}
    out.update(_ridge_metrics(y, gm))
    out.update(_spectrum_metrics(y, gm))
    out.update(_hem_metrics(gm))
    out.update(_silhouette_metrics(gm, bm))
    # Keep these in the original image frame.  ROI-normalised metrics cannot
    # detect an overall length change because the crop scales every garment
    # back to the same box.
    out["garment_height_image_ratio"] = float(garment_h / max(image.height, 1))
    out["garment_width_image_ratio"] = float(garment_w / max(image.width, 1))
    out["garment_aspect_ratio"] = float(garment_h / max(garment_w, 1))
    out["hem_y_image_ratio"] = float(y1 / max(image.height, 1))
    out["mask_area_ratio"] = float((mask > 0.5).mean())
    out["interior_area_ratio"] = float((_morph(gm, max(2, round(size / 128))) > 0.5).mean())
    return out


def measurement_noise(
    image: Image.Image, mask: np.ndarray, body_mask: np.ndarray | None = None, size: int = 384
) -> list[dict[str, float | None]]:
    """Deterministic codec/resampling/mask-jitter repeats for a noise floor."""
    base_size = image.size
    variants: list[tuple[Image.Image, np.ndarray, np.ndarray | None]] = []
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=95, subsampling=0)
    buf.seek(0)
    variants.append((Image.open(buf).convert("RGB"), mask, body_mask))
    down = image.resize(
        (max(2, round(image.width * 0.97)), max(2, round(image.height * 0.97))),
        Image.Resampling.LANCZOS,
    ).resize(base_size, Image.Resampling.LANCZOS)
    variants.append((down, mask, body_mask))
    variants.append((image, _morph(mask, 1), body_mask))
    variants.append((image, _morph(mask, 1, dilate=True), body_mask))
    return [measure_image(im, m, bm, size=size) for im, m, bm in variants]


def _entry(spec: Any) -> dict[str, Any] | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        return {"image": spec}
    return spec


def _existing_entry(spec: Any) -> dict[str, Any] | None:
    e = _entry(spec)
    if not e or not e.get("image") or not e.get("mask"):
        return None
    return e if Path(e["image"]).exists() and Path(e["mask"]).exists() else None


def _preservation(
    source: Image.Image, generated: Image.Image, change_mask: np.ndarray | None
) -> dict[str, float]:
    if generated.size != source.size:
        generated = generated.resize(source.size, Image.Resampling.LANCZOS)
    a, b = np.asarray(source, np.int16), np.asarray(generated, np.int16)
    d = np.abs(a - b).max(2)
    outside = np.ones(d.shape, bool) if change_mask is None else change_mask <= 0.0
    if not outside.any():
        return {
            "outside_mae": float("nan"),
            "outside_changed_fraction": float("nan"),
            "outside_max_delta": float("nan"),
        }
    return {
        "outside_mae": float(d[outside].mean()),
        "outside_changed_fraction": float((d[outside] > 0).mean()),
        "outside_max_delta": float(d[outside].max()),
    }


def _balanced_summary(values: Iterable[tuple[str, float]]) -> dict[str, float | int | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    for group, value in values:
        if np.isfinite(value):
            groups[group].append(float(value))
    per_group = [float(np.median(v)) for v in groups.values() if v]
    if not per_group:
        return {"n_products": 0, "median": None, "p90": None}
    return {
        "n_products": len(per_group),
        "median": float(np.median(per_group)),
        "p90": float(np.quantile(per_group, 0.90)),
    }


def evaluate_manifest(manifest_path: Path, size: int = 384) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", manifest if isinstance(manifest, list) else [])
    records: list[dict[str, Any]] = []
    noise_delta: dict[str, list[tuple[str, float]]] = defaultdict(list)
    missing_ground_truth: list[dict[str, str]] = []
    for i, sample in enumerate(samples, 1):
        sid = sample["sample_id"]
        product = sample.get("product_id", sid)
        gt = _existing_entry(sample.get("ground_truth"))
        if gt is None:
            raw_gt = _entry(sample.get("ground_truth")) or {}
            missing_ground_truth.append(
                {
                    "sample_id": sid,
                    "image": str(raw_gt.get("image", "")),
                    "mask": str(raw_gt.get("mask", "")),
                }
            )
            continue
        gt_im = _as_rgb(gt["image"])
        gt_mask = _as_mask(gt["mask"], gt_im.size)
        body = (
            _as_mask(gt["body_mask"], gt_im.size)
            if gt.get("body_mask") and Path(gt["body_mask"]).exists()
            else None
        )
        gt_metrics = measure_image(gt_im, gt_mask, body, size)
        records.append(
            {
                "sample_id": sid,
                "product_id": product,
                "variant": "ground_truth",
                "metrics": gt_metrics,
            }
        )
        for rep in measurement_noise(gt_im, gt_mask, body, size):
            for key, value in gt_metrics.items():
                if key in METRIC_DIRECTIONS and _finite(value) and _finite(rep.get(key)):
                    noise_delta[key].append((product, abs(float(rep[key]) - float(value))))
        for variant, raw_spec in sample.get("variants", {}).items():
            spec = _existing_entry(raw_spec)
            if spec is None:
                continue
            im = _as_rgb(spec["image"])
            mask = _as_mask(spec["mask"], im.size)
            bm = (
                _as_mask(spec["body_mask"], im.size)
                if spec.get("body_mask") and Path(spec["body_mask"]).exists()
                else None
            )
            metrics = measure_image(im, mask, bm, size)
            change = None
            if spec.get("change_mask") and Path(spec["change_mask"]).exists():
                change = _as_mask(spec["change_mask"], gt_im.size)
            records.append(
                {
                    "sample_id": sid,
                    "product_id": product,
                    "variant": variant,
                    "metrics": metrics,
                    "preservation": _preservation(gt_im, im, change),
                }
            )
        print(f"[{i}/{len(samples)}] {sid}", flush=True)

    if not records:
        preview = missing_ground_truth[:3]
        raise ValueError(
            f"manifest produced zero evaluable records; missing ground truth "
            f"for {len(missing_ground_truth)} samples, examples={preview}"
        )

    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        by_sample[r["sample_id"]][r["variant"]] = r
    variants = sorted({r["variant"] for r in records if r["variant"] != "ground_truth"})
    comparisons: dict[str, Any] = {}
    for variant in variants:
        metrics_summary = {}
        for metric in METRIC_DIRECTIONS:
            vals = []
            signed = []
            for sid, rr in by_sample.items():
                if "ground_truth" not in rr or variant not in rr:
                    continue
                g, v = rr["ground_truth"]["metrics"].get(metric), rr[variant]["metrics"].get(metric)
                if _finite(g) and _finite(v):
                    prod = rr[variant]["product_id"]
                    vals.append((prod, abs(float(v) - float(g))))
                    signed.append((prod, float(v) - float(g)))
            metrics_summary[metric] = {
                "absolute_error": _balanced_summary(vals),
                "signed_bias": _balanced_summary(signed),
            }
        preserve = {}
        for key in ("outside_mae", "outside_changed_fraction", "outside_max_delta"):
            preserve[key] = _balanced_summary(
                (r["product_id"], float(r["preservation"][key]))
                for r in records
                if r["variant"] == variant
                and r.get("preservation")
                and _finite(r["preservation"].get(key))
            )
        comparisons[variant] = {"metrics": metrics_summary, "preservation": preserve}

    # Same garment, different real photos: this is real pose/view variation,
    # explicitly not measurement noise.
    gt_by_product: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["variant"] == "ground_truth":
            gt_by_product[r["product_id"]].append(r["metrics"])
    pose: dict[str, Any] = {}
    for metric in METRIC_DIRECTIONS:
        per_product = []
        for product, rows in gt_by_product.items():
            vals = [float(r[metric]) for r in rows if _finite(r.get(metric))]
            diffs = [
                abs(vals[a] - vals[b]) for a in range(len(vals)) for b in range(a + 1, len(vals))
            ]
            if diffs:
                per_product.append((product, float(np.median(diffs))))
        pose[metric] = _balanced_summary(per_product)

    return {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "normalised_size": size,
        "caveats": {
            "metrics_are_image_space_proxies": True,
            "body_gap_requires_body_without_garment_mask": True,
            "pose_variation_is_not_measurement_noise": True,
        },
        "counts": {
            "manifest_samples": len(samples),
            "samples": len(by_sample),
            "records": len(records),
            "variants": variants,
            "missing_ground_truth": len(missing_ground_truth),
        },
        "missing_ground_truth": missing_ground_truth,
        "measurement_noise": {k: _balanced_summary(v) for k, v in noise_delta.items()},
        "pose_variation": pose,
        "comparisons": comparisons,
        "records": records,
    }


def _write_csv(result: dict[str, Any], path: Path) -> None:
    metric_names = sorted({k for r in result["records"] for k in r["metrics"]})
    preserve_names = ["outside_mae", "outside_changed_fraction", "outside_max_delta"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f, fieldnames=["sample_id", "product_id", "variant"] + metric_names + preserve_names
        )
        w.writeheader()
        for r in result["records"]:
            row = {
                "sample_id": r["sample_id"],
                "product_id": r["product_id"],
                "variant": r["variant"],
            }
            row.update(r["metrics"])
            row.update(r.get("preservation", {}))
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()
    result = evaluate_manifest(args.manifest, args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(result, args.out.with_suffix(".csv"))
    print(f"\nJSON: {args.out}\nCSV : {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
