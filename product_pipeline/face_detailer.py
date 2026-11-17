"""Re-render the face region of a generated base at high resolution.

Measured problem (identity_score.py): the identity LoRA knows the face, but
output framing decides whether it can be expressed.

    upper-body portrait (~250px face)  cosine 0.467   (real baseline 0.499)
    half body           (~150px face)  cosine 0.341
    full body           (~90px  face)  cosine 0.209-0.272

Commercial cuts need full body, so the face has to be recovered after the fact.
This crops the face, re-renders it at 1024 with the same identity LoRA via
img2img, and pastes it back under a feathered mask.

Placement in the pipeline matters: this runs on the BASE, before FASHN. The
garment stage never sees a half-finished face, and the v13 hard composite
afterwards protects these pixels exactly, so the detailed face survives VTON.

Denoise strength is the whole tradeoff -- too low and nothing changes, too high
and it invents a different person. Sweep it and score, do not guess.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLImg2ImgPipeline
from PIL import Image, ImageFilter

MODEL = Path(r"C:\Dev\StabilityMatrix\Data\Models\StableDiffusion\RealVisXL_V5.0_fp16.safetensors")
YUNET = Path(r"fashion_pipeline/models/opencv_face/face_detection_yunet_2023mar.onnx")

NEGATIVE = (
    "blurry, low resolution, jpeg artifacts, deformed face, distorted features, "
    "extra eyes, asymmetric eyes, plastic skin, airbrushed, waxy, cartoon, "
    "illustration, 3d render, watermark, text"
)


def detect_face(img: Image.Image, pad: float):
    """Return a padded square face box, or None."""
    arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.6, 0.3, 5000)
    det.setInputSize((w, h))
    ok, faces = det.detect(arr)
    if faces is None or len(faces) == 0:
        return None
    x, y, fw, fh = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0][:4]
    cx, cy = x + fw / 2, y + fh / 2
    side = max(fw, fh) * pad
    side = min(side, w, h)
    left = int(max(0, min(cx - side / 2, w - side)))
    top = int(max(0, min(cy - side / 2, h - side)))
    return left, top, int(left + side), int(top + side), float(fh)


def feather_mask(size: tuple[int, int], inset: float, blur: float) -> Image.Image:
    """White in the middle, fading to black at the crop edge, so the paste has no seam."""
    w, h = size
    m = Image.new("L", (w, h), 0)
    pad_x, pad_y = int(w * inset), int(h * inset)
    m.paste(255, (pad_x, pad_y, w - pad_x, h - pad_y))
    return m.filter(ImageFilter.GaussianBlur(blur))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--lora-scale", type=float, default=0.9)
    ap.add_argument("--token", default="hgnst_model man")
    ap.add_argument("--strengths", default="0.35,0.5")
    ap.add_argument("--pad", type=float, default=2.2, help="face box expansion factor")
    ap.add_argument("--work-size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--inset", type=float, default=0.10)
    ap.add_argument("--blur", type=float, default=18.0)
    ap.add_argument(
        "--canvas-scale",
        type=float,
        default=1.0,
        help=(
            "Upscale the complete generated base before face detection/detailing. "
            "This preserves more of the 1024px face render when it is pasted back "
            "into a full-body commercial frame. 1.0 keeps the legacy behavior."
        ),
    )
    args = ap.parse_args()

    strengths = [float(s) for s in args.strengths.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for p in args.images:
        paths.extend(sorted(p.glob("*.png"))) if p.is_dir() else paths.append(p)

    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        str(MODEL), torch_dtype=torch.float16, add_watermarker=False
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"
    )
    pipe.load_lora_weights(str(args.lora))
    pipe.fuse_lora(lora_scale=args.lora_scale)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    prompt = (
        f"{args.token}, close up face portrait, sharp focus, natural skin texture, "
        f"detailed eyes, photograph"
    )

    records = []
    for path in paths:
        base = Image.open(path).convert("RGB")
        source_size = base.size
        if args.canvas_scale != 1.0:
            if args.canvas_scale <= 0:
                raise ValueError("--canvas-scale must be greater than zero")
            scaled_size = tuple(max(1, round(v * args.canvas_scale)) for v in base.size)
            base = base.resize(scaled_size, Image.Resampling.LANCZOS)
        box = detect_face(base, args.pad)
        if box is None:
            print(f"[skip] no face detected: {path.name}")
            continue
        x0, y0, x1, y1, face_px = box
        crop = base.crop((x0, y0, x1, y1))
        up = crop.resize((args.work_size, args.work_size), Image.LANCZOS)

        for strength in strengths:
            gen = torch.Generator(device="cpu").manual_seed(args.seed)
            fixed = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE,
                image=up,
                strength=strength,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=gen,
            ).images[0]

            back = fixed.resize(crop.size, Image.LANCZOS)
            mask = feather_mask(crop.size, args.inset, args.blur * args.canvas_scale)
            out = base.copy()
            out.paste(back, (x0, y0), mask)

            name = f"{path.stem}_detail{strength}"
            out.save(args.out_dir / f"{name}.png")
            records.append(
                {
                    "source": str(path),
                    "output": str(args.out_dir / f"{name}.png"),
                    "face_box": [x0, y0, x1, y1],
                    "orig_face_height_px": round(face_px, 1),
                    "source_size": list(source_size),
                    "canvas_size": list(base.size),
                    "canvas_scale": args.canvas_scale,
                    "work_size": args.work_size,
                    "strength": strength,
                    "lora": str(args.lora),
                    "lora_scale": args.lora_scale,
                }
            )
            print(f"[ok] {name}  (face was {face_px:.0f}px -> rendered at {args.work_size})")

    (args.out_dir / "detail_manifest.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {len(records)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
